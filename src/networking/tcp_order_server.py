"""
src/networking/tcp_order_server.py
=====================================
Minimal async TCP server for order entry with low-latency focus.

HFT relevance
-------------
Order entry protocols (FIX, OUCH, BATS BOE) run over TCP.  TCP provides
reliable, ordered delivery — essential for orders where duplication or
reordering would be catastrophic.

Latency considerations for TCP order servers:
  1. TCP_NODELAY: Disable Nagle's algorithm (40 ms buffering by default!)
  2. SO_REUSEADDR: Fast rebind after crash
  3. asyncio: Handle many connections with one event loop thread
  4. Avoid per-message allocation: use pre-allocated byte buffers

This server implements a simple echo-order protocol:
  - Client sends: 4-byte length prefix + message body
  - Server echoes the same message back (simulating ACK)

In production, the server would:
  - Validate the order
  - Route to the matching engine
  - Return an execution report

TODO:
  - Add TLS (required for some venues)
  - Add sequence numbers for de-duplication
  - Measure actual RTT vs the bench_latency_test.py loopback results
"""

from __future__ import annotations

import asyncio
import struct
import time
from typing import Final


HOST: Final = "127.0.0.1"
PORT: Final = 19877
HEADER_FMT: Final = struct.Struct("!I")  # 4-byte big-endian message length


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class OrderServer:
    """Async TCP order entry server.

    Handles multiple concurrent connections using asyncio.  Each
    connection gets its own reader/writer pair; the event loop multiplexes
    them without OS thread overhead.

    Args:
        host: Bind address.
        port: Bind port.
    """

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._connections = 0
        self._messages = 0

    async def start(self) -> None:
        """Start the TCP server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
            reuse_address=True,
        )
        addr = self._server.sockets[0].getsockname()  # type: ignore[index]
        print(f"OrderServer listening on {addr[0]}:{addr[1]}")

    async def stop(self) -> None:
        """Gracefully stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one client connection.

        Reads length-prefixed messages and echoes them back (ACK).
        Uses `TCP_NODELAY` to ensure ACKs are sent immediately without
        Nagle buffering.

        Args:
            reader: Async stream reader for inbound data.
            writer: Async stream writer for outbound data.
        """
        peer = writer.get_extra_info("peername")
        self._connections += 1

        # Apply TCP_NODELAY on the underlying socket
        sock = writer.get_extra_info("socket")
        if sock:
            import socket
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            while True:
                # Read the 4-byte length header
                header_bytes = await reader.readexactly(HEADER_FMT.size)
                (msg_len,) = HEADER_FMT.unpack(header_bytes)

                if msg_len == 0:
                    break  # Client signalled end of session

                # Read the message body
                body = await reader.readexactly(msg_len)
                self._messages += 1

                # Echo back: length prefix + body
                # In production: validate, route, return execution report
                response = header_bytes + body
                writer.write(response)
                await writer.drain()

        except asyncio.IncompleteReadError:
            pass  # Client disconnected cleanly
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    @property
    def stats(self) -> dict[str, int]:
        """Return server statistics.

        Returns:
            Dict with total_connections and total_messages.
        """
        return {
            "total_connections": self._connections,
            "total_messages": self._messages,
        }


# ---------------------------------------------------------------------------
# Client (for testing)
# ---------------------------------------------------------------------------

class OrderClient:
    """Minimal async TCP client for benchmarking the order server.

    Args:
        host: Server host.
        port: Server port.
    """

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Connect to the order server."""
        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port
        )
        # Disable Nagle on client side too
        sock = self._writer.get_extra_info("socket")
        if sock:
            import socket
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    async def send_order(self, payload: bytes) -> bytes:
        """Send an order and receive the ACK.

        Args:
            payload: Order message bytes.

        Returns:
            Echo response bytes (should equal payload).
        """
        assert self._writer and self._reader
        msg_len = len(payload)
        frame = HEADER_FMT.pack(msg_len) + payload
        self._writer.write(frame)
        await self._writer.drain()

        header = await self._reader.readexactly(HEADER_FMT.size)
        (resp_len,) = HEADER_FMT.unpack(header)
        body = await self._reader.readexactly(resp_len)
        return body

    async def close(self) -> None:
        """Close the connection."""
        if self._writer:
            # Send zero-length message as disconnect signal
            self._writer.write(HEADER_FMT.pack(0))
            await self._writer.drain()
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Demo / benchmark
# ---------------------------------------------------------------------------

async def _run_demo() -> None:
    """Run a round-trip latency test: N orders → server → ACK."""
    server = OrderServer()
    await server.start()

    client = OrderClient()
    await client.connect()

    n = 1_000
    payload = b"NEW_ORDER|AAPL|BUY|100|99.50"
    rtts: list[int] = []

    for _ in range(n):
        t0 = time.perf_counter_ns()
        ack = await client.send_order(payload)
        rtts.append(time.perf_counter_ns() - t0)
        assert ack == payload

    await client.close()
    await server.stop()

    rtts.sort()
    p50 = rtts[len(rtts) // 2]
    p99 = rtts[int(len(rtts) * 0.99)]
    print(f"TCP order RTT ({n} orders):")
    print(f"  min={rtts[0]//1000} µs  p50={p50//1000} µs  p99={p99//1000} µs  max={rtts[-1]//1000} µs")
    print(f"  Server stats: {server.stats}")


if __name__ == "__main__":
    print("=== Async TCP Order Server Demo ===\n")
    asyncio.run(_run_demo())
