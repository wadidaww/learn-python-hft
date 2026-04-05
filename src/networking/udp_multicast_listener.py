"""
src/networking/udp_multicast_listener.py
==========================================
UDP multicast receiver for market-data feeds.

HFT relevance
-------------
Major exchange market-data feeds (ITCH, OPRA, CME MDP3) are distributed
via UDP multicast.  Multicast allows one sender to reach thousands of
receivers without unicast duplication at the switch level.

Key points:
  - Packet loss is possible; sequence gaps must be detected and filled
    via a unicast retransmit channel (e.g. NASDAQ GapFill).
  - `SO_RCVBUF` must be large enough to absorb bursts without drop.
  - Joining the right multicast group requires `IP_ADD_MEMBERSHIP`.
  - CPU affinity and NUMA awareness are critical for sub-10 µs processing.

This module implements a basic multicast listener that can be used for
testing with a local multicast sender (see the __main__ block for both
sender and receiver).

TODO:
  - Add sequence gap detection and retransmit request.
  - Add a timestamp comparison: kernel SO_TIMESTAMPNS vs perf_counter_ns.
  - Test with `tcpdump -i lo udp and host 239.1.1.1` to verify packets.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Callable


# ---------------------------------------------------------------------------
# Multicast listener
# ---------------------------------------------------------------------------

MCAST_GROUP = "239.1.1.1"
MCAST_PORT = 5007
BUFFER_SIZE = 65_536  # 64 KB — typical for a full MTU-sized UDP datagram


def create_multicast_socket(
    group: str = MCAST_GROUP,
    port: int = MCAST_PORT,
    rcvbuf_bytes: int = 8 * 1024 * 1024,  # 8 MB receive buffer
) -> socket.socket:
    """Create and configure a UDP socket joined to a multicast group.

    Args:
        group:       Multicast group IP address.
        port:        UDP port to bind.
        rcvbuf_bytes: Receive buffer size; increase for bursty feeds.

    Returns:
        Bound, multicast-joined UDP socket ready to receive.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Allow multiple processes to bind the same port (required for multicast)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Tune receive buffer to absorb market-data bursts
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf_bytes)

    sock.bind(("", port))

    # Join multicast group on all interfaces
    mreq = struct.pack(
        "4s4s",
        socket.inet_aton(group),
        socket.inet_aton("0.0.0.0"),
    )
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    return sock


class MulticastListener:
    """Non-blocking UDP multicast listener.

    Receives datagrams in a background thread and dispatches to a
    user-supplied callback.

    Args:
        group:    Multicast group IP.
        port:     UDP port.
        callback: Callable invoked for each received datagram.
                  Signature: ``(data: bytes, addr: tuple) -> None``
        rcvbuf:   Kernel receive buffer size in bytes.

    Example::

        def on_tick(data: bytes, addr: tuple) -> None:
            print(f"Received {len(data)} bytes from {addr}")

        listener = MulticastListener(callback=on_tick)
        listener.start()
        time.sleep(5)
        listener.stop()
    """

    def __init__(
        self,
        group: str = MCAST_GROUP,
        port: int = MCAST_PORT,
        callback: Callable[[bytes, tuple[str, int]], None] | None = None,
        rcvbuf: int = 8 * 1024 * 1024,
    ) -> None:
        self._group = group
        self._port = port
        self._callback = callback or (lambda data, addr: None)
        self._rcvbuf = rcvbuf
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._packet_count = 0
        self._byte_count = 0

    def start(self) -> None:
        """Start the listener thread."""
        self._sock = create_multicast_socket(self._group, self._port, self._rcvbuf)
        self._sock.settimeout(0.1)  # non-blocking with timeout for clean shutdown
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the listener and close the socket."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()

    def _recv_loop(self) -> None:
        """Main receive loop — runs in background thread."""
        assert self._sock is not None
        while self._running:
            try:
                data, addr = self._sock.recvfrom(BUFFER_SIZE)
                self._packet_count += 1
                self._byte_count += len(data)
                self._callback(data, addr)
            except socket.timeout:
                continue
            except OSError:
                break

    @property
    def stats(self) -> dict[str, int]:
        """Return receive statistics.

        Returns:
            Dict with packet_count and byte_count.
        """
        return {
            "packet_count": self._packet_count,
            "byte_count": self._byte_count,
        }


# ---------------------------------------------------------------------------
# Simple multicast sender (for testing)
# ---------------------------------------------------------------------------

def multicast_sender(
    group: str = MCAST_GROUP,
    port: int = MCAST_PORT,
    n_packets: int = 100,
    interval_s: float = 0.01,
    payload_size: int = 64,
) -> None:
    """Send test UDP multicast packets.

    Args:
        group:        Multicast group IP.
        port:         UDP port.
        n_packets:    Number of packets to send.
        interval_s:   Delay between packets.
        payload_size: Bytes per packet.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # TTL=1: multicast stays on local subnet
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack("b", 1))

    payload = b"M" * payload_size
    for seq in range(n_packets):
        # Prepend a 4-byte sequence number for gap detection
        msg = struct.pack("!I", seq) + payload
        sock.sendto(msg, (group, port))
        if interval_s > 0:
            time.sleep(interval_s)

    sock.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    received: list[tuple[int, bytes]] = []

    def on_packet(data: bytes, addr: tuple[str, int]) -> None:
        (seq,) = struct.unpack_from("!I", data, 0)
        received.append((seq, data))
        if len(received) % 10 == 0:
            print(f"  Received {len(received)} packets (last seq={seq})")

    print("=== UDP Multicast Demo ===")
    print(f"  Group: {MCAST_GROUP}:{MCAST_PORT}")
    print("  Starting listener...")

    listener = MulticastListener(callback=on_packet)
    listener.start()

    # Give the socket time to join the group
    time.sleep(0.1)

    print("  Sending 50 test packets...")
    sender_thread = threading.Thread(
        target=multicast_sender,
        kwargs={"n_packets": 50, "interval_s": 0.005},
        daemon=True,
    )
    sender_thread.start()
    sender_thread.join()

    time.sleep(0.5)  # allow last packets to be received
    listener.stop()

    stats = listener.stats
    print(f"\n  Packets received: {stats['packet_count']}")
    print(f"  Bytes received:   {stats['byte_count']:,}")

    if len(received) > 1:
        seqs = [r[0] for r in received]
        gaps = [seqs[i+1] - seqs[i] for i in range(len(seqs)-1) if seqs[i+1] - seqs[i] > 1]
        print(f"  Sequence gaps:    {len(gaps)} (packet loss detected)")
