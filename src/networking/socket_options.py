"""
src/networking/socket_options.py
==================================
Utility module to configure low-latency socket options.

HFT relevance
-------------
The Linux kernel TCP/IP stack adds latency even on localhost:
  - Nagle's algorithm (TCP_NODELAY=0): buffers small writes, adding 40 ms delay.
  - Interrupt coalescing: batches interrupts, increasing latency up to 50 µs.
  - Socket buffer sizing: undersized buffers cause drops; oversized add latency.

This module provides helpers to set the right options for:
  - Order entry TCP connections (low latency, small messages)
  - Market data UDP feeds (high throughput, can lose packets)

Kernel bypass note:
  Technologies like Solarflare OpenOnload, DPDK, and RDMA bypass the
  kernel TCP/IP stack entirely.  RTT drops from ~20 µs to ~1 µs.
  Python cannot directly use these (they require C driver integration),
  but SO_BUSY_POLL (Linux 3.11+) is a partial kernel-level equivalent
  that reduces interrupt latency without full bypass.
"""

from __future__ import annotations

import socket
import struct
import sys


# ---------------------------------------------------------------------------
# Low-latency TCP options
# ---------------------------------------------------------------------------

def configure_tcp_low_latency(sock: socket.socket) -> None:
    """Apply low-latency options to a TCP socket.

    Options set:
      - TCP_NODELAY: Disable Nagle's algorithm.  Critical for small messages
        like order entry packets.  Without this, the kernel may buffer a
        64-byte order for up to 40 ms waiting to fill an MSS-sized segment.
      - SO_KEEPALIVE: Enable TCP keepalive to detect dead connections.
      - SO_REUSEADDR: Allow immediate bind after crash (avoids TIME_WAIT).

    Args:
        sock: A TCP socket (SOCK_STREAM).
    """
    # Disable Nagle — most important option for HFT order entry
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Allow address reuse (avoids "Address already in use" after crash)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Enable keepalive probes
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    # Linux-specific: set socket priority (0=lowest, 6=highest non-root, 7=root only)
    if sys.platform == "linux":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)


def configure_tcp_keepalive(
    sock: socket.socket,
    idle_s: int = 10,
    interval_s: int = 5,
    probes: int = 3,
) -> None:
    """Configure TCP keepalive timing (Linux only).

    Keepalive probes detect dead peer connections.  In HFT, a dead
    connection to an exchange is a critical event requiring immediate
    reconnection and position reconciliation.

    Args:
        sock:       TCP socket with SO_KEEPALIVE already enabled.
        idle_s:     Seconds of idle before probes start (TCP_KEEPIDLE).
        interval_s: Seconds between probes (TCP_KEEPINTVL).
        probes:     Number of failed probes before declaring dead (TCP_KEEPCNT).
    """
    if sys.platform != "linux":
        return

    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle_s)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_s)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, probes)


def configure_socket_buffers(
    sock: socket.socket,
    rcvbuf_bytes: int = 4 * 1024 * 1024,
    sndbuf_bytes: int = 1 * 1024 * 1024,
) -> None:
    """Set socket send and receive buffer sizes.

    For market-data UDP feeds, a large receive buffer absorbs bursts:
      - ITCH 5.0 (NASDAQ): ~500k msgs/s peak → set rcvbuf >= 8 MB
      - Order entry TCP: smaller buffer is fine (1 MB)

    The kernel may cap the buffer at `net.core.rmem_max`.  Check with:
        sysctl net.core.rmem_max

    To increase the kernel max:
        sysctl -w net.core.rmem_max=134217728  # 128 MB

    Args:
        sock:         Socket to configure.
        rcvbuf_bytes: Receive buffer size in bytes.
        sndbuf_bytes: Send buffer size in bytes.
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf_bytes)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf_bytes)

    # Verify actual sizes (kernel may have capped them)
    actual_rcv = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    actual_snd = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    # Note: Linux doubles the requested size (for overhead), so actual may be 2×
    return actual_rcv, actual_snd  # type: ignore[return-value]


def configure_busy_poll(sock: socket.socket, budget_us: int = 50) -> None:
    """Enable SO_BUSY_POLL on the socket (Linux 3.11+).

    SO_BUSY_POLL instructs the kernel to spin on the NIC receive queue
    for up to `budget_us` microseconds before blocking.  This reduces
    interrupt latency for the first packet, at the cost of CPU burn.

    Requires: kernel >= 3.11, supported NIC driver (e.g. Intel i40e, mlx5).

    Args:
        sock:      Socket to configure.
        budget_us: Maximum busy-poll time in microseconds (0 = disabled).
    """
    if not hasattr(socket, "SO_BUSY_POLL"):
        # SO_BUSY_POLL = 46 on Linux; not defined in older Python socket module
        SO_BUSY_POLL = 46
    else:
        SO_BUSY_POLL = socket.SO_BUSY_POLL  # type: ignore[attr-defined]

    try:
        sock.setsockopt(socket.SOL_SOCKET, SO_BUSY_POLL, budget_us)
    except OSError:
        pass  # Not supported on this kernel/NIC


def configure_udp_multicast(
    sock: socket.socket,
    mcast_group: str,
    iface_ip: str = "0.0.0.0",
) -> None:
    """Join a UDP multicast group.

    Sets IP_ADD_MEMBERSHIP to receive multicast datagrams.
    Also sets SO_REUSEADDR so multiple processes can bind the same port.

    Args:
        sock:        UDP socket (SOCK_DGRAM).
        mcast_group: Multicast group IP (e.g. "239.1.1.1").
        iface_ip:    Local interface IP to receive on ("0.0.0.0" = all).
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    mreq = struct.pack("4s4s", socket.inet_aton(mcast_group), socket.inet_aton(iface_ip))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)


def print_socket_options(sock: socket.socket) -> None:
    """Print current socket option values for diagnostics.

    Args:
        sock: Socket to inspect.
    """
    options = [
        ("SO_RCVBUF", socket.SOL_SOCKET, socket.SO_RCVBUF),
        ("SO_SNDBUF", socket.SOL_SOCKET, socket.SO_SNDBUF),
        ("SO_KEEPALIVE", socket.SOL_SOCKET, socket.SO_KEEPALIVE),
        ("SO_REUSEADDR", socket.SOL_SOCKET, socket.SO_REUSEADDR),
        ("TCP_NODELAY", socket.IPPROTO_TCP, socket.TCP_NODELAY),
    ]
    print("Socket options:")
    for name, level, opt in options:
        try:
            val = sock.getsockopt(level, opt)
            print(f"  {name:<20} = {val}")
        except OSError as e:
            print(f"  {name:<20} = <error: {e}>")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Low-Latency Socket Options Demo ===\n")

    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    configure_tcp_low_latency(tcp_sock)
    configure_tcp_keepalive(tcp_sock)
    print("TCP socket (before buffer config):")
    print_socket_options(tcp_sock)

    actual = configure_socket_buffers(tcp_sock, rcvbuf_bytes=8 * 1024 * 1024)
    print(f"\nAfter buffer config: rcvbuf={actual[0]:,}  sndbuf={actual[1]:,}")
    print_socket_options(tcp_sock)
    tcp_sock.close()
