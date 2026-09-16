"""Small, ROS-version-independent wire protocol for the RACER bridge.

Only the integration boundary crosses this socket:

* SCAN_BUNDLE: registered LiDAR points plus the LiDAR origin pose
* ODOMETRY: body state in RACER's common world frame
* POSITION_COMMAND: RACER's position/velocity/acceleration setpoint
* CLOCK: Gazebo simulation time for ROS 1 RACER
* EXECUTION_BLOCKED: sustained raw-LiDAR or inter-robot safety intervention

The fixed header is network byte order. Numeric payloads are explicitly
little-endian because both supported containers run on x86_64.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass


MAGIC = b"RCR1"
VERSION = 1

SCAN_BUNDLE = 1
ODOMETRY = 2
POSITION_COMMAND = 3
CLOCK = 4
EXECUTION_BLOCKED = 5

HEADER = struct.Struct("!4sBBHIdI")
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class Frame:
    message_type: int
    robot_id: int
    sequence: int
    stamp: float
    payload: bytes


def encode_frame(
    message_type: int,
    robot_id: int,
    sequence: int,
    stamp: float,
    payload: bytes,
) -> bytes:
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload too large: {len(payload)} bytes")
    return HEADER.pack(
        MAGIC,
        VERSION,
        int(message_type),
        int(robot_id),
        int(sequence) & 0xFFFFFFFF,
        float(stamp),
        len(payload),
    ) + payload


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("peer closed the RACER bridge connection")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_frame(connection: socket.socket) -> Frame:
    raw_header = _recv_exact(connection, HEADER.size)
    magic, version, message_type, robot_id, sequence, stamp, size = HEADER.unpack(
        raw_header
    )
    if magic != MAGIC:
        raise ValueError(f"bad RACER bridge magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported RACER bridge version: {version}")
    if size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"refusing oversized RACER bridge payload: {size}")
    return Frame(
        message_type=message_type,
        robot_id=robot_id,
        sequence=sequence,
        stamp=stamp,
        payload=_recv_exact(connection, size),
    )
