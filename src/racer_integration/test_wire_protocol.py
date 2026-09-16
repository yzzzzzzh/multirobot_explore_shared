#!/usr/bin/env python3

import socket
import struct
import threading
import unittest

import racer_wire


class WireProtocolTest(unittest.TestCase):
    def test_fragmented_socket_round_trip(self):
        left, right = socket.socketpair()
        payload = struct.pack("<7dI", 1, 2, 3, 0, 0, 0, 1, 2)
        payload += struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 8)
        encoded = racer_wire.encode_frame(
            racer_wire.SCAN_BUNDLE, 6, 123, 42.25, payload
        )

        def fragmented_writer():
            for index in range(0, len(encoded), 7):
                left.sendall(encoded[index:index + 7])
            left.close()

        thread = threading.Thread(target=fragmented_writer)
        thread.start()
        frame = racer_wire.recv_frame(right)
        thread.join()
        right.close()

        self.assertEqual(frame.message_type, racer_wire.SCAN_BUNDLE)
        self.assertEqual(frame.robot_id, 6)
        self.assertEqual(frame.sequence, 123)
        self.assertAlmostEqual(frame.stamp, 42.25)
        self.assertEqual(frame.payload, payload)

    def test_rejects_bad_magic(self):
        left, right = socket.socketpair()
        valid = racer_wire.encode_frame(racer_wire.ODOMETRY, 1, 1, 0.0, b"x")
        left.sendall(b"NOPE" + valid[4:])
        with self.assertRaisesRegex(ValueError, "magic"):
            racer_wire.recv_frame(right)
        left.close()
        right.close()

    def test_clock_frame_round_trip(self):
        left, right = socket.socketpair()
        left.sendall(
            racer_wire.encode_frame(
                racer_wire.CLOCK, 0, 7, 123.456, b""
            )
        )
        frame = racer_wire.recv_frame(right)
        left.close()
        right.close()
        self.assertEqual(frame.message_type, racer_wire.CLOCK)
        self.assertEqual(frame.robot_id, 0)
        self.assertEqual(frame.payload, b"")
        self.assertAlmostEqual(frame.stamp, 123.456)

    def test_execution_blocked_frame_round_trip(self):
        left, right = socket.socketpair()
        payload = struct.pack("<B", 1)
        left.sendall(
            racer_wire.encode_frame(
                racer_wire.EXECUTION_BLOCKED, 4, 8, 15.0, payload
            )
        )
        frame = racer_wire.recv_frame(right)
        left.close()
        right.close()
        self.assertEqual(frame.message_type, racer_wire.EXECUTION_BLOCKED)
        self.assertEqual(frame.robot_id, 4)
        self.assertEqual(frame.payload, payload)


if __name__ == "__main__":
    unittest.main()
