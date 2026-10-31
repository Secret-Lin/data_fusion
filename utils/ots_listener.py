# utils/ots_listener.py
# utils.ots_listener
# ~~~~~~~~~~~~~~~~~~~~
#
# Standalone OptiTrack NatNet v3.1+ UDP listener.
# No external dependencies beyond the stdlib + numpy.
#
# Usage::
#
#     from utils.ots_listener import OptiTrackListener
#
#     listener = OptiTrackListener(host="192.168.1.1", rb_names=["Cube"])
#     listener.start()
#
#     while running:
#         T_world_cube = listener.get_latest("Cube")  # 4x4 or None
#         time.sleep(0.001)
#
#     listener.stop()

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# NatNet protocol constants
# ---------------------------------------------------------------------------

NATNET_VERSION = (3, 0, 0, 0)
MSG_FRAME_OF_DATA = 7  # NatNet 3.x uses ID 7


def _unpack_mat44(data: bytes, offset: int) -> np.ndarray:
    """Read 16 floats → 4×4 column-major matrix."""
    vals = struct.unpack_from("<16f", data, offset)
    M = np.array(vals, dtype=np.float64).reshape((4, 4), order="F")
    return M


# ---------------------------------------------------------------------------
# OptiTrackListener
# ---------------------------------------------------------------------------


class OptiTrackListener:
    """
    Minimal NatNet v3.x UDP listener that extracts rigid-body poses.

    Parameters
    ----------
    ots_ip : str
        IP address of the OptiTrack workstation / Motive server.
        Used to send discovery pings (can be the same as multicast_addr).
    multicast_addr : str
        NatNet multicast group (default "239.255.42.99").
        Motive must be configured to stream to this group.
    rb_names : list[str]
        List of rigid-body names to track.
        Case-insensitive matching against Motive's asset names.
    command_port : int
        NatNet command port on the server (default 1510).
    data_port : int
        NatNet data (tracking) port on the server (default 1511).
    recv_buffer_size : int
        Socket receive buffer size (bytes). Default 2 MB.
    """

    def __init__(
        self,
        ots_ip: str = "192.168.1.1",
        multicast_addr: str = "239.255.42.99",
        rb_names: Optional[list[str]] = None,
        command_port: int = 1510,
        data_port: int = 1511,
        recv_buffer_size: int = 2 * 1024 * 1024,
    ):
        self.ots_ip = ots_ip
        self.multicast_addr = multicast_addr
        self.rb_names = [n.lower() for n in (rb_names or [])]
        self.command_port = command_port
        self.data_port = data_port

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Latest pose per rigid-body name (lowercase) → 4×4 ndarray or None
        self._latest: dict[str, Optional[np.ndarray]] = {}
        self._seq_num: dict[str, int] = {}

        # ---- bind UDP socket ----
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer_size)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                socket.inet_aton(self.multicast_addr) + socket.inet_aton("0.0.0.0"),
            )
        except OSError:
            # Already joined or not supported — try unicast mode
            pass
        self._sock.bind(("", self.data_port))
        self._sock.settimeout(0.5)
        self._sock.setblocking(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the receive thread."""
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the receive thread and close the socket."""
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self, rb_name: str) -> Optional[np.ndarray]:
        """
        Get the most recent 4×4 world→rigidbody transform for *rb_name*.

        Returns ``None`` if no data received yet.
        The transform maps world → rigid-body body frame:
          p_world = T @ p_body
        """
        with self._lock:
            return self._latest.get(rb_name.lower())

    def is_tracking(self, rb_name: str) -> bool:
        """Return True if the named rigid body is currently tracked."""
        return self.get_latest(rb_name) is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                data, _ = self._sock.recvfrom(32768)
            except (socket.timeout, BlockingIOError, OSError):
                continue

            try:
                self._parse(data)
            except Exception:
                # Never let a parse error crash the listener
                pass

    def _parse(self, data: bytes) -> None:
        """Parse a NatNet MoCap frame packet (Message ID 7)."""
        if len(data) < 4:
            return

        msg_id = struct.unpack_from("<h", data, 0)[0]
        if msg_id != MSG_FRAME_OF_DATA:
            return

        # Skip message size (int) — total header is 8 bytes in NatNet 3.x
        # MessageID(2) + MessageSize(2) + FrameNumber(4) + MarkerSetData...
        pos = 4
        frame_num = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        # ---- Marker Set Data (skipped) ----
        if pos + 4 > len(data):
            return
        name_len = struct.unpack_from("<i", data, pos)[0]
        pos += 4 + name_len + 4  # name + marker count

        # ---- Other sets (skip) ----
        for _ in range(3):  # unlabeled, rigid, skeleton
            if pos + 4 > len(data):
                return
            n_items = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            for _ in range(n_items):
                if pos + 52 > len(data):  # 1 int + 3*4 pos + 4 size + 3*4 cov
                    return
                pos += 52

        # ---- Rigid Bodies ----
        if pos + 4 > len(data):
            return
        n_rb = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        for _ in range(n_rb):
            if pos + 84 > len(data):  # header bytes before markers
                return
            rb_id = struct.unpack_from("<i", data, pos)[0]
            pos += 4

            # Position (x, y, z) — NatNet uses Y-up right-handed
            px, py, pz = struct.unpack_from("<fff", data, pos)
            pos += 12

            # Orientation quaternion (qx, qy, qz, qw)
            qx, qy, qz, qw = struct.unpack_from("<ffff", data, pos)
            pos += 16

            # Build 4×4 transform (world → rigid body)
            R = self._quat_to_matrix(qx, qy, qz, qw)
            t = np.array([px, py, pz], dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t

            # Skip mean error (4) + marker data
            pos += 4
            n_markers = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            for _ in range(n_markers):
                pos += 36  # pos(12) + cov(12) + id(4) + size(4) + error(4)

            # Look up name from ID — in NatNet 3.x we don't get names in the
            # binary packet, only IDs.  We keep a manual mapping set by
            # set_rb_id_name() so we can still match by ID.
            key = self._rb_id_to_name.get(rb_id)
            if key is not None:
                with self._lock:
                    self._latest[key] = T
                    self._seq_num[key] = frame_num

    @staticmethod
    def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Quaternion (qx,qy,qz,qw) → 3×3 rotation matrix (world-up, right-handed)."""
        # Normalise
        norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        # Direct conversion (avoiding scipy)
        R = np.zeros((3, 3), dtype=np.float64)
        R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
        R[0, 1] = 2 * (qx * qy - qz * qw)
        R[0, 2] = 2 * (qx * qz + qy * qw)
        R[1, 0] = 2 * (qx * qy + qz * qw)
        R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
        R[1, 2] = 2 * (qy * qz - qx * qw)
        R[2, 0] = 2 * (qx * qz - qy * qw)
        R[2, 1] = 2 * (qy * qz + qx * qw)
        R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
        return R

    # ------------------------------------------------------------------
    # Name→ID mapping (needed because NatNet binary doesn't send names)
    # ------------------------------------------------------------------

    # Sub-classes / callers can populate this dict:
    #   listener._rb_id_to_name[rb_id_int] = "Cube"
    _rb_id_to_name: dict[int, str] = {}

    def register_rb(self, rb_id: int, name: str) -> None:
        """Register a rigid-body ID → name mapping.

        Motive assigns IDs automatically; you can look them up from Motive's
        asset panel or from a NatNet connection log.
        """
        self._rb_id_to_name[rb_id] = name.lower()
        self._latest.setdefault(name.lower(), None)
        self._seq_num.setdefault(name.lower(), 0)
