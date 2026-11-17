"""Wire format shared by the policy server and its clients.

Observations and actions travel as msgpack maps; numpy arrays are packed as
``{"__ndarray__": True, "dtype", "shape", "data"}``. Images are carried per
camera as raw RGB bytes (lossless, default) or as a JPEG / PNG bitstream.

This module only needs numpy and msgpack (plus OpenCV for JPEG / PNG), so it
can be copied to a robot or simulator machine together with ``client.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import msgpack
import numpy as np


PROTOCOL_MAGIC = "motus2-policy"
PROTOCOL_VERSION = 1
SUPPORTED_TRANSPORTS = ("raw_rgb8", "jpeg", "png")
DEFAULT_TRANSPORT = "raw_rgb8"


class ProtocolError(ValueError):
    pass


def normalize_transport(value: Optional[str]) -> str:
    key = str(value or DEFAULT_TRANSPORT).strip().lower()
    key = {"raw": "raw_rgb8", "rgb8": "raw_rgb8", "jpg": "jpeg"}.get(key, key)
    if key not in SUPPORTED_TRANSPORTS:
        raise ProtocolError(
            f"unsupported image transport {value!r}; expected one of {SUPPORTED_TRANSPORTS}"
        )
    return key


def protocol_descriptor(transport: str = DEFAULT_TRANSPORT) -> dict:
    resolved = normalize_transport(transport)
    return {
        "magic": PROTOCOL_MAGIC,
        "version": PROTOCOL_VERSION,
        "transport": resolved,
        "image_layout": "HWC_RGB_uint8",
        "lossless": resolved != "jpeg",
    }


def validate_protocol_descriptor(actual: Mapping[str, Any], expected_transport: str) -> None:
    expected = protocol_descriptor(expected_transport)
    for key in ("magic", "version", "transport"):
        if actual.get(key) != expected[key]:
            raise ProtocolError(
                f"protocol handshake mismatch for {key}: "
                f"server={expected[key]!r}, client={actual.get(key)!r}"
            )


# ---------------------------------------------------------------------------
# msgpack with numpy arrays
# ---------------------------------------------------------------------------

def _encode_default(obj):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            "__ndarray__": True,
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "data": arr.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj)!r}")


def _decode_hook(obj):
    if obj.get("__ndarray__") is True:
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    return obj


def packb(obj) -> bytes:
    return msgpack.packb(obj, default=_encode_default, use_bin_type=True)


def unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode_hook, raw=False)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def require_rgb_uint8(image: np.ndarray, label: str = "image") -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ProtocolError(f"{label} must be HWC RGB with 3 channels, got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ProtocolError(f"{label} must have dtype uint8, got {arr.dtype}")
    return np.ascontiguousarray(arr)


def encode_image(image: np.ndarray, transport: str = DEFAULT_TRANSPORT, *, jpeg_quality: int = 95) -> dict:
    arr = require_rgb_uint8(image)
    resolved = normalize_transport(transport)
    if resolved == "raw_rgb8":
        return {
            "encoding": "raw_rgb8",
            "shape": [int(v) for v in arr.shape],
            "dtype": "uint8",
            "data": arr.tobytes(order="C"),
        }
    import cv2

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if resolved == "jpeg":
        quality = int(jpeg_quality)
        if not 1 <= quality <= 100:
            raise ProtocolError("jpeg_quality must be in [1, 100]")
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return {"encoding": resolved, "data": buf.tobytes()}


def decode_image(packet: Mapping[str, Any]) -> np.ndarray:
    encoding = str(packet.get("encoding", "")).strip().lower()
    data = packet.get("data")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ProtocolError(f"{encoding} image packet data must be bytes")
    raw = bytes(data)
    if encoding == "raw_rgb8":
        if packet.get("dtype") != "uint8":
            raise ProtocolError(f"raw image dtype must be uint8, got {packet.get('dtype')!r}")
        shape = tuple(int(v) for v in packet.get("shape", ()))
        if len(shape) != 3 or shape[2] != 3 or any(v <= 0 for v in shape):
            raise ProtocolError(f"invalid raw image shape {shape}")
        if len(raw) != int(np.prod(shape)):
            raise ProtocolError(
                f"raw image byte length mismatch: got {len(raw)}, expected {int(np.prod(shape))}"
            )
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    if encoding not in {"jpeg", "png"}:
        raise ProtocolError(f"unsupported image encoding {encoding!r}")
    import cv2

    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ProtocolError(f"{encoding} decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """One synchronized observation.

    images: camera name -> HxWx3 uint8 RGB. Names must match ``vision.cameras``
        of the server config; the server composes them into the model input.
    state: robot state in the model's state space, shape [state_dim]. Required
        for /infer, ignored by /update_obs and /prepare_infer.
    meta: ``observation_id`` (unique within an episode) and ``session_epoch``
        (the value sent with /reset) are required.
    """

    images: Dict[str, np.ndarray]
    state: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self, transport: str = DEFAULT_TRANSPORT, *, jpeg_quality: int = 95) -> bytes:
        resolved = normalize_transport(transport)
        payload = {
            "magic": PROTOCOL_MAGIC,
            "version": PROTOCOL_VERSION,
            "transport": resolved,
            "images": {
                str(name): encode_image(img, resolved, jpeg_quality=jpeg_quality)
                for name, img in self.images.items()
            },
            "state": None if self.state is None else np.asarray(self.state, dtype=np.float32),
            "meta": dict(self.meta),
        }
        return packb(payload)

    @staticmethod
    def from_bytes(data: bytes, *, expected_transport: Optional[str] = None) -> "Observation":
        payload = unpackb(data)
        if not isinstance(payload, dict):
            raise ProtocolError("observation payload must be a mapping")
        if payload.get("magic") != PROTOCOL_MAGIC:
            raise ProtocolError("not a motus2 observation payload")
        if int(payload.get("version", -1)) != PROTOCOL_VERSION:
            raise ProtocolError(
                f"protocol version mismatch: got {payload.get('version')}, expected {PROTOCOL_VERSION}"
            )
        transport = normalize_transport(payload.get("transport"))
        if expected_transport is not None and transport != normalize_transport(expected_transport):
            raise ProtocolError(f"transport mismatch: got {transport!r}, expected {expected_transport!r}")
        packets = payload.get("images")
        if not isinstance(packets, dict) or not packets:
            raise ProtocolError("observation must carry at least one image")
        state = payload.get("state")
        return Observation(
            images={str(name): decode_image(packet) for name, packet in packets.items()},
            state=None if state is None else np.asarray(state, dtype=np.float32),
            meta=dict(payload.get("meta") or {}),
        )


@dataclass
class Action:
    """Action chunk returned by /infer and /predict.

    actions: [chunk_length, action_dim] float32 in the model's action space
        (see README "Tensor conventions").
    """

    actions: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        return packb({"actions": np.asarray(self.actions, dtype=np.float32), "meta": dict(self.meta)})

    @staticmethod
    def from_bytes(data: bytes) -> "Action":
        payload = unpackb(data)
        return Action(
            actions=np.asarray(payload["actions"], dtype=np.float32),
            meta=dict(payload.get("meta") or {}),
        )
