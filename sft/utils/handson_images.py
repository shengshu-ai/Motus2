"""HandsOn released WAM mosaic; shared by offline encoding and online policy.

PIL bilinear letterboxing, RGB uint8, H384 W320. No second video codec.
Reference: TsinghuaC3I/HandsOn fee28452 policies/native/wam_images.py.
"""
import numpy as np
from PIL import Image


def letterbox(image, size):
    w, h = size
    factor = min(w / image.width, h / image.height)
    image = image.resize(
        (max(1, round(image.width * factor)), max(1, round(image.height * factor))),
        Image.Resampling.BILINEAR,
    )
    out = Image.new("RGB", size)
    out.paste(image, ((w - image.width) // 2, (h - image.height) // 2))
    return out


def mosaic(front, left, right):
    out = Image.new("RGB", (320, 384))
    out.paste(letterbox(front, (320, 256)), (0, 0))
    out.paste(letterbox(left, (160, 128)), (0, 256))
    out.paste(letterbox(right, (160, 128)), (160, 256))
    return out


def canvas(images):
    frames = []
    for name in ("front", "wrist_left", "wrist_right"):
        value = np.asarray(images[name])
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{name} must be uint8 HxWx3 RGB")
        frames.append(Image.fromarray(value))
    return np.asarray(mosaic(*frames)).copy()
