# frontend/drawing_utils.py

import base64
import io
from PIL import Image
import numpy as np

def convert_array_to_image(arr):
    img = Image.fromarray(arr.astype('uint8'))
    return img

def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def bytes_to_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))

def base64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64.split(",")[-1])))
