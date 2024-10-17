from imagehash import ImageHash
from safetensors import numpy


__KEY = "DEFAULT"


GENERATION_TIME_DIFFERENCE_THRESHOLD = 0.02


def save_image_hash(image_hash: ImageHash) -> bytes:
    return numpy.save(
        {
            __KEY: image_hash.hash,
        },
    )


def load_image_hash(image_hash_bytes: bytes) -> ImageHash:
    return ImageHash(numpy.load(image_hash_bytes)[__KEY])
