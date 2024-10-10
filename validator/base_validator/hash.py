from imagehash import ImageHash
from safetensors import numpy


__KEY = "DEFAULT"


HASH_DIFFERENCE_THRESHOLD = 8


def save_image_hash(image_hash: ImageHash) -> bytes:
    return numpy.save(
        {
            __KEY: image_hash.hash,
        },
    )


def load_image_hash(image_hash_bytes: bytes) -> ImageHash:
    return ImageHash(numpy.load(image_hash_bytes)[__KEY])
