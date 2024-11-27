import gc
from abc import ABC, abstractmethod
from io import BytesIO
from typing import ContextManager

from transformers import CLIPProcessor, CLIPVisionModelWithProjection, PreTrainedModel

from base.device import Device


class OutputComparator(ContextManager, ABC):
    @abstractmethod
    def compare(self, baseline: bytes, optimized: bytes) -> float:
        pass

    def __wrapped_compare(self, baseline: bytes, optimized: bytes):
        return self.compare(baseline, optimized)

    __call__ = __wrapped_compare


class ImageOutputComparator(OutputComparator):
    device: Device
    clip: PreTrainedModel
    processor: CLIPProcessor

    def __init__(self, device: Device):
        self.device = device
        self.clip = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device.get_name())
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    def compare(self, baseline: bytes, optimized: bytes):
        from torch import manual_seed
        from torch.nn.functional import cosine_similarity

        from PIL import Image

        from skimage import metrics
        import cv2

        import numpy

        manual_seed(0)

        def load_image(data: bytes):
            with BytesIO(data) as fp:
                return numpy.array(Image.open(fp).convert("RGB"))

        def clip_embeddings(image: numpy.ndarray):
            processed_input = self.processor(images=image, return_tensors="pt").to(self.device.get_name())

            return self.clip(**processed_input).image_embeds.to(self.device.get_name())

        baseline_array = load_image(baseline)
        optimized_array = load_image(optimized)

        grayscale_baseline = cv2.cvtColor(baseline_array, cv2.COLOR_RGB2GRAY)
        grayscale_optimized = cv2.cvtColor(optimized_array, cv2.COLOR_RGB2GRAY)

        structural_similarity = metrics.structural_similarity(grayscale_baseline, grayscale_optimized, full=True)[0]

        del grayscale_baseline
        del grayscale_optimized

        baseline_embeddings = clip_embeddings(baseline_array)
        optimized_embeddings = clip_embeddings(optimized_array)

        clip_similarity = cosine_similarity(baseline_embeddings, optimized_embeddings)[0].item()

        return clip_similarity * 0.35 + structural_similarity * 0.65

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.clip
        del self.processor

        gc.collect()
        self.device.empty_cache()
