from os.path import basename
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from diffusers import LatentConsistencyModelPipeline

BASELINE_CHECKPOINT = \
    "https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7/resolve/main/LCM_Dreamshaper_v7_4k.safetensors?download=true"


def load_pipeline(device: str):
    output_file = Path(__file__).parent.parent.parent / "checkpoints" / basename(urlparse(BASELINE_CHECKPOINT).path)

    urlretrieve(BASELINE_CHECKPOINT, output_file)

    pipeline = LatentConsistencyModelPipeline.from_single_file(output_file).to(device)

    return pipeline, output_file
