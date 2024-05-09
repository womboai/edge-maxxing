from os.path import basename
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

BASELINE_CHECKPOINT = \
    "https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7/resolve/main/LCM_Dreamshaper_v7_4k.safetensors?download=true"

AVERAGE_TIME = 30.0


def download_pipeline() -> Path:
    output_file = Path(__file__).parent.parent.parent / "checkpoints" / basename(urlparse(BASELINE_CHECKPOINT).path)

    urlretrieve(BASELINE_CHECKPOINT, output_file)

    return output_file
