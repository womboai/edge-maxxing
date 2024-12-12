import json
import os
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download


def main():
    for model_specification in sys.argv[1:]:
        model = json.loads(model_specification)

        snapshot_download(
            model["repository"],

            repo_type="model",
            revision=model["revision"],
            allow_patterns=model["include"] or None,
            ignore_patterns=model["exclude"] or None,
        )

if __name__ == '__main__':
    main()
