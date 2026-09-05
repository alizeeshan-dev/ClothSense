from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.inference_service import InferenceService


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one local JPG or PNG in memory")
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--alpha", type=float)
    inversion = parser.add_mutually_exclusive_group()
    inversion.add_argument("--invert", dest="invert", action="store_true")
    inversion.add_argument("--no-invert", dest="invert", action="store_false")
    parser.set_defaults(invert=None)
    args = parser.parse_args()

    config = load_config(args.config)
    content_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    service = InferenceService(config)
    result = service.classify_upload(
        args.image.read_bytes(),
        filename=args.image.name,
        content_type=content_type,
        invert=args.invert,
        alpha=args.alpha,
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
