from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
from pathlib import Path

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.inference_service import InferenceService


def _labels(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["filename"]: row.get("expected_class", "")
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Process the qualitative personal-photo demo set")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    input_directory = args.input or config.paths.personal_photos
    output_path = args.output or config.paths.results / "personal_photo_demo.json"
    images = sorted(
        path
        for path in input_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No personal demo images found in {input_directory}")

    labels = _labels(input_directory / "labels.csv")
    service = InferenceService(config)
    config.paths.processed_examples.mkdir(parents=True, exist_ok=True)
    records = []
    for index, image_path in enumerate(images, start=1):
        result = service.classify_upload(
            image_path.read_bytes(),
            filename=image_path.name,
            content_type=mimetypes.guess_type(image_path.name)[0] or "application/octet-stream",
            invert=args.invert,
        ).as_dict()
        data_url = result["processed_image"].pop("data_url")
        processed_path = config.paths.processed_examples / f"personal_{index:02d}_{image_path.stem}.png"
        processed_path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
        result["processed_image"]["path"] = str(processed_path)
        records.append(
            {
                "filename": image_path.name,
                "expected_class": labels.get(image_path.name) or None,
                "result": result,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"processed={len(records)} output={output_path}")


if __name__ == "__main__":
    main()
