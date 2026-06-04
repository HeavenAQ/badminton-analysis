"""Copy generated annotation assets into the Nuxt public directory for Netlify."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare llm-annotations assets for annotation-frontend Netlify deploy."
    )
    parser.add_argument("--source", type=Path, default=Path("llm-annotations"))
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=Path("annotation-frontend/public"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    public_dir = args.public_dir

    if not source.exists():
        raise FileNotFoundError(f"Annotation source does not exist: {source}")

    manifest_files = [
        "annotation_template.jsonl",
        "annotation_template.csv",
        "annotation_schema.json",
    ]
    missing = [name for name in manifest_files if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing generated annotation files: {', '.join(missing)}. "
            "Finish extraction first."
        )

    public_dir.mkdir(parents=True, exist_ok=True)
    image_dest = public_dir / "annotation-images"
    if image_dest.exists():
        shutil.rmtree(image_dest)
    image_dest.mkdir(parents=True)

    for manifest_file in manifest_files:
        shutil.copy2(source / manifest_file, public_dir / manifest_file)

    copied = 0
    for child in source.iterdir():
        if not child.is_dir():
            continue
        shutil.copytree(child, image_dest / child.name)
        copied += len(list((image_dest / child.name).rglob("*.jpg")))

    print(f"Copied manifests to {public_dir}")
    print(f"Copied {copied} images to {image_dest}")


if __name__ == "__main__":
    main()
