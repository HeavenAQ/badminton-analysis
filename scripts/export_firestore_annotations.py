"""Merge Firestore expert annotations back into local annotation files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from google.cloud import firestore


ANNOTATION_FIELDS = [
    "score",
    "feedback",
    "correction_suggestion",
    "usable_for_training",
    "annotator",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Firestore expert annotations into llm-annotation CSV/JSONL files."
    )
    parser.add_argument("--project", default="moe-linebot-2025")
    parser.add_argument("--database", default="badminton-annotations")
    parser.add_argument("--collection", default="badminton_vlm_annotations")
    parser.add_argument("--annotation-root", type=Path, default=Path("llm-annotations"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("llm-annotations/annotation_merged.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("llm-annotations/annotation_merged.jsonl"),
    )
    parser.add_argument(
        "--overwrite-template",
        action="store_true",
        help="Also overwrite annotation_template.csv/jsonl with the merged annotations.",
    )
    return parser.parse_args()


def load_firestore_annotations(
    *,
    project: str,
    database: str,
    collection: str,
) -> dict[str, dict[str, Any]]:
    client = firestore.Client(project=project, database=database)
    annotations: dict[str, dict[str, Any]] = {}
    for doc in client.collection(collection).stream():
        data = doc.to_dict() or {}
        sample_id = str(data.get("sample_id") or doc.id)
        annotations[sample_id] = data
    return annotations


def merge_csv(
    template_path: Path,
    output_path: Path,
    annotations: dict[str, dict[str, Any]],
) -> int:
    merged_count = 0
    with template_path.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise RuntimeError(f"No CSV header found in {template_path}")
        fieldnames = list(reader.fieldnames)
        for field in ANNOTATION_FIELDS:
            if field not in fieldnames:
                fieldnames.append(field)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                annotation = annotations.get(row["sample_id"])
                if annotation:
                    merged_count += 1
                    for field in ANNOTATION_FIELDS:
                        value = annotation.get(field, "")
                        row[field] = "" if value is None else str(value)
                writer.writerow(row)
    return merged_count


def merge_jsonl(
    template_path: Path,
    output_path: Path,
    annotations: dict[str, dict[str, Any]],
) -> int:
    merged_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open(encoding="utf-8") as src, output_path.open(
        "w",
        encoding="utf-8",
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            annotation = annotations.get(record["sample_id"])
            if annotation:
                merged_count += 1
                expert_annotation = record.setdefault("expert_annotation", {})
                for field in ANNOTATION_FIELDS:
                    expert_annotation[field] = annotation.get(field, expert_annotation.get(field, ""))

                assistant_content = record.get("vlm_sft_messages", [{}, {}])[-1].get("content")
                if isinstance(assistant_content, dict):
                    assistant_content["score"] = expert_annotation.get("score", "")
                    assistant_content["feedback"] = expert_annotation.get("feedback", "")
                    assistant_content["correction_suggestion"] = expert_annotation.get(
                        "correction_suggestion",
                        "",
                    )
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    return merged_count


def main() -> None:
    args = parse_args()
    csv_template = args.annotation_root / "annotation_template.csv"
    jsonl_template = args.annotation_root / "annotation_template.jsonl"

    if not csv_template.exists() or not jsonl_template.exists():
        raise FileNotFoundError(
            "annotation_template.csv/jsonl not found. Finish frame extraction first."
        )

    annotations = load_firestore_annotations(
        project=args.project,
        database=args.database,
        collection=args.collection,
    )
    csv_count = merge_csv(csv_template, args.output_csv, annotations)
    jsonl_count = merge_jsonl(jsonl_template, args.output_jsonl, annotations)

    if args.overwrite_template:
        args.output_csv.replace(csv_template)
        args.output_jsonl.replace(jsonl_template)

    print(f"Loaded Firestore annotations: {len(annotations)}")
    print(f"Merged CSV rows: {csv_count}")
    print(f"Merged JSONL rows: {jsonl_count}")
    print(f"Wrote: {csv_template if args.overwrite_template else args.output_csv}")
    print(f"Wrote: {jsonl_template if args.overwrite_template else args.output_jsonl}")


if __name__ == "__main__":
    main()
