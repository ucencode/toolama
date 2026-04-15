#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process all PDF files in the project root using pdf2img-ocr.py"
    )
    parser.add_argument(
        "--preset",
        type=str,
        metavar="FILE",
        required=True,
        help="Preset file to use (from presets/<FILE>), passed to each pdf2img-ocr.py call",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).parent.resolve()
    main_script = project_root / "pdf2img-ocr.py"

    pdfs = sorted(project_root.glob("*.pdf"))

    if not pdfs:
        print("No PDF files found in the project root.")
        sys.exit(1)

    total = len(pdfs)
    print(f"Found {total} PDF file(s). Starting batch processing with preset: {args.preset}\n")

    succeeded = []
    failed = []

    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{total}] Processing: {pdf.name}")
        print("-" * 60)

        result = subprocess.run(
            [sys.executable, str(main_script), str(pdf), "--preset", args.preset]
        )

        if result.returncode == 0:
            succeeded.append(pdf.name)
            print(f"[{i}/{total}] Done: {pdf.name}\n")
        else:
            failed.append(pdf.name)
            print(f"[{i}/{total}] Failed (exit code {result.returncode}): {pdf.name}\n")

    print("=" * 60)
    print(f"Batch complete: {len(succeeded)}/{total} succeeded, {len(failed)}/{total} failed.")

    if succeeded:
        print("\nSucceeded:")
        for name in succeeded:
            print(f"  [ok] {name}")

    if failed:
        print("\nFailed:")
        for name in failed:
            print(f"  [!!] {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
