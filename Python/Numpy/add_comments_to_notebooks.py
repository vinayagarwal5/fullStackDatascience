#!/usr/bin/env python3
"""Add explanatory comments to all code cells in Numpy notebooks without removing outputs.

This script scans the Python/Numpy folder for .ipynb files and inserts lightweight
header comments at the start of each code cell. It preserves all existing outputs,
metadata, and execution counts.

Usage:
    python Python/Numpy/add_comments_to_notebooks.py --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SECTION_HINTS = [
    ("np.array", "# Create NumPy arrays and inspect their structure"),
    ("np.arange", "# Generate a range of NumPy values using arange"),
    ("np.random", "# Generate random values for NumPy examples"),
    ("reshape", "# Reshape an array to change its dimension layout"),
    ("transpose", "# Transpose an array to switch rows and columns"),
    ("sum", "# Use NumPy aggregation functions to summarize data"),
    ("mean", "# Compute the mean of array elements"),
    ("median", "# Compute the median value in the array"),
    ("std", "# Calculate the standard deviation of the array"),
    ("var", "# Calculate the variance of the array"),
    ("sin", "# Apply trigonometric functions to each array element"),
    ("dot", "# Perform a dot product between two arrays"),
    ("exp", "# Apply exponential functions element-wise"),
    ("log", "# Apply logarithm functions element-wise"),
    ("max", "# Find the maximum value in the array"),
    ("min", "# Find the minimum value in the array"),
    ("flatten", "# Flatten the array into a 1D view"),
    ("broadcast", "# Demonstrate broadcasting behavior in NumPy"),
    ("axis", "# Use axis to apply operations across rows or columns"),
    ("concatenate", "# Join arrays together along an axis"),
    ("vstack", "# Stack arrays vertically"),
    ("hstack", "# Stack arrays horizontally"),
    ("split", "# Split an array into multiple smaller arrays"),
    ("unique", "# Extract unique elements from the array"),
    ("where", "# Apply conditional selection using np.where"),
    ("clip", "# Clip values to a defined range"),
    ("astype", "# Convert array elements to a different data type"),
    ("copy", "# Copy an array to create an independent data structure"),
    ("view", "# Use a view of the array without copying memory"),
]


def choose_comment(code: str) -> str:
    code_clean = code.strip()
    if not code_clean:
        return "# Example code cell"

    lowered = code_clean.lower()
    for token, comment in SECTION_HINTS:
        if token in lowered:
            return comment

    if any(keyword in lowered for keyword in ["for ", "if ", "while "]):
        return "# Example of control flow using NumPy operations"
    if "print(" in lowered:
        return "# Display the output of the NumPy example"
    return "# NumPy example"


def add_comments_to_notebook(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if not isinstance(source, list):
            source = [source]

        text = "".join(source)
        stripped = text.lstrip()
        if stripped.startswith("#"):
            continue

        comment = choose_comment(text)
        new_source = [comment + "\n\n", *source]
        cell["source"] = new_source
        changed = True

    if changed:
        with path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Add comments to NumPy notebook code cells.")
    parser.add_argument("--apply", action="store_true", help="Apply the change to the notebooks.")
    args = parser.parse_args()

    if not args.apply:
        print("Dry run only. Pass --apply to modify the notebooks.")
        return

    base_dir = Path(__file__).resolve().parent
    notebooks = sorted(base_dir.glob("*.ipynb"))

    if not notebooks:
        print(f"No notebooks found in {base_dir}")
        return

    updated = []
    for nb in notebooks:
        if add_comments_to_notebook(nb):
            updated.append(str(nb.name))

    if updated:
        print("Updated notebooks:")
        for name in updated:
            print(f"- {name}")
    else:
        print("No notebook changes were needed.")


if __name__ == "__main__":
    main()
