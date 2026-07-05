import csv
import os
from pathlib import Path


def ensure_outputs_structure(outputs_dir):
    """Create the CSV-only result structure used by the benchmark."""
    base = Path(outputs_dir)
    for relative in ["results", "debug", "metadata"]:
        (base / relative).mkdir(parents=True, exist_ok=True)
    return {
        "base": str(base),
        "results": str(base / "results"),
        "debug": str(base / "debug"),
        "metadata": str(base / "metadata"),
    }


def csv_path(outputs_dir, group, filename):
    paths = ensure_outputs_structure(outputs_dir)
    return os.path.join(paths[group], filename)


def _normalize_row(row, fieldnames):
    return {field: row.get(field, "") for field in fieldnames}


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(_normalize_row(row, fieldnames))
    print(f"[INFO] Saved {path}")


def append_csv_rows(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row, fieldnames))


def append_csv_row(path, row, fieldnames=None):
    append_csv_rows(path, [row], fieldnames=fieldnames)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value, default=None):
    try:
        if value in ("", None, "nan", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
