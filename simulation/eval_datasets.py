import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

DATASET = "skin_cancer"
STEPS = ["import", "anon", "anon_synth", "synth"]
# Cell values (compared case-insensitively after stripping) that count as missing.
MISSING_TOKENS = {"", "nan", "null", "none"}

BASE_DIR = Path(__file__).resolve().parent
DATASET_CONFIG_DIR = BASE_DIR / "dataset_configs" / DATASET
RESULTS_DIR = BASE_DIR / "experiment_results" / DATASET
OUTPUT_FILE = RESULTS_DIR / "dataset_statistics.json"

# Run: python simulation/eval_datasets.py


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS


def analyze_csv(path: Path) -> Dict[str, Any]:
    rows = 0
    missing_values = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        columns = len(header)
        for row in reader:
            if not row:
                continue
            rows += 1
            # Rows shorter than the header are padded with missing values.
            missing_values += sum(1 for value in row if is_missing(value))
            missing_values += max(columns - len(row), 0)

    total_cells = rows * columns
    return {
        "file": path.name,
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "missing_ratio": missing_values / total_cells if total_cells else 0.0,
    }


def find_original_dataset() -> Optional[Path]:
    for name in ("original_dataset.csv", "original-dataset.csv"):
        candidate = DATASET_CONFIG_DIR / name
        if candidate.exists():
            return candidate
    csv_files = sorted(DATASET_CONFIG_DIR.glob("*.csv"))
    return csv_files[0] if csv_files else None


def average(files: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [entry[key] for entry in files]
    return statistics.mean(values) if values else None


def stdev(files: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [entry[key] for entry in files]
    return statistics.stdev(values) if len(values) > 1 else None


def analyze_step(step: str) -> Dict[str, Any]:
    folder = RESULTS_DIR / step / "files"
    if not folder.exists():
        print(f"Warning: folder not found, skipping: {folder}")
        files = []
    else:
        files = [analyze_csv(path) for path in sorted(folder.glob("*.csv"))]
        print(f"[{step}] analyzed {len(files)} file(s) in {folder}")

    return {
        "folder": str(folder.relative_to(BASE_DIR)),
        "file_count": len(files),
        "average": {
            "rows": average(files, "rows"),
            "missing_values": average(files, "missing_values"),
            "missing_ratio": average(files, "missing_ratio"),
        },
        "stdev": {
            "rows": stdev(files, "rows"),
            "missing_values": stdev(files, "missing_values"),
            "missing_ratio": stdev(files, "missing_ratio"),
        },
        "files": files,
    }


def main():
    original_path = find_original_dataset()
    if original_path is None:
        raise FileNotFoundError(f"No original dataset CSV found in {DATASET_CONFIG_DIR}")
    original = analyze_csv(original_path)
    print(
        f"[original] {original_path.name}: {original['rows']} rows, "
        f"{original['missing_values']} missing values"
    )

    results = {
        "dataset": DATASET,
        "original": original,
        "steps": {step: analyze_step(step) for step in STEPS},
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
