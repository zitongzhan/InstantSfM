#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import subprocess
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path


SCOPES_TO_SKIP = {"summary_all"}
SUMMARY_SCOPE = "summary_avg"
SCENE_SCOPE = "scene"
AVERAGE_KEY = "__average__"
METRIC_PREFIXES = ("AUC",)
FILENAME_PATTERNS = (
    re.compile(r"^(?P<dataset>.+?)(?:__|--)(?P<commit>[0-9a-f]{7,40})$", re.IGNORECASE),
    re.compile(r"^(?P<commit>[0-9a-f]{7,40})$", re.IGNORECASE),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a JSON manifest for the InstantSfM KPI dashboard."
    )
    parser.add_argument(
        "--input-dir",
        default="performance-data",
        help="Directory containing CSV evaluation files.",
    )
    parser.add_argument(
        "--output",
        default="static/data/performance-manifest.json",
        help="Output manifest path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (repo_root / args.input_dir).resolve()
    output_path = (repo_root / args.output).resolve()

    datasets = {}
    for csv_path in sorted(input_dir.rglob("*.csv")) if input_dir.exists() else []:
        dataset_name, commit_sha = infer_dataset_and_commit(input_dir, csv_path)
        if not dataset_name or not commit_sha:
            continue

        parsed = parse_csv(csv_path)
        if not parsed["metrics"] or not parsed["scenes"] or not parsed["summary"]:
            continue

        commit_info = lookup_commit_info(repo_root, commit_sha)
        dataset_bucket = datasets.setdefault(
            dataset_name,
            {
                "metrics": OrderedDict(),
                "scenes": OrderedDict(),
                "commits": [],
            },
        )

        for metric in parsed["metrics"]:
            dataset_bucket["metrics"][metric] = True
        for scene in parsed["scenes"]:
            dataset_bucket["scenes"][scene] = True

        dataset_bucket["commits"].append(
            {
                "commit": commit_info["commit"],
                "short_commit": commit_info["short_commit"],
                "committed_at": commit_info["committed_at"],
                "source_file": make_display_path(csv_path, input_dir, repo_root),
                "summary": parsed["summary"],
                "scenes": parsed["scenes"],
                "sort_key": commit_info["sort_key"],
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    for dataset_name, dataset in sorted(datasets.items()):
        commits = sorted(dataset["commits"], key=lambda item: (item["sort_key"], item["commit"]))
        for commit in commits:
            commit.pop("sort_key", None)

        manifest["datasets"][dataset_name] = {
            "metrics": sorted(dataset["metrics"].keys(), key=natural_metric_key),
            "scenes": sorted(dataset["scenes"].keys()),
            "commits": commits,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def infer_dataset_and_commit(input_dir: Path, csv_path: Path):
    relative = csv_path.relative_to(input_dir)
    stem = csv_path.stem
    dataset_name = relative.parent.as_posix() if relative.parent != Path(".") else None

    for pattern in FILENAME_PATTERNS:
        match = pattern.match(stem)
        if match:
            groups = match.groupdict()
            commit = groups["commit"].lower()
            dataset = groups.get("dataset") or dataset_name
            return sanitize_dataset(dataset), commit

    if dataset_name:
        commit_match = re.search(r"([0-9a-f]{7,40})", stem, flags=re.IGNORECASE)
        if commit_match:
            return sanitize_dataset(dataset_name), commit_match.group(1).lower()

    return None, None


def sanitize_dataset(value):
    if not value:
        return None
    return value.strip("/").replace("\\", "/")


def make_display_path(csv_path: Path, input_dir: Path, repo_root: Path):
    for base in (repo_root, input_dir):
        try:
            return str(csv_path.relative_to(base))
        except ValueError:
            continue
    return str(csv_path)


def natural_metric_key(value):
    parts = re.split(r"(\d+(?:\.\d+)?)", value)
    key = []
    for part in parts:
        if re.fullmatch(r"\d+(?:\.\d+)?", part):
            key.append((0, float(part)))
        else:
            key.append((1, part))
    return key


def parse_csv(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        metrics = [name for name in fieldnames if name and name.startswith(METRIC_PREFIXES)]
        summary = None
        scenes = OrderedDict()

        for row in reader:
            scope = (row.get("scope") or "").strip()
            scene_name = (row.get("scene") or "").strip()
            if scope in SCOPES_TO_SKIP:
                continue

            values = {
                metric: to_float(row.get(metric))
                for metric in metrics
            }
            values[AVERAGE_KEY] = average_metric(values.values())

            if scope == SUMMARY_SCOPE:
                summary = values
            elif scope == SCENE_SCOPE and scene_name:
                scenes[scene_name] = values

        if summary is None and scenes:
            summary = {
                metric: average_metric(scene_values.get(metric) for scene_values in scenes.values())
                for metric in metrics
            }
            summary[AVERAGE_KEY] = average_metric(summary.values())

        return {
            "metrics": metrics,
            "summary": summary,
            "scenes": scenes,
        }


def average_metric(values):
    numeric_values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def lookup_commit_info(repo_root: Path, commit_sha: str):
    if not git_commit_exists(repo_root, commit_sha):
        return {
            "commit": commit_sha,
            "short_commit": commit_sha[:7],
            "committed_at": None,
            "sort_key": "9999-12-31T23:59:59+00:00",
        }

    result = subprocess.run(
        ["git", "show", "-s", "--format=%H%n%h%n%cI", commit_sha],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    full_sha, short_sha, committed_at = result.stdout.strip().splitlines()
    return {
        "commit": full_sha,
        "short_commit": short_sha,
        "committed_at": committed_at,
        "sort_key": committed_at,
    }


def git_commit_exists(repo_root: Path, commit_sha: str):
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


if __name__ == "__main__":
    main()
