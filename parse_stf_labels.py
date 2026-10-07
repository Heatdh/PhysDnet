#!/usr/bin/env python3
"""
Parse STF weather-condition labels and produce per-timestamp annotations.

The STF dataset labels are stored per-frame as JSON files inside
``gt_labels/cam_left_labels_TMP/``.  Each JSON contains bounding boxes
**and** a weather tag for the frame.

This script:
  1. Scans all label JSONs and extracts ``{timestamp: weather_condition}``.
  2. Cross-references with available camera image timestamps.
  3. Outputs:
       - ``stf_frame_weather.json``  – full mapping {timestamp: weather}
       - ``clear_timestamps.txt``    – one timestamp per line (clear-weather)
       - ``fog_timestamps.txt``      – timestamps with fog/rain/snow

Usage:
    python parse_stf_labels.py --stf_root data/stf/SeeingThroughFog \
                               --out_dir data/stf/meta
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


# STF label files may be JSON with a "weather" or "weather_original" field,
# or TXT with a specific format.  We try to handle both.

def _extract_weather_from_json(path: str) -> str | None:
    """Read a single label JSON and return the weather tag."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    # Different STF label versions use different keys
    for key in ("weather", "weather_original", "Weather", "daytime_weather"):
        if key in data:
            return str(data[key]).strip().lower()

    # Sometimes the weather is inside a nested "image_attributes" dict
    attrs = data.get("image_attributes", data.get("attributes", {}))
    if isinstance(attrs, dict):
        for key in ("weather", "weather_original"):
            if key in attrs:
                return str(attrs[key]).strip().lower()

    return None


def _extract_weather_from_txt(path: str) -> str | None:
    """Fallback: read a TXT label file line-by-line looking for weather."""
    try:
        with open(path) as f:
            for line in f:
                if "weather" in line.lower():
                    # Try "weather: clear" style
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        return parts[1].strip().lower()
    except Exception:
        pass
    return None


def _timestamp_from_filename(filename: str) -> str | None:
    """Extract the nanosecond-precision timestamp from an STF filename.

    STF naming: ``1538062105_427018693.{png,json,txt}``
    """
    m = re.match(r"(\d{10}_\d{9})", filename)
    return m.group(1) if m else None


def parse_labels(labels_dir: str) -> dict[str, str]:
    """Scan label directory and return {timestamp: weather_condition}."""
    mapping: dict[str, str] = {}
    labels_path = Path(labels_dir)

    if not labels_path.exists():
        print(f"[WARN] Labels directory not found: {labels_dir}")
        return mapping

    # Walk all files
    for fpath in sorted(labels_path.rglob("*")):
        if fpath.is_dir():
            continue

        ts = _timestamp_from_filename(fpath.name)
        if ts is None:
            continue

        weather = None
        if fpath.suffix == ".json":
            weather = _extract_weather_from_json(str(fpath))
        elif fpath.suffix in (".txt", ".csv"):
            weather = _extract_weather_from_txt(str(fpath))

        if weather:
            mapping[ts] = weather

    return mapping


def get_available_timestamps(cam_dir: str) -> set[str]:
    """Get set of timestamps that have camera images."""
    timestamps = set()
    cam_path = Path(cam_dir)
    if not cam_path.exists():
        return timestamps

    for fpath in cam_path.rglob("*.png"):
        ts = _timestamp_from_filename(fpath.name)
        if ts:
            timestamps.add(ts)
    # Also check jpg
    for fpath in cam_path.rglob("*.jpg"):
        ts = _timestamp_from_filename(fpath.name)
        if ts:
            timestamps.add(ts)
    return timestamps


CLEAR_KEYWORDS = {"clear", "clear_day", "clear_night", "sunny", "sun", "good"}
FOG_KEYWORDS = {"fog", "light_fog", "dense_fog", "heavy_fog", "moderate_fog",
                "rain", "snow", "haze", "mist"}


def classify_weather(label: str) -> str:
    """Classify a weather label as 'clear' or 'adverse'."""
    label = label.lower().strip()
    if any(kw in label for kw in CLEAR_KEYWORDS):
        return "clear"
    if any(kw in label for kw in FOG_KEYWORDS):
        return "adverse"
    # Unknown — conservatively treat as adverse
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Parse STF weather labels")
    parser.add_argument("--stf_root", type=str,
                        default="data/stf/SeeingThroughFog")
    parser.add_argument("--out_dir", type=str, default="data/stf/meta")
    parser.add_argument("--cam_dir", type=str, default=None,
                        help="Camera image dir (default: stf_root/cam_stereo_left)")
    args = parser.parse_args()

    stf = Path(args.stf_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cam_dir = args.cam_dir or str(stf / "cam_stereo_left")

    # --- Parse labels ---
    # Try multiple possible label locations
    label_dirs = [
        stf / "gt_labels" / "cam_left_labels_TMP",
        stf / "gt_labels",
    ]

    weather_map: dict[str, str] = {}
    for ldir in label_dirs:
        if ldir.exists():
            print(f"Scanning labels in {ldir} ...")
            weather_map.update(parse_labels(str(ldir)))

    print(f"Found weather labels for {len(weather_map)} frames")

    # --- Available camera images ---
    cam_timestamps = get_available_timestamps(cam_dir)
    print(f"Available camera images: {len(cam_timestamps)}")

    # --- Cross-reference ---
    labeled = {ts: w for ts, w in weather_map.items() if ts in cam_timestamps}
    unlabeled = cam_timestamps - set(weather_map.keys())
    print(f"Frames with both image + label: {len(labeled)}")
    print(f"Frames with image but no label: {len(unlabeled)}")

    # --- Statistics ---
    weather_counts = Counter(weather_map.values())
    print("\nWeather distribution (all labeled):")
    for w, c in weather_counts.most_common():
        print(f"  {w}: {c}")

    # --- Classify and write outputs ---
    clear_ts = []
    fog_ts = []
    for ts in sorted(labeled.keys()):
        cat = classify_weather(labeled[ts])
        if cat == "clear":
            clear_ts.append(ts)
        else:
            fog_ts.append(ts)

    # If no labels were found, fall back to using ALL camera timestamps
    # (treat all as clear for synthetic-haze training)
    if len(clear_ts) == 0 and len(cam_timestamps) > 0:
        print("\n[WARN] No clear-weather labels found. "
              "Using ALL camera timestamps as clear (for synthetic haze training).")
        clear_ts = sorted(cam_timestamps)

    # Write full weather map
    weather_out = out / "stf_frame_weather.json"
    with open(weather_out, "w") as f:
        json.dump(weather_map, f, indent=2, sort_keys=True)
    print(f"\nSaved weather map -> {weather_out}")

    # Write timestamp lists
    clear_out = out / "clear_timestamps.txt"
    with open(clear_out, "w") as f:
        f.write("\n".join(clear_ts) + "\n")
    print(f"Clear timestamps ({len(clear_ts)}) -> {clear_out}")

    fog_out = out / "fog_timestamps.txt"
    with open(fog_out, "w") as f:
        f.write("\n".join(fog_ts) + "\n")
    print(f"Adverse timestamps ({len(fog_ts)}) -> {fog_out}")

    # Write unlabeled list (can be manually inspected)
    if unlabeled:
        unlabeled_out = out / "unlabeled_timestamps.txt"
        with open(unlabeled_out, "w") as f:
            f.write("\n".join(sorted(unlabeled)) + "\n")
        print(f"Unlabeled timestamps ({len(unlabeled)}) -> {unlabeled_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
