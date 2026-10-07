"""
Selective downloader for the Seeing Through Fog (STF) dataset.

Downloads ONLY the segments needed for LiDAR+RGB dehazing:
  - cam_stereo_left      (RGB images)
  - lidar_hdl64_strongest (LiDAR point clouds)
  - lidar_hdl64_strongest_stereo_left (LiDAR projected to left camera)
  - calib_*              (calibration files)
  - gt_labels            (weather condition labels, for filtering fog/clear)

Usage:
    # Download all needed segments:
    python download_stf.py --json data_download.json --out_dir data/stf

    # Dry run (show what would be downloaded):
    python download_stf.py --json data_download.json --out_dir data/stf --dry_run

    # Download only LiDAR + calibration (skip images for now):
    python download_stf.py --json data_download.json --out_dir data/stf \
        --segments lidar_hdl64_strongest calib gt_labels

    # Limit parallel downloads:
    python download_stf.py --json data_download.json --out_dir data/stf --workers 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] Install tqdm for progress bars: pip install tqdm")

# Segments needed for dehazing
DEFAULT_SEGMENTS = [
    "cam_stereo_left",                     # RGB images (split zip)
    "lidar_hdl64_strongest",               # LiDAR 64-beam point clouds
    "lidar_hdl64_strongest_stereo_left",   # LiDAR projected to left cam frame
    "calib_cam_stereo_left.json",          # Camera intrinsics
    "calib_tf_tree_full.json",             # Full transform tree (LiDAR→cam extrinsics)
    "gt_labels",                           # Weather labels (fog/clear/rain)
]

# Segments you do NOT need (for reference):
SKIP_EXPLANATION = {
    "cam_stereo_right":          "Right stereo camera — not needed for single-cam dehazing",
    "cam_stereo_left_lut":       "Lookup-table corrected — raw is fine",
    "cam_stereo_left_raw_history_*": "Temporal history frames — not needed for single-frame",
    "cam_stereo_sgm":            "Precomputed stereo disparity — we use LiDAR instead",
    "psmnet_sweden_stf":         "PSMNet stereo predictions — not needed",
    "gated*":                    "Gated NIR camera — different modality",
    "cam_stereo_right_lut":      "Right camera LUT — not needed",
    "radar_targets":             "Radar — not our input modality",
    "fir_axis":                  "Far-infrared camera — not needed",
    "road_friction":             "Vehicle telemetry",
    "weather_station":           "Weather station data (could be useful for metadata)",
    "filtered_relevant_can_data":"CAN bus data — not needed",
}


def matches_segment(key: str, segments: list[str]) -> bool:
    """Check if a key belongs to one of the desired segments."""
    # Strip the top-level prefix (e.g. "SeeingThroughFog/")
    parts = key.split("/", 1)
    if len(parts) < 2:
        return False
    subpath = parts[1]

    for seg in segments:
        # Exact match for calibration files
        if subpath == seg:
            return True
        # Prefix match for folders (cam_stereo_left/*, lidar_hdl64_strongest/*)
        if subpath.startswith(seg + "/"):
            return True
        # Also match the folder entry itself
        if subpath.rstrip("/") == seg:
            return True
    return False


def format_size(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def get_remote_size(url: str) -> int | None:
    """Get file size from Content-Length header (HEAD request)."""
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=10) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def _download_with_progress(url: str, dest: str, desc: str) -> int:
    """Download a file with a per-file tqdm progress bar showing bytes."""
    req = Request(url)
    with urlopen(req, timeout=120) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        block_size = 1024 * 64  # 64 KB

        if HAS_TQDM:
            pbar = tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=desc, leave=False, ncols=100,
            )
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if HAS_TQDM:
                    pbar.update(len(chunk))
        if HAS_TQDM:
            pbar.close()
    return downloaded


def download_file(url: str, dest: str, key: str) -> tuple[str, bool, str]:
    """Download a single file. Returns (key, success, message)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Skip if already downloaded
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        return (key, True, f"already exists ({format_size(size)})")

    short_name = key.split("/")[-1][:30]
    try:
        nbytes = _download_with_progress(url, dest, desc=short_name)
        return (key, True, f"downloaded ({format_size(nbytes)})")
    except (URLError, HTTPError) as e:
        # Clean up partial file
        if os.path.exists(dest):
            os.remove(dest)
        return (key, False, f"FAILED: {e}")
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        return (key, False, f"FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download STF dataset (dehazing-relevant segments only)"
    )
    parser.add_argument("--json", type=str, default="data_download.json",
                        help="Path to data_download.json with presigned URLs")
    parser.add_argument("--out_dir", type=str, default="data/stf",
                        help="Output directory")
    parser.add_argument("--segments", nargs="+", default=None,
                        help="Override which segments to download (default: all needed)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only show what would be downloaded")
    parser.add_argument("--check_sizes", action="store_true",
                        help="Query remote file sizes (slower, needs HEAD requests)")
    parser.add_argument("--all", action="store_true",
                        help="Download ALL segments (not just dehazing-relevant)")
    args = parser.parse_args()

    segments = args.segments or DEFAULT_SEGMENTS

    # Load URLs
    with open(args.json) as f:
        data = json.load(f)

    all_items = data["urls"]
    print(f"Total entries in JSON: {len(all_items)}")

    # Filter
    if args.all:
        filtered = [item for item in all_items if item["key"].count("/") >= 1]
        print(f"Downloading ALL segments")
    else:
        filtered = [item for item in all_items if matches_segment(item["key"], segments)]
        print(f"Segments selected: {segments}")

    # Skip directory entries (keys ending with /)
    filtered = [item for item in filtered if not item["key"].endswith("/")]

    print(f"Files to download: {len(filtered)}")

    if not filtered:
        print("No files matched. Check segment names or JSON contents.")
        sys.exit(1)

    # Show what will be downloaded
    print(f"\nFiles:")
    total_size = 0
    for item in filtered:
        key = item["key"]
        size_str = ""
        if args.check_sizes:
            sz = get_remote_size(item["url"])
            if sz:
                total_size += sz
                size_str = f"  ({format_size(sz)})"
        dest = os.path.join(args.out_dir, key)
        exists = " [EXISTS]" if os.path.exists(dest) else ""
        print(f"  {key}{size_str}{exists}")

    if args.check_sizes and total_size > 0:
        print(f"\nEstimated total download: {format_size(total_size)}")

    if args.dry_run:
        print("\n[DRY RUN] No files downloaded.")
        return

    # Download
    print(f"\nDownloading to {args.out_dir}/ with {args.workers} threads...")
    t0 = time.time()
    success_count = 0
    fail_count = 0
    total_bytes = 0

    # Overall progress bar (file count)
    if HAS_TQDM:
        overall_pbar = tqdm(
            total=len(filtered), unit="file", desc="Overall",
            position=0, ncols=100,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for item in filtered:
            key = item["key"]
            url = item["url"]
            dest = os.path.join(args.out_dir, key)
            fut = pool.submit(download_file, url, dest, key)
            futures[fut] = key

        for fut in as_completed(futures):
            key, ok, msg = fut.result()
            status = "\u2713" if ok else "\u2717"
            if HAS_TQDM:
                overall_pbar.set_postfix_str(key.split("/")[-1][:25])
                overall_pbar.update(1)
            else:
                i = success_count + fail_count + 1
                print(f"  [{i}/{len(filtered)}] [{status}] {key}: {msg}")

            if ok:
                success_count += 1
            else:
                fail_count += 1
                if HAS_TQDM:
                    tqdm.write(f"  [\u2717 FAIL] {key}: {msg}")

    if HAS_TQDM:
        overall_pbar.close()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Success: {success_count}/{len(filtered)}")
    print(f"  Failed:  {fail_count}")
    print(f"{'='*60}")

    if fail_count > 0:
        print("\nSome downloads failed. Presigned URLs may have expired (5-day TTL).")
        print("Re-generate data_download.json from the STF download page and retry.")


if __name__ == "__main__":
    main()
