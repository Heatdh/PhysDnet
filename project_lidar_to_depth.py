#!/usr/bin/env python3
"""
Project LiDAR point clouds into 2D depth maps for the left stereo camera.

STF provides ``lidar_hdl64_strongest_stereo_left`` — point clouds already
transformed into the left camera coordinate frame.  We only need to:
  1. Read the .bin point cloud (Velodyne format: N×4 float32 = x, y, z, intensity)
  2. Project 3D -> 2D using the camera projection matrix P (3×4)
  3. Create a sparse depth image at camera resolution (1920×1024)
  4. Save as .npy (float32, meters; 0 = missing)

Usage:
    python project_lidar_to_depth.py \
        --stf_root data/stf/SeeingThroughFog \
        --out_dir data/stf/projected_depth \
        --workers 8
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def load_calibration(calib_path: str) -> np.ndarray:
    """Load the 3×4 camera projection matrix P from the calibration JSON."""
    with open(calib_path) as f:
        calib = json.load(f)

    # P is stored as a flat 12-element list (row-major 3×4)
    P = np.array(calib["P"]).reshape(3, 4)
    return P


def load_pointcloud(bin_path: str) -> np.ndarray:
    """Load a Velodyne-format binary point cloud: N×4 float32 (x,y,z,intensity)."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def project_to_depth(
    points: np.ndarray,
    P: np.ndarray,
    img_h: int,
    img_w: int,
    min_depth: float = 0.5,
    max_depth: float = 200.0,
) -> np.ndarray:
    """
    Project 3D points (already in camera frame) to a 2D sparse depth image.

    Args:
        points:    N×4 (x, y, z, intensity) in camera coordinates
        P:         3×4 projection matrix
        img_h:     image height in pixels
        img_w:     image width in pixels
        min_depth: minimum valid depth (meters)
        max_depth: maximum valid depth (meters)

    Returns:
        depth_map: H×W float32 array (depth in meters, 0 = missing)
    """
    xyz = points[:, :3]  # N×3
    depth = xyz[:, 2]    # z = depth in camera frame

    # Filter: keep only points in front of camera within valid range
    valid = (depth > min_depth) & (depth < max_depth)
    xyz = xyz[valid]
    depth = depth[valid]

    if len(xyz) == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    # Homogeneous coordinates: N×4
    ones = np.ones((len(xyz), 1), dtype=np.float32)
    xyz_h = np.hstack([xyz, ones])  # N×4

    # Project: pixel_coords = P @ [x, y, z, 1]^T -> 3×N
    proj = (P @ xyz_h.T)  # 3×N
    proj[0, :] /= proj[2, :]  # u = x/z
    proj[1, :] /= proj[2, :]  # v = y/z

    u = np.round(proj[0, :]).astype(np.int32)
    v = np.round(proj[1, :]).astype(np.int32)

    # Keep only points within image bounds
    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[in_bounds]
    v = v[in_bounds]
    depth = depth[in_bounds]

    # Create depth map — for overlapping projections, keep the closest point
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)

    # Sort by depth (farthest first) so closer points overwrite farther ones
    order = np.argsort(-depth)
    u, v, depth = u[order], v[order], depth[order]

    depth_map[v, u] = depth

    return depth_map


def process_single(args_tuple):
    """Process a single .bin file -> .npy depth map."""
    bin_path, out_path, P, img_h, img_w = args_tuple

    if os.path.exists(out_path):
        return bin_path, True, "exists"

    try:
        points = load_pointcloud(bin_path)
        depth_map = project_to_depth(points, P, img_h, img_w)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, depth_map)

        n_valid = int(np.count_nonzero(depth_map))
        return bin_path, True, f"OK ({n_valid} pts)"
    except Exception as e:
        return bin_path, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Project STF LiDAR to sparse depth maps")
    parser.add_argument("--stf_root", type=str,
                        default="data/stf/SeeingThroughFog")
    parser.add_argument("--out_dir", type=str,
                        default="data/stf/projected_depth")
    parser.add_argument("--lidar_segment", type=str,
                        default="lidar_hdl64_strongest_stereo_left",
                        help="Which LiDAR segment to use")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--img_h", type=int, default=1024)
    parser.add_argument("--img_w", type=int, default=1920)
    args = parser.parse_args()

    stf = Path(args.stf_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load calibration
    calib_path = stf / "calib_cam_stereo_left.json"
    assert calib_path.exists(), f"Missing calibration: {calib_path}"
    P = load_calibration(str(calib_path))
    print(f"Projection matrix P:\n{P}")

    # Find .bin files
    lidar_dir = stf / args.lidar_segment
    if not lidar_dir.exists():
        # Maybe extracted into a subdirectory
        lidar_dir2 = lidar_dir / args.lidar_segment
        if lidar_dir2.exists():
            lidar_dir = lidar_dir2

    bin_files = sorted(lidar_dir.rglob("*.bin"))
    print(f"\nFound {len(bin_files)} .bin point cloud files in {lidar_dir}")

    if len(bin_files) == 0:
        print("[ERROR] No .bin files found. Is the archive extracted?")
        return

    # Prepare work items
    tasks = []
    for bf in bin_files:
        stem = bf.stem  # e.g., "1538062105_427018693"
        npy_path = str(out / f"{stem}.npy")
        tasks.append((str(bf), npy_path, P, args.img_h, args.img_w))

    # Process
    print(f"Projecting {len(tasks)} point clouds -> depth maps ...")
    success, fail = 0, 0

    if args.workers <= 1:
        iterator = (process_single(t) for t in tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(process_single, t) for t in tasks]
        iterator = (f.result() for f in as_completed(futures))

    if HAS_TQDM:
        iterator = tqdm(iterator, total=len(tasks), desc="Projecting", unit="file")

    for bin_path, ok, msg in iterator:
        if ok:
            success += 1
        else:
            fail += 1
            print(f"  FAIL: {bin_path}: {msg}")

    print(f"\nDone: {success} success, {fail} failed")
    print(f"Depth maps saved to: {out}")

    # Quick stats on one file
    sample_npy = sorted(out.glob("*.npy"))
    if sample_npy:
        d = np.load(str(sample_npy[0]))
        valid = d[d > 0]
        print(f"\nSample depth map stats ({sample_npy[0].name}):")
        print(f"  Shape: {d.shape}")
        print(f"  Valid pixels: {len(valid)} / {d.size} "
              f"({100*len(valid)/d.size:.2f}%)")
        if len(valid) > 0:
            print(f"  Depth range: {valid.min():.2f} - {valid.max():.2f} m")
            print(f"  Depth mean: {valid.mean():.2f} m")


if __name__ == "__main__":
    main()
