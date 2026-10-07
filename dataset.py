"""
Dataset for LiDAR+RGB dehazing.

Supports three modes:
  1. Pre-generated: load hazy/clear/sparse_depth/mask from disk.
  2. On-the-fly synthesis: load clear + dense depth, synthesize haze & sparsify
     depth at each __getitem__ (preferred for training diversity).
  3. STF mode: load from Seeing-Through-Fog dataset structure with pre-projected
     LiDAR depth maps, synthesize haze on-the-fly from clear-weather frames.
"""

import json
import os
import random
import numpy as np
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class DehazeDataset(Dataset):
    """
    Dataset for training / evaluating the dehazing model.

    Directory layout (on-the-fly mode):
        root/
          clear/    *.png or *.jpg
          depth/    *.npy or *.png (uint16, depth_m * 256)

    Directory layout (pre-generated mode):
        root/
          clear/    *.png
          hazy/     *.png         (same filenames as clear)
          sparse_depth/ *.npy
          mask/         *.npy
    """

    def __init__(
        self,
        root: str,
        mode: Literal["on_the_fly", "pregenerated"] = "on_the_fly",
        img_size: tuple[int, int] = (256, 256),
        beta_range: tuple[float, float] = (0.04, 0.20),
        airlight_range: tuple[float, float] = (0.7, 1.0),
        keep_ratio: float = 0.03,
        augment: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.img_size = img_size  # (H, W)
        self.beta_range = beta_range
        self.airlight_range = airlight_range
        self.keep_ratio = keep_ratio
        self.augment = augment

        # Discover files
        clear_dir = self.root / "clear"
        assert clear_dir.exists(), f"Missing {clear_dir}"
        self.clear_files = sorted(clear_dir.glob("*"))

        if mode == "on_the_fly":
            depth_dir = self.root / "depth"
            assert depth_dir.exists(), f"Missing {depth_dir}"
            self.depth_files = sorted(depth_dir.glob("*"))
            assert len(self.clear_files) == len(self.depth_files)
        else:
            # pregenerated
            self.hazy_dir = self.root / "hazy"
            self.sparse_dir = self.root / "sparse_depth"
            self.mask_dir = self.root / "mask"
            self.hazy_files = sorted(self.hazy_dir.glob("*"))
            # In pregenerated mode, number of hazy files may be > clear files
            # (multiple augmentations per clear). We iterate over hazy files.
            self.clear_files = self.hazy_files  # override

    def __len__(self) -> int:
        return len(self.clear_files)

    @staticmethod
    def _load_depth(path: str) -> np.ndarray:
        if str(path).endswith(".npy"):
            return np.load(path).astype(np.float32)
        d = np.array(Image.open(path)).astype(np.float32)
        if d.max() > 500:
            d /= 256.0
        return d

    def _synthesize_haze(
        self, clear_np: np.ndarray, depth_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Generate hazy image + sparse depth + mask from clear + dense depth."""
        H, W = clear_np.shape[:2]
        beta = np.random.uniform(*self.beta_range)
        A = np.random.uniform(*self.airlight_range, size=(1, 1, 3)).astype(np.float32)

        t_map = np.exp(-beta * depth_np).astype(np.float32)
        t_3 = t_map[..., None]
        hazy = (clear_np * t_3 + A * (1.0 - t_3)).astype(np.float32)
        hazy = np.clip(hazy, 0.0, 1.0)

        # Add light noise
        hazy += np.random.randn(*hazy.shape).astype(np.float32) * 0.01
        hazy = np.clip(hazy, 0.0, 1.0)

        # Sparsify depth
        mask = np.zeros((H, W), dtype=np.float32)
        n_keep = max(1, int(H * W * self.keep_ratio))
        idx = np.random.choice(H * W, n_keep, replace=False)
        mask.flat[idx] = 1.0
        sparse_depth = depth_np * mask

        return hazy, sparse_depth, mask, t_map, beta

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.mode == "on_the_fly":
            # Load clear image
            clear_np = np.array(
                Image.open(self.clear_files[idx]).convert("RGB")
            ).astype(np.float32) / 255.0

            # Load dense depth
            depth_np = self._load_depth(str(self.depth_files[idx]))

            # Resize
            H, W = self.img_size
            clear_pil = Image.fromarray((clear_np * 255).astype(np.uint8)).resize(
                (W, H), Image.BILINEAR)
            clear_np = np.array(clear_pil).astype(np.float32) / 255.0

            depth_pil = Image.fromarray(depth_np).resize((W, H), Image.BILINEAR)
            depth_np = np.array(depth_pil).astype(np.float32)

            hazy_np, sparse_np, mask_np, t_map, beta = self._synthesize_haze(
                clear_np, depth_np
            )
        else:
            # Pregenerated mode
            hazy_np = np.array(
                Image.open(self.hazy_files[idx]).convert("RGB")
            ).astype(np.float32) / 255.0

            stem = self.hazy_files[idx].stem
            sparse_np = np.load(self.sparse_dir / f"{stem}.npy")
            mask_np = np.load(self.mask_dir / f"{stem}.npy")

            # Find matching clear image (stem might have _augN suffix)
            base_stem = stem.rsplit("_aug", 1)[0]
            clear_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                p = self.root / "clear" / f"{base_stem}{ext}"
                if p.exists():
                    clear_path = p
                    break
            assert clear_path is not None, f"No clear image for {stem}"
            clear_np = np.array(
                Image.open(clear_path).convert("RGB")
            ).astype(np.float32) / 255.0

            # Resize all to target size
            H, W = self.img_size
            hazy_np = np.array(Image.fromarray(
                (hazy_np * 255).astype(np.uint8)).resize((W, H))) / 255.0
            clear_np = np.array(Image.fromarray(
                (clear_np * 255).astype(np.uint8)).resize((W, H))) / 255.0
            sparse_np = np.array(Image.fromarray(sparse_np).resize((W, H)))
            mask_np = np.array(Image.fromarray(mask_np).resize((W, H)))
            mask_np = (mask_np > 0.5).astype(np.float32)

        # Augmentation (training only)
        if self.augment:
            if np.random.rand() > 0.5:
                hazy_np = np.fliplr(hazy_np).copy()
                clear_np = np.fliplr(clear_np).copy()
                sparse_np = np.fliplr(sparse_np).copy()
                mask_np = np.fliplr(mask_np).copy()

        # To tensors  (C, H, W)
        hazy_t = torch.from_numpy(hazy_np.transpose(2, 0, 1)).float()
        clear_t = torch.from_numpy(clear_np.transpose(2, 0, 1)).float()
        sparse_t = torch.from_numpy(sparse_np[None]).float()
        mask_t = torch.from_numpy(mask_np[None]).float()

        return {
            "hazy": hazy_t,         # 3 x H x W
            "clear": clear_t,       # 3 x H x W
            "sparse_depth": sparse_t,  # 1 x H x W
            "mask": mask_t,         # 1 x H x W
        }


# ---------------------------------------------------------------------------
# Dummy / demo dataset (generates random data for quick pipeline testing)
# ---------------------------------------------------------------------------

class DummyDehazeDataset(Dataset):
    """
    Generates random synthetic data on-the-fly for pipeline testing.
    No files needed. Useful to verify train loop works end-to-end.
    """

    def __init__(self, length: int = 200, img_size: tuple[int, int] = (256, 256)):
        self.length = length
        self.H, self.W = img_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Random "clear" image
        clear = torch.rand(3, self.H, self.W)

        # Random smooth depth (simulate a scene)
        depth = torch.rand(1, self.H // 8, self.W // 8)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(self.H, self.W), mode="bilinear"
        ).squeeze(0)
        depth = depth * 10.0 + 1.0  # range ~1-11 meters

        # Synthesize haze
        beta = np.random.uniform(0.04, 0.20)
        A = np.random.uniform(0.7, 1.0, size=(3, 1, 1)).astype(np.float32)
        A_t = torch.from_numpy(A)

        t_map = torch.exp(-beta * depth)  # 1 x H x W
        hazy = clear * t_map + A_t * (1.0 - t_map)
        hazy = torch.clamp(hazy, 0, 1)

        # Sparse depth
        mask = (torch.rand(1, self.H, self.W) > 0.97).float()
        sparse_depth = depth * mask

        return {
            "hazy": hazy,
            "clear": clear,
            "sparse_depth": sparse_depth,
            "mask": mask,
        }


if __name__ == "__main__":
    ds = DummyDehazeDataset(length=10, img_size=(128, 128))
    sample = ds[0]
    for k, v in sample.items():
        print(f"{k}: {v.shape}, range [{v.min():.2f}, {v.max():.2f}]")


# ---------------------------------------------------------------------------
# STF (Seeing-Through-Fog) Dataset
# ---------------------------------------------------------------------------

class STFDehazeDataset(Dataset):
    """
    Seeing-Through-Fog dataset for LiDAR-guided dehazing.

    Uses clear-weather frames + LiDAR sparse depth to synthesize haze
    on-the-fly for supervised training.  The exact same atmospheric
    scattering model is applied: I(x) = J(x)*t(x) + A*(1-t(x)).

    Expected directory structure (after extraction):
        stf_root/
            cam_stereo_left/                          *.tiff (1920x1024)
            lidar_hdl64_strongest_stereo_left/         *.npz  (1024x1920 depth)

    The .npz files contain pre-projected sparse depth maps (key: 'arr_0',
    shape 1024x1920, float32, values in meters).

    Args:
        stf_root:        Path to SeeingThroughFog/ directory
        depth_dir:       Path to depth maps (.npz); defaults to
                         stf_root/lidar_hdl64_strongest_stereo_left
        timestamps_file: Path to timestamps list (one per line);
                         if None, uses all frames with matched pairs
        crop_size:       Random crop (H, W) for training; None = full res
        beta_range:      Scattering coefficient range for haze synthesis
        airlight_range:  Atmospheric light range [0, 1]
        augment:         Apply random flip + color jitter
        use_all_if_no_labels: If True and timestamp file not found, use all
                              available images.
    """

    def __init__(
        self,
        stf_root: str,
        depth_dir: str | None = None,
        timestamps_file: str | None = None,
        crop_size: tuple[int, int] | None = (512, 512),
        beta_range: tuple[float, float] = (0.005, 0.04),
        airlight_range: tuple[float, float] = (0.7, 1.0),
        max_depth: float = 120.0,
        augment: bool = True,
        use_all_if_no_labels: bool = True,
        weather_filter: list[str] | None = None,
    ):
        super().__init__()
        self.stf_root = Path(stf_root)
        self.weather_filter = weather_filter  # e.g. ["clear", "overcast"]
        # Default: LiDAR pre-projected depth lives alongside camera data
        if depth_dir is None:
            depth_dir = str(self.stf_root / "lidar_hdl64_strongest_stereo_left")
        self.depth_dir = Path(depth_dir)
        self.crop_size = crop_size
        self.max_depth = max_depth
        self.beta_range = beta_range
        self.airlight_range = airlight_range
        self.augment = augment

        # Find camera image directory (may be nested)
        self.cam_dir = self._find_cam_dir()

        # Load weather data if filtering is requested
        self._weather_data = {}
        if self.weather_filter is not None:
            self._weather_data = self._load_weather()
            print(f"[STF] Weather filter: {self.weather_filter} "
                  f"({len(self._weather_data)} weather records loaded)")

        # Discover available data
        self.samples = self._build_sample_list(timestamps_file,
                                                use_all_if_no_labels)

        # Color jitter for augmentation (mild)
        self.color_jitter = T.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        ) if augment else None

    def _load_weather(self) -> dict[str, dict]:
        """Load weather station JSONs and return {stem: weather_dict}."""
        weather_dir = self.stf_root / "weather_station" / "weather_station"
        if not weather_dir.exists():
            # Also try flat layout
            weather_dir = self.stf_root / "weather_station"
        if not weather_dir.exists():
            print(f"[STF] WARNING: weather_station dir not found at {weather_dir}")
            return {}
        weather = {}
        for f in weather_dir.glob("*.json"):
            with open(f) as fp:
                weather[f.stem] = json.load(fp)
        return weather

    @staticmethod
    def classify_weather(w: dict) -> str:
        """
        Classify a weather record into a category.

        Categories: clear, overcast, light_fog, dense_fog, snow, rain
        Based on humidity, temperature-dewpoint spread, rain intensity.
        """
        humidity = w.get("outHumidity", 0)
        temp_f   = w.get("outTemp", 32)
        dewpoint = w.get("dewpoint", 0)
        rain     = w.get("rain_intensity", 0)
        temp_c   = (temp_f - 32) * 5.0 / 9.0
        spread   = temp_f - dewpoint  # °F spread

        if rain > 0.1:
            return "rain"
        if humidity >= 85 and spread < 5:
            return "dense_fog"
        if humidity >= 75 and spread < 8:
            return "light_fog"
        if temp_c < 0 and humidity >= 70:
            return "snow"
        if humidity < 65 and spread > 12:
            return "clear"
        return "overcast"

    def _find_cam_dir(self) -> Path:
        """Locate the camera image directory (handles nested extraction)."""
        candidates = [
            self.stf_root / "cam_stereo_left" / "cam_stereo_left",
            self.stf_root / "cam_stereo_left",
        ]
        # Check for .tiff first (STF default), then .png
        for ext in ("*.tiff", "*.png"):
            for d in candidates:
                if d.exists() and any(d.glob(ext)):
                    return d
        raise FileNotFoundError(
            f"Camera images not found (.tiff/.png). "
            f"Checked: {[str(c) for c in candidates]}"
        )

    def _build_sample_list(
        self, timestamps_file: str | None, use_all: bool
    ) -> list[dict]:
        """Build list of {timestamp, cam_path, depth_path} dicts."""

        # Get available depth maps (.npz from STF or .npy from projection)
        depth_files = {}
        for ext in ("*.npz", "*.npy"):
            for p in self.depth_dir.glob(ext):
                depth_files[p.stem] = p

        # Get requested timestamps
        if timestamps_file and os.path.exists(timestamps_file):
            with open(timestamps_file) as f:
                requested_ts = [line.strip() for line in f
                                if line.strip()]
            print(f"[STF] Loaded {len(requested_ts)} timestamps "
                  f"from {timestamps_file}")
        elif use_all:
            # Use all camera images that have a matching depth map
            requested_ts = sorted(depth_files.keys())
            print(f"[STF] No timestamp file. Using all {len(requested_ts)} "
                  "frames with depth maps.")
        else:
            raise FileNotFoundError(
                f"Timestamps file not found: {timestamps_file}")

        # Match camera image + depth map + optional weather filter
        samples = []
        missing_cam = 0
        missing_depth = 0
        corrupt = 0
        weather_filtered = 0

        for ts in requested_ts:
            # Weather filter: skip frames whose condition is not in the list
            if self.weather_filter and self._weather_data:
                w = self._weather_data.get(ts)
                if w is None:
                    weather_filtered += 1
                    continue
                cat = self.classify_weather(w)
                if cat not in self.weather_filter:
                    weather_filtered += 1
                    continue

            # Camera image — try .tiff first, then .png
            cam_path = self.cam_dir / f"{ts}.tiff"
            if not cam_path.exists():
                cam_path = self.cam_dir / f"{ts}.png"
            if not cam_path.exists():
                missing_cam += 1
                continue

            # Depth map
            if ts not in depth_files:
                missing_depth += 1
                continue

            # Quick validity check (skip corrupt / non-image files)
            try:
                with Image.open(str(cam_path)) as _img:
                    _img.verify()
            except Exception:
                corrupt += 1
                continue

            samples.append({
                "timestamp": ts,
                "cam_path": str(cam_path),
                "depth_path": str(depth_files[ts]),
            })

        if missing_cam > 0:
            print(f"[STF] Skipped {missing_cam} frames: camera image missing")
        if missing_depth > 0:
            print(f"[STF] Skipped {missing_depth} frames: depth map missing")
        if corrupt > 0:
            print(f"[STF] Skipped {corrupt} frames: corrupt / unreadable image")
        if weather_filtered > 0:
            print(f"[STF] Skipped {weather_filtered} frames: weather filter")
        print(f"[STF] Final dataset: {len(samples)} samples")

        if len(samples) == 0:
            raise RuntimeError(
                "No valid samples found! Check paths and ensure archives "
                "are extracted and depth maps are projected."
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_stf_image(path: str) -> np.ndarray:
        """
        Load an STF camera image and return H×W×3 float32 in [0, 1].

        STF uses a 12-bit gated camera stored as 16-bit grayscale TIFF
        (PIL mode 'I;16', dtype uint16, value range ≈ 0–4095).
        We normalise to [0,1] and replicate to 3 channels.
        Also handles standard 8-bit RGB/PNG gracefully.
        """
        if str(path).lower().endswith((".tif", ".tiff")):
            # Pillow >= 12 segfaults on STF's 16-bit TIFFs — decode via cv2
            import cv2
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise IOError(f"cv2 failed to read {path}")
            raw = raw.astype(np.float32)
            if raw.ndim == 3:  # BGR -> grey (STF tiffs are single-channel)
                raw = raw.mean(axis=2)
            max_val = raw.max()
            if max_val <= 0:
                scale = 1.0
            elif max_val <= 255:
                scale = 255.0
            elif max_val <= 4095:
                scale = 4095.0
            else:
                scale = 65535.0
            grey = np.clip(raw / scale, 0.0, 1.0)
            return np.stack([grey, grey, grey], axis=-1)

        pil = Image.open(path)
        if pil.mode in ('I;16', 'I'):
            # 16-bit grayscale (12-bit sensor) → float [0, 1]
            raw = np.array(pil).astype(np.float32)
            # auto-detect bit depth from max value
            max_val = raw.max()
            if max_val <= 0:
                scale = 1.0
            elif max_val <= 255:
                scale = 255.0
            elif max_val <= 4095:
                scale = 4095.0
            else:
                scale = 65535.0
            grey = np.clip(raw / scale, 0.0, 1.0)  # H×W
            return np.stack([grey, grey, grey], axis=-1)  # H×W×3
        else:
            # Standard 8-bit RGB/L
            rgb = pil.convert('RGB')
            return np.array(rgb).astype(np.float32) / 255.0

    def _synthesize_haze(
        self, clear_np: np.ndarray, depth_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Synthesize hazy image from clear image + sparse depth.

        Returns:
            hazy:  H x W x 3  float32 [0,1]
            t_map: H x W      float32 transmission
            beta:  float      scattering coefficient
            A:     1x1x3      airlight
        """
        beta = np.random.uniform(*self.beta_range)
        A = np.random.uniform(*self.airlight_range,
                              size=(1, 1, 3)).astype(np.float32)

        # For haze synthesis we need dense-ish depth.  Our depth is sparse
        # (LiDAR projection).  We fill missing pixels with a simple NN
        # interpolation for the transmission computation only.
        depth_filled = self._fill_depth_nn(depth_np)

        t_map = np.exp(-beta * depth_filled).astype(np.float32)
        t_3 = t_map[..., None]
        hazy = (clear_np * t_3 + A * (1.0 - t_3)).astype(np.float32)
        hazy = np.clip(hazy, 0.0, 1.0)

        # Light camera noise
        noise_sigma = np.random.uniform(0.005, 0.015)
        hazy += np.random.randn(*hazy.shape).astype(np.float32) * noise_sigma
        hazy = np.clip(hazy, 0.0, 1.0)

        return hazy, t_map, beta, A

    @staticmethod
    def _fill_depth_nn(depth: np.ndarray, default_depth: float = 50.0) -> np.ndarray:
        """
        Fill missing depth values (0) with nearest-neighbor interpolation.

        Uses scipy if available, otherwise fills with a constant.
        This is for haze synthesis only — the model still sees sparse depth.
        """
        if np.count_nonzero(depth) == 0:
            return np.full_like(depth, default_depth)

        try:
            from scipy.ndimage import distance_transform_edt
            mask = depth > 0
            _, indices = distance_transform_edt(~mask, return_distances=True,
                                                 return_indices=True)
            filled = depth[tuple(indices)]
            return filled
        except ImportError:
            # Fallback: fill missing with median of valid
            valid = depth[depth > 0]
            fill_val = np.median(valid) if len(valid) > 0 else default_depth
            filled = depth.copy()
            filled[filled == 0] = fill_val
            return filled

    def _random_crop(
        self,
        *arrays: np.ndarray,
        crop_h: int,
        crop_w: int,
    ) -> list[np.ndarray]:
        """Apply the same random crop to all arrays."""
        h, w = arrays[0].shape[:2]
        if h <= crop_h:
            top = 0
        else:
            top = np.random.randint(0, h - crop_h)
        if w <= crop_w:
            left = 0
        else:
            left = np.random.randint(0, w - crop_w)

        results = []
        for arr in arrays:
            if arr.ndim == 3:
                results.append(arr[top:top+crop_h, left:left+crop_w, :])
            else:
                results.append(arr[top:top+crop_h, left:left+crop_w])
        return results

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load clear image (handles 16-bit TIFF from gated camera)
        clear_np = self._load_stf_image(sample["cam_path"])

        # Load sparse depth map (.npz with 'arr_0' key, or .npy)
        depth_raw = np.load(sample["depth_path"])
        if isinstance(depth_raw, np.lib.npyio.NpzFile):
            depth_np = depth_raw["arr_0"].astype(np.float32)
        else:
            depth_np = depth_raw.astype(np.float32)

        # Synthesize haze
        hazy_np, t_map, beta, A = self._synthesize_haze(clear_np, depth_np)

        # The model input: original sparse depth (NOT the filled version)
        # Normalize depth to [0, 1] by dividing by max_depth
        sparse_depth = np.clip(depth_np.copy() / self.max_depth, 0.0, 1.0)
        mask = (depth_np > 0).astype(np.float32)

        # Random crop (include t_map)
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            hazy_np, clear_np, sparse_depth, mask, t_map = self._random_crop(
                hazy_np, clear_np, sparse_depth, mask, t_map,
                crop_h=crop_h, crop_w=crop_w,
            )

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                hazy_np = np.fliplr(hazy_np).copy()
                clear_np = np.fliplr(clear_np).copy()
                sparse_depth = np.fliplr(sparse_depth).copy()
                mask = np.fliplr(mask).copy()
                t_map = np.fliplr(t_map).copy()

        # To tensors (C, H, W)
        hazy_t = torch.from_numpy(hazy_np.transpose(2, 0, 1)).float()
        clear_t = torch.from_numpy(clear_np.transpose(2, 0, 1)).float()
        sparse_t = torch.from_numpy(sparse_depth[None]).float()
        mask_t = torch.from_numpy(mask[None]).float()
        trans_gt_t = torch.from_numpy(t_map[None]).float()  # 1 x H x W

        # Color jitter on hazy input only (simulates varying camera exposure)
        if self.color_jitter is not None and np.random.rand() > 0.5:
            hazy_t = self.color_jitter(hazy_t)
            hazy_t = torch.clamp(hazy_t, 0.0, 1.0)

        return {
            "hazy": hazy_t,           # 3 x H x W
            "clear": clear_t,         # 3 x H x W
            "sparse_depth": sparse_t, # 1 x H x W  (normalized [0,1])
            "mask": mask_t,           # 1 x H x W
            "trans_gt": trans_gt_t,   # 1 x H x W  (GT transmission)
        }

    @staticmethod
    def make_train_val_split(
        timestamps_file: str,
        out_dir: str,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Split a timestamps file into train and val sets.

        Saves:
            out_dir/train_timestamps.txt
            out_dir/val_timestamps.txt
        """
        with open(timestamps_file) as f:
            timestamps = [l.strip() for l in f if l.strip()]

        random.seed(seed)
        random.shuffle(timestamps)

        n_val = max(1, int(len(timestamps) * val_ratio))
        val_ts = sorted(timestamps[:n_val])
        train_ts = sorted(timestamps[n_val:])

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train_timestamps.txt"), "w") as f:
            f.write("\n".join(train_ts) + "\n")
        with open(os.path.join(out_dir, "val_timestamps.txt"), "w") as f:
            f.write("\n".join(val_ts) + "\n")

        print(f"Split: {len(train_ts)} train / {len(val_ts)} val "
              f"(seed={seed}, ratio={val_ratio})")
        return train_ts, val_ts
