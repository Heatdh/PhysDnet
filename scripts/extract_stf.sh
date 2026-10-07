#!/usr/bin/env bash
# ============================================================================
# Extract STF dataset split-zip archives
#
# Usage:
#   chmod +x extract_stf.sh
#   ./extract_stf.sh [STF_ROOT]
#
# Default STF_ROOT: data/stf/SeeingThroughFog
# ============================================================================
set -euo pipefail

STF_ROOT="${1:-data/stf/SeeingThroughFog}"

echo "=== STF Archive Extraction ==="
echo "Root: $STF_ROOT"
echo ""

# ---------- cam_stereo_left (26 split-zip parts) ----------
CAM_DIR="$STF_ROOT/cam_stereo_left"
if [ -d "$CAM_DIR" ] && ls "$CAM_DIR"/*.z01 1>/dev/null 2>&1; then
    if [ -d "$CAM_DIR/cam_stereo_left" ] || ls "$CAM_DIR"/*.png 1>/dev/null 2>&1; then
        echo "[SKIP] cam_stereo_left already extracted"
    else
        echo "[1/4] Combining cam_stereo_left split archive..."
        cd "$CAM_DIR"
        zip -s 0 cam_stereo_left.zip --out cam_stereo_left_combined.zip
        echo "  Extracting cam_stereo_left_combined.zip ..."
        unzip -o cam_stereo_left_combined.zip
        rm -f cam_stereo_left_combined.zip
        echo "  Done. Files: $(find . -name '*.png' | wc -l) PNGs"
        cd - > /dev/null
    fi
else
    echo "[SKIP] cam_stereo_left archive not found"
fi
echo ""

# ---------- lidar_hdl64_strongest (7 split-zip parts) ----------
LIDAR_DIR="$STF_ROOT/lidar_hdl64_strongest"
if [ -d "$LIDAR_DIR" ] && ls "$LIDAR_DIR"/*.z01 1>/dev/null 2>&1; then
    if [ -d "$LIDAR_DIR/lidar_hdl64_strongest" ] || ls "$LIDAR_DIR"/*.bin 1>/dev/null 2>&1; then
        echo "[SKIP] lidar_hdl64_strongest already extracted"
    else
        echo "[2/4] Combining lidar_hdl64_strongest split archive..."
        cd "$LIDAR_DIR"
        zip -s 0 lidar_hdl64_strongest.zip --out lidar_hdl64_strongest_combined.zip
        echo "  Extracting lidar_hdl64_strongest_combined.zip ..."
        unzip -o lidar_hdl64_strongest_combined.zip
        rm -f lidar_hdl64_strongest_combined.zip
        echo "  Done. Files: $(find . -name '*.bin' | wc -l) BINs"
        cd - > /dev/null
    fi
else
    echo "[SKIP] lidar_hdl64_strongest archive not found"
fi
echo ""

# ---------- lidar_hdl64_strongest_stereo_left (single zip) ----------
LIDAR_SL_DIR="$STF_ROOT/lidar_hdl64_strongest_stereo_left"
if [ -d "$LIDAR_SL_DIR" ] && [ -f "$LIDAR_SL_DIR/lidar_hdl64_strongest_stereo_left.zip" ]; then
    if [ -d "$LIDAR_SL_DIR/lidar_hdl64_strongest_stereo_left" ] || find "$LIDAR_SL_DIR" -name '*.bin' | head -1 | grep -q .; then
        echo "[SKIP] lidar_hdl64_strongest_stereo_left already extracted"
    else
        echo "[3/4] Extracting lidar_hdl64_strongest_stereo_left.zip ..."
        cd "$LIDAR_SL_DIR"
        unzip -o lidar_hdl64_strongest_stereo_left.zip
        echo "  Done. Files: $(find . -name '*.bin' | wc -l) BINs"
        cd - > /dev/null
    fi
else
    echo "[SKIP] lidar_hdl64_strongest_stereo_left archive not found"
fi
echo ""

# ---------- gt_labels (2 zips) ----------
LABELS_DIR="$STF_ROOT/gt_labels"
if [ -d "$LABELS_DIR" ]; then
    # cam_left_labels_TMP.zip
    if [ -f "$LABELS_DIR/cam_left_labels_TMP.zip" ]; then
        if [ -d "$LABELS_DIR/cam_left_labels_TMP" ]; then
            echo "[SKIP] cam_left_labels_TMP already extracted"
        else
            echo "[4a/4] Extracting cam_left_labels_TMP.zip ..."
            cd "$LABELS_DIR"
            unzip -o cam_left_labels_TMP.zip
            echo "  Done."
            cd - > /dev/null
        fi
    fi
    # gated_labels_TMPv2.zip (optional — we primarily need cam labels)
    if [ -f "$LABELS_DIR/gated_labels_TMPv2.zip" ]; then
        if [ -d "$LABELS_DIR/gated_labels_TMPv2" ]; then
            echo "[SKIP] gated_labels_TMPv2 already extracted"
        else
            echo "[4b/4] Extracting gated_labels_TMPv2.zip ..."
            cd "$LABELS_DIR"
            unzip -o gated_labels_TMPv2.zip
            echo "  Done."
            cd - > /dev/null
        fi
    fi
else
    echo "[SKIP] gt_labels directory not found"
fi

echo ""
echo "=== Extraction complete ==="
echo ""
echo "Summary:"
for d in cam_stereo_left lidar_hdl64_strongest lidar_hdl64_strongest_stereo_left gt_labels; do
    path="$STF_ROOT/$d"
    if [ -d "$path" ]; then
        count=$(find "$path" -type f \( -name '*.png' -o -name '*.bin' -o -name '*.json' -o -name '*.txt' \) 2>/dev/null | wc -l)
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  $d: $count data files, $size total"
    fi
done
