#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/03_test_regions.sh path/to/image.png
# Splits the image into a few regions and runs DeepSeek-OCR on each crop.

if [ $# -lt 1 ]; then
	echo "Usage: $0 /path/to/image.(png|jpg|jpeg|webp)" >&2
	exit 1
fi
IMG="$1"
if [ ! -f "$IMG" ]; then
	echo "Image not found: $IMG" >&2
	exit 1
fi

activate_venv() {
	if [ -f ".venv/bin/activate" ]; then
		source ".venv/bin/activate"
	elif [ -f ".venv/Scripts/activate" ]; then
		source ".venv/Scripts/activate"
	else
		echo "Cannot find venv. Run scripts/01_setup_cpu_env.sh first." >&2
		exit 1
	fi
}
activate_venv

export DEEPSEEK_DEVICE="${DEEPSEEK_DEVICE:-cpu}"
export DEEPSEEK_BASE_SIZE="${DEEPSEEK_BASE_SIZE:-512}"
export DEEPSEEK_IMAGE_SIZE="${DEEPSEEK_IMAGE_SIZE:-512}"
export DEEPSEEK_CROP_MODE="${DEEPSEEK_CROP_MODE:-0}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export DEEPSEEK_OUTPUT_DIR="${DEEPSEEK_OUTPUT_DIR:-./.tmp/deepseek_ocr}"
mkdir -p "$DEEPSEEK_OUTPUT_DIR"

python - <<PY
from PIL import Image
from src.modules.ocr.deepseek_local import infer_text

img_path = r"$IMG"
img = Image.open(img_path).convert("RGB")
W, H = img.width, img.height

# Define a few sample regions
regions = [
	(0, 0, W//2, H//3),
	(W//2, 0, W, H//3),
	(0, H//3, W, (2*H)//3),
	(0, (2*H)//3, W, H),
]

for i, box in enumerate(regions, 1):
	crop = img.crop(box)
	text = infer_text(crop, prompt="<image>\\nFree OCR.")
	print(f"\\n=== Region {i} {box} ===")
	print(text or "[no text detected]")
PY


