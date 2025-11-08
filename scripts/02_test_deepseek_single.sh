#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/02_test_deepseek_single.sh path/to/image.png
# Runs DeepSeek-OCR locally (CPU-friendly) on a single image.

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

# CPU-friendly defaults
export DEEPSEEK_DEVICE="${DEEPSEEK_DEVICE:-cpu}"
export DEEPSEEK_BASE_SIZE="${DEEPSEEK_BASE_SIZE:-512}"
export DEEPSEEK_IMAGE_SIZE="${DEEPSEEK_IMAGE_SIZE:-512}"
export DEEPSEEK_CROP_MODE="${DEEPSEEK_CROP_MODE:-0}"
# Silence Windows symlink cache warnings from HF hub
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Ensure a writable output directory (required by model code)
export DEEPSEEK_OUTPUT_DIR="${DEEPSEEK_OUTPUT_DIR:-./.tmp/deepseek_ocr}"
mkdir -p "$DEEPSEEK_OUTPUT_DIR"

python - <<PY
from PIL import Image
from src.modules.ocr.deepseek_local import infer_text
img_path = r"$IMG"
img = Image.open(img_path).convert("RGB")
text = infer_text(
	img,
	prompt="<image>\nFree OCR.",
)
print("=== DeepSeek-OCR text ===")
print(text)
PY


