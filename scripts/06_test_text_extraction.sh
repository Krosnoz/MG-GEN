#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/06_test_text_extraction.sh path/to/image.png
# Tests text extraction flow: OCR + optional DeepSeek refinement.

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

# Change to project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-./.tmp/text_extraction}"
mkdir -p "$OUTPUT_DIR"

python - <<PY
import os
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
	load_dotenv(dotenv_path=env_path)

img_path = r"$IMG"
output_dir = r"$OUTPUT_DIR"

img = Image.open(img_path).convert("RGB")
print(f"=== Testing Text Extraction Flow ===")
print(f"Image size: {img.size}")

# Test PaddleOCR
print("\n=== Step 1: PaddleOCR ===")
try:
	from src.modules.ocr.main import PaddleOCRClient
	
	ocr_client = PaddleOCRClient()
	ocr_text_list, ocr_text_mask = ocr_client.run_ocr(img)
	
	print(f"Found {len(ocr_text_list)} text regions:")
	for i, item in enumerate(ocr_text_list, 1):
		text = item.get("text", "")
		vertices = item.get("vertices", [])
		print(f"  {i}. Text: '{text}'")
		if vertices:
			x_coords = [v['x'] for v in vertices]
			y_coords = [v['y'] for v in vertices]
			print(f"     BBox: ({min(x_coords)}, {min(y_coords)}) to ({max(x_coords)}, {max(y_coords)})")
	
	ocr_mask_path = os.path.join(output_dir, "ocr_mask.png")
	ocr_text_mask.save(ocr_mask_path)
	print(f"✓ OCR mask saved to: {ocr_mask_path}")
	
except Exception as e:
	print(f"✗ PaddleOCR failed: {e}")
	ocr_text_list = []
	ocr_text_mask = None

# Test DeepSeek OCR refinement (if enabled and OCR found text)
if len(ocr_text_list) > 0:
	print("\n=== Step 2: DeepSeek OCR Refinement (optional) ===")
	try:
		from src.modules.ocr.deepseek_refine import DeepSeekOCRRefiner
		
		# CPU-friendly defaults
		os.environ.setdefault("DEEPSEEK_DEVICE", "cpu")
		os.environ.setdefault("DEEPSEEK_BASE_SIZE", "512")
		os.environ.setdefault("DEEPSEEK_IMAGE_SIZE", "512")
		os.environ.setdefault("DEEPSEEK_CROP_MODE", "0")
		os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
		
		refiner = DeepSeekOCRRefiner()
		refined_list = refiner.refine(ocr_text_list, img)
		
		print(f"Refined {len(refined_list)} text regions:")
		for i, (original, refined) in enumerate(zip(ocr_text_list, refined_list), 1):
			orig_text = original.get("text", "")
			refined_text = refined.get("text", "")
			if orig_text != refined_text:
				print(f"  {i}. '{orig_text}' -> '{refined_text}'")
			else:
				print(f"  {i}. '{orig_text}' (unchanged)")
		
		print("✓ DeepSeek refinement completed")
	except Exception as e:
		print(f"⚠ DeepSeek refinement skipped: {e}")
		print("  (This is optional and may require model download)")
else:
	print("\n=== Step 2: Skipped (no text found) ===")

print("\n=== Test completed ===")
PY

