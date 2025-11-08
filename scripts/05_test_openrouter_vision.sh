#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/05_test_openrouter_vision.sh path/to/image.png [mask.png]
# Tests OpenRouter vision client for image editing/inpainting.
# If mask.png is provided, uses it for editing. Otherwise creates a simple test mask.

if [ $# -lt 1 ]; then
	echo "Usage: $0 /path/to/image.(png|jpg|jpeg|webp) [mask.png]" >&2
	exit 1
fi
IMG="$1"
MASK="${2:-}"
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

# Check for OpenRouter API key
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
	echo "Warning: OPENROUTER_API_KEY not set. Set it in .env or export it." >&2
	echo "Continuing anyway..." >&2
fi

OUTPUT_DIR="${OUTPUT_DIR:-./.tmp/openrouter_vision}"
mkdir -p "$OUTPUT_DIR"

python - <<PY
import os
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# Load .env from project root
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
	load_dotenv(dotenv_path=env_path)

from src.ai_client.provider import get_vision_client

img_path = r"$IMG"
mask_path = r"$MASK" if "$MASK" else None
output_dir = r"$OUTPUT_DIR"

img = Image.open(img_path).convert("RGB")
print(f"=== Testing OpenRouter Vision Client ===")
print(f"Image size: {img.size}")

# Create test mask if not provided
if mask_path and os.path.exists(mask_path):
	mask = Image.open(mask_path).convert("L")
	print(f"Using provided mask: {mask_path}")
else:
	# Create a simple rectangular mask in the center
	W, H = img.size
	mask = Image.new("L", (W, H), 0)
	draw = ImageDraw.Draw(mask)
	# Mask a 20% area in the center
	x1, y1 = int(W * 0.4), int(H * 0.4)
	x2, y2 = int(W * 0.6), int(H * 0.6)
	draw.rectangle([x1, y1, x2, y2], fill=255)
	mask_path = os.path.join(output_dir, "test_mask.png")
	mask.save(mask_path)
	print(f"Created test mask: {mask_path}")

# Test vision client
client = get_vision_client()
print(f"Model: {client.model_name}")

# Test image editing
print("\n=== Testing image editing with mask ===")
prompt = "Remove the masked region and inpaint with a seamless background that matches the surrounding area."
print(f"Prompt: {prompt}")

try:
	edited_img = client.edit_image_with_mask(
		image=img,
		mask=mask,
		prompt=prompt
	)
	
	output_path = os.path.join(output_dir, "edited_image.png")
	edited_img.save(output_path)
	print(f"✓ Edited image saved to: {output_path}")
	print(f"  Original size: {img.size}, Edited size: {edited_img.size}")
	print("\n=== Test completed successfully ===")
except Exception as e:
	print(f"\n✗ Error during image editing: {e}")
	print("Note: Some models may not support image editing. Check model compatibility.")
	raise
PY

