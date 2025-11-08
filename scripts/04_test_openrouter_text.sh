#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/04_test_openrouter_text.sh path/to/image.png
# Tests OpenRouter text client for generating anime.js animation scripts.

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

# Check for OpenRouter API key
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
	echo "Warning: OPENROUTER_API_KEY not set. Set it in .env or export it." >&2
	echo "Continuing anyway..." >&2
fi

python - <<PY
import os
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
	load_dotenv(dotenv_path=env_path)

from src.ai_client.provider import get_text_client

img_path = r"$IMG"
img = Image.open(img_path).convert("RGB")

# Test OpenRouter text client
print("=== Testing OpenRouter Text Client ===")
client = get_text_client()
print(f"Model: {client.model_name}")

# Upload image
image_base64 = client.upload_image(img)
print(f"Image uploaded (base64 length: {len(image_base64)})")

# Simple test prompt
messages = [
	{
		"role": "user",
		"content": [
			{"type": "text", "text": "Describe this image briefly. What text and elements do you see?"},
			{
				"type": "image_url",
				"image_url": {"url": f"data:image/png;base64,{image_base64}"}
			}
		]
	}
]

print("\n=== Generating response ===")
response = client.generate_content(messages, temperature=0.0)
print(response.text)
print("\n=== Test completed successfully ===")
PY

