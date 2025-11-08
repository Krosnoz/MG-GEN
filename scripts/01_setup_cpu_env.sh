#!/usr/bin/env bash
set -euo pipefail

# Minimal CPU-only setup for quick DeepSeek-OCR tests
# - Creates .venv
# - Installs CPU PyTorch and required libs for DeepSeek local inference

activate_venv() {
	if [ -f ".venv/bin/activate" ]; then
		source ".venv/bin/activate"
	elif [ -f ".venv/Scripts/activate" ]; then
		# Git Bash on Windows
		source ".venv/Scripts/activate"
	else
		echo "Cannot find venv activation script. Did venv creation fail?" >&2
		exit 1
	fi
}

if ! command -v python >/dev/null 2>&1; then
	echo "Python not found. Please install Python 3.10+." >&2
	exit 1
fi

if [ ! -d ".venv" ]; then
	python -m venv .venv
fi

activate_venv
python -m pip install --upgrade pip

# Install CPU PyTorch wheels and essentials for DeepSeek local
# Using CPU wheel index for PyTorch
python - <<'PY' || true
import sys, subprocess
def run(cmd):
	print("+", " ".join(cmd))
	subprocess.check_call(cmd)
try:
	run([sys.executable, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.4.1", "torchvision==0.19.1"])
except Exception as e:
	print("CPU PyTorch install via index-url failed; retrying with default index and extra-index-url...")
	run([sys.executable, "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cpu", "torch==2.4.1", "torchvision==0.19.1"])
PY

pip install \
	transformers==4.46.3 \
	tokenizers==0.20.3 \
	einops \
	safetensors \
	addict \
	easydict \
	pillow \
	dotenv \
	openai

echo "Environment ready."
echo "Tip: export DEEPSEEK_DEVICE=cpu; DEEPSEEK_BASE_SIZE=512; DEEPSEEK_IMAGE_SIZE=512; DEEPSEEK_CROP_MODE=0 for faster tests."


