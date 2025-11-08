from __future__ import annotations

import os
import tempfile
from typing import Tuple

from PIL import Image as PILImage
from transformers import AutoModel, AutoTokenizer
import torch

# Singleton caches
_MODEL = None
_TOKENIZER = None
_DEVICE = None
_DTYPE = None


def _load_model_and_tokenizer() -> Tuple[AutoModel, AutoTokenizer, str, torch.dtype]:
	"""
	Lazy-load the DeepSeek-OCR model and tokenizer once.
	- Uses eager attention (no flash-attn dependency)
	- Picks device/dtype automatically unless overridden by environment
	"""
	global _MODEL, _TOKENIZER, _DEVICE, _DTYPE
	if _MODEL is not None and _TOKENIZER is not None:
		return _MODEL, _TOKENIZER, _DEVICE, _DTYPE

	# Hugging Face cache directory
	os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

	model_name = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")

	# Load tokenizer
	_TOKENIZER = AutoTokenizer.from_pretrained(
		model_name,
		trust_remote_code=True,
	)

	# Device and dtype selection
	_DEVICE = os.getenv("DEEPSEEK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
	if _DEVICE == "cuda":
		if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
			_DTYPE = torch.bfloat16
		else:
			_DTYPE = torch.float16
	else:
		# Force-disable CUDA visibility for downstream libraries expecting GPUs
		os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
		_DTYPE = torch.float32

	# Load model (eager attention to avoid extra deps)
	_MODEL = AutoModel.from_pretrained(
		model_name,
		trust_remote_code=True,
		use_safetensors=True,
		attn_implementation="eager",
		torch_dtype=_DTYPE,
	).eval().to(_DEVICE)

	# On CPU, explicitly convert float parameters/buffers to float32 to avoid dtype mismatches
	# (some model components may have been loaded with bfloat16 from checkpoint)
	# Preserve integer buffers (like position_ids) which must remain Long/Int for embedding operations
	if _DEVICE == "cpu":
		for param in _MODEL.parameters():
			if param.dtype in (torch.bfloat16, torch.float16):
				param.data = param.data.to(torch.float32)
		for buffer in _MODEL.buffers():
			# Skip integer buffers (position_ids, etc.) - they must remain Long/Int for embeddings
			if buffer.dtype in (torch.int64, torch.int32, torch.long, torch.int):
				continue
			# Only convert float buffers to float32
			if buffer.dtype in (torch.bfloat16, torch.float16):
				buffer.data = buffer.data.to(torch.float32)

	# Ensure pad token setup (best-effort)
	try:
		if getattr(_TOKENIZER, "pad_token_id", None) is None and getattr(_TOKENIZER, "eos_token_id", None) is not None:
			_TOKENIZER.pad_token = _TOKENIZER.eos_token
		if getattr(_MODEL.config, "pad_token_id", None) is None and getattr(_TOKENIZER, "pad_token_id", None) is not None:
			_MODEL.config.pad_token_id = _TOKENIZER.pad_token_id
	except Exception:
		pass

	return _MODEL, _TOKENIZER, _DEVICE, _DTYPE


def infer_text(
	pil_img: PILImage.Image,
	prompt: str = "<image>\nFree OCR.",
	base_size: int | None = None,
	image_size: int | None = None,
	crop_mode: bool | None = None,
	test_compress: bool = False,
) -> str:
	"""
	Run DeepSeek-OCR on a PIL image and return extracted text (string).
	Saves the image to a temporary file and calls model.infer(...) from the HF model.
	"""
	model, tokenizer, device, dtype = _load_model_and_tokenizer()

	# Parameters (env-overridable)
	base_size = base_size or int(os.getenv("DEEPSEEK_BASE_SIZE", "1024"))
	image_size = image_size or int(os.getenv("DEEPSEEK_IMAGE_SIZE", "640"))
	crop_mode = crop_mode if crop_mode is not None else (os.getenv("DEEPSEEK_CROP_MODE", "1") == "1")
	# output_dir must be a valid directory (model code unconditionally creates it)
	output_dir_env = os.getenv("DEEPSEEK_OUTPUT_DIR")
	if output_dir_env and output_dir_env.strip():
		output_dir = output_dir_env
	else:
		# Default to a stable temp subdir
		output_dir = os.path.join(tempfile.gettempdir(), "deepseek_ocr_outputs")
	os.makedirs(output_dir, exist_ok=True)

	# On CPU-only machines, patch autocast and .cuda() calls to prevent dtype mismatches
	if device == "cpu":
		# Save original autocast
		_original_autocast = torch.cuda.amp.autocast
		_original_tensor_cuda = torch.Tensor.cuda
		_original_module_cuda = torch.nn.Module.cuda
		
		def _patched_autocast(*args, **kwargs):
			# Force disable autocast on CPU - always return a no-op context
			class NoOpAutocast:
				def __enter__(self):
					return self
				def __exit__(self, *args):
					pass
			return NoOpAutocast()
		
		def _noop_cuda_tensor(self, *args, **kwargs):
			# For float tensors: convert bfloat16/float16 to float32 on CPU
			# For integer tensors: keep original dtype (don't convert!)
			if self.dtype in (torch.bfloat16, torch.float16):
				return self.to(torch.float32)
			# For other dtypes (int64, int32, etc.), just return as-is
			return self
		
		def _noop_cuda_module(self, *args, **kwargs):
			return self.to("cpu")
		
		# Apply patches
		torch.cuda.amp.autocast = _patched_autocast  # type: ignore[assignment]
		torch.Tensor.cuda = _noop_cuda_tensor  # type: ignore[attr-defined]
		torch.nn.Module.cuda = _noop_cuda_module  # type: ignore[attr-defined]

	tmp_path = None
	try:
		with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
			pil_img.save(tmp.name)
			tmp_path = tmp.name

		res = model.infer(
			tokenizer,
			prompt=prompt,
			image_file=tmp_path,
			output_path=output_dir,
			base_size=base_size,
			image_size=image_size,
			crop_mode=crop_mode,
			save_results=False,
			test_compress=test_compress,
		)
		
		# Restore original functions if we patched them
		if device == "cpu":
			try:
				torch.cuda.amp.autocast = _original_autocast  # type: ignore[assignment]
				torch.Tensor.cuda = _original_tensor_cuda  # type: ignore[attr-defined]
				torch.nn.Module.cuda = _original_module_cuda  # type: ignore[attr-defined]
			except Exception:
				pass

		if isinstance(res, str):
			return res.strip()
		if isinstance(res, dict) and "text" in res:
			return str(res["text"]).strip()
		if isinstance(res, (list, tuple)):
			return "\n".join(map(str, res)).strip()
		return ""
	finally:
		if tmp_path:
			try:
				os.remove(tmp_path)
			except Exception:
				pass


