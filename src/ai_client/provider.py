from __future__ import annotations

import os
from .chat import OpenRouterClient


def get_text_client():
	"""
	OpenRouter client for text-first tasks (anime.js planning/coding).
	Defaults to openrouter/polaris-alpha, can override via OPENROUTER_MODEL_TEXT.
	"""
	model = os.getenv("OPENROUTER_MODEL_TEXT", "openrouter/polaris-alpha")
	return OpenRouterClient(model_name=model)


def get_vision_client():
	"""
	OpenRouter client for image-first tasks (image understanding/editing).
	Defaults to google/gemini-2.5-flash-image, can override via OPENROUTER_MODEL_VISION.
	"""
	model = os.getenv("OPENROUTER_MODEL_VISION", "google/gemini-2.5-flash-image")
	return OpenRouterClient(model_name=model)


def get_llm_client(model_name: str | None = None):
	# Backwards-compatible single entrypoint
	return OpenRouterClient(model_name=model_name)


