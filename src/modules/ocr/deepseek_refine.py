from __future__ import annotations

import io
import os
import base64
import json
import requests
from typing import List, Dict

from PIL import Image as PILImage


class DeepSeekOCRRefiner:
	def __init__(self):
		# Remote inference endpoint (e.g., your RunPod-served DeepSeek-OCR API)
		# Expected to accept POST /infer with JSON: {"prompt": "...", "image_b64": "..."}
		# and return {"text": "..."} or {"outputs":[{"text":"..."}]}
		self.endpoint = os.getenv("DEEPSEEK_OCR_URL")  # e.g., https://<your-runpod-or-api>/infer
		self.enabled = bool(self.endpoint)

	def _encode_image_b64(self, pil_img: PILImage.Image) -> str:
		buf = io.BytesIO()
		pil_img.save(buf, format="PNG")
		return base64.b64encode(buf.getvalue()).decode("utf-8")

	def _post_image(self, pil_img: PILImage.Image, prompt: str) -> str | None:
		if not self.endpoint:
			return None
		try:
			payload = {"prompt": prompt, "image_b64": self._encode_image_b64(pil_img)}
			resp = requests.post(self.endpoint, json=payload, timeout=30)
			if resp.status_code != 200:
				return None
			data = resp.json()
			if isinstance(data, dict):
				if "text" in data and isinstance(data["text"], str):
					return data["text"].strip()
				if "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
					first = data["outputs"][0]
					if isinstance(first, dict) and "text" in first:
						return str(first["text"]).strip()
			# Fallback try text content
			return (resp.text or "").strip()
		except Exception:
			return None

	def refine(self, ocr_text_list: List[Dict], image: PILImage.Image) -> List[Dict]:
		# If not configured, return unchanged (we preserve PaddleOCR geometry)
		if not self.enabled:
			return ocr_text_list

		refined: List[Dict] = []
		for item in ocr_text_list:
			try:
				# Crop region by bbox derived from polygon
				vertices = item.get("vertices")
				if not vertices:
					refined.append(item)
					continue
				min_x = min(v["x"] for v in vertices)
				min_y = min(v["y"] for v in vertices)
				max_x = max(v["x"] for v in vertices)
				max_y = max(v["y"] for v in vertices)
				crop = image.crop((min_x, min_y, max_x, max_y))

				# Prompt: refined OCR for the cropped region
				prompt = "<image>\\nFree OCR."
				deepseek_text = self._post_image(crop, prompt) or item.get("text", "")

				# Update only text; keep geometry
				new_item = dict(item)
				new_item["text"] = deepseek_text
				refined.append(new_item)
			except Exception:
				refined.append(item)
		return refined


