from __future__ import annotations

from typing import List, Dict

from PIL import Image as PILImage

from .deepseek_local import infer_text


class DeepSeekOCRRefiner:
	def __init__(self):
		# Always enabled: local transformers model usage only
		self.enabled = True

	def refine(self, ocr_text_list: List[Dict], image: PILImage.Image) -> List[Dict]:

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
				prompt = "<image>\nFree OCR."
				deepseek_text = infer_text(crop, prompt=prompt) or item.get("text", "")

				# Update only text; keep geometry
				new_item = dict(item)
				new_item["text"] = deepseek_text
				refined.append(new_item)
			except Exception:
				refined.append(item)
		return refined


