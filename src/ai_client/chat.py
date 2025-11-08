import io
import os
import base64

from openai import OpenAI
from PIL import Image as PILImage


class OpenRouterClient:
    def __init__(self, model_name: str = None):
        # Get environment variables with defaults
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        default_model = os.getenv("OPENROUTER_MODEL", "openrouter/polaris-alpha")
        
        self.model_name = model_name if model_name is not None else default_model
        
        # Initialize OpenAI client configured for OpenRouter
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def upload_image(self, pil_image: PILImage.Image) -> str:
        """Convert PIL image to base64 string for OpenAI vision API."""
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        image_data = image_bytes.read()
        
        # Encode to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return image_base64

    def generate_content(self, messages, temperature=None, response_format=None):
        """Generate content using OpenRouter API via OpenAI client.
        
        Args:
            messages: List of message dicts in OpenAI format, e.g.:
                [{"role": "user", "content": [...]}]
            temperature: Optional temperature parameter
            response_format: Optional response format, e.g. {"type": "json_object"}
        
        Returns:
            Response object with .text attribute containing the generated content
        """
        api_params = {
            "model": self.model_name,
            "messages": messages,
        }
        
        if temperature is not None:
            api_params["temperature"] = temperature
        
        if response_format is not None:
            api_params["response_format"] = response_format
        
        # Make API call
        response = self.client.chat.completions.create(**api_params)
        
        # Return wrapper with .text attribute for convenience
        class ResponseWrapper:
            def __init__(self, response):
                self._response = response
                self.text = response.choices[0].message.content if response.choices else ""
        
        return ResponseWrapper(response)

    def edit_image_with_mask(self, image: PILImage.Image, mask: PILImage.Image, prompt: str) -> PILImage.Image:
        """
        Perform image edit/inpainting using OpenRouter Images API compatible with OpenAI.
        The mask specifies regions to edit. Unmasked regions must remain unchanged.
        """
        # Prepare image bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        img_bytes.name = "image.png"  # required by OpenAI images API

        # Prepare mask: convert L mask (0/255) to RGBA transparency mask where alpha=0 => edit region
        if mask.mode != "L":
            mask = mask.convert("L")
        # Create an RGBA mask where edited regions are transparent
        alpha = mask.point(lambda p: 255 - p)  # invert: 255->0 (edit), 0->255 (keep)
        rgba_mask = PILImage.new("RGBA", mask.size, (0, 0, 0, 0))
        rgba_mask.putalpha(alpha)

        mask_bytes = io.BytesIO()
        rgba_mask.save(mask_bytes, format="PNG")
        mask_bytes.seek(0)
        mask_bytes.name = "mask.png"

        result = self.client.images.edits.create(
            model=self.model_name,
            image=[img_bytes],
            mask=mask_bytes,
            prompt=prompt,
        )
        # Parse base64 output
        b64 = result.data[0].b64_json if getattr(result, "data", None) else None
        if not b64:
            raise RuntimeError("Image edit returned no data")
        edited_bytes = base64.b64decode(b64)
        return PILImage.open(io.BytesIO(edited_bytes)).convert("RGB")
