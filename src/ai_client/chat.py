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
        Perform image edit/inpainting using OpenRouter API.
        For Gemini models: Uses chat/completions API with multiple images.
        For OpenAI-compatible models: Uses images.edits API.
        The mask specifies regions to edit. Unmasked regions must remain unchanged.
        """
        # Check if this is a Gemini model
        is_gemini = "gemini" in self.model_name.lower()
        
        if is_gemini:
            # Gemini models use chat/completions API with multiple images
            return self._edit_image_gemini(image, mask, prompt)
        else:
            # OpenAI-compatible models use images.edits API
            return self._edit_image_openai(image, mask, prompt)
    
    def _edit_image_gemini(self, image: PILImage.Image, mask: PILImage.Image, prompt: str) -> PILImage.Image:
        """Edit image using Gemini chat/completions API with multiple images."""
        # Prepare images as base64
        image_base64 = self.upload_image(image)
        
        # Prepare mask as base64 (convert to RGB for better compatibility)
        if mask.mode != "RGB":
            mask_rgb = mask.convert("RGB")
        else:
            mask_rgb = mask
        mask_base64 = self.upload_image(mask_rgb)
        
        # Create messages with both images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\n\nUse the first image as the base image and the second image as a mask indicating which regions to edit. The white areas in the mask should be edited according to the prompt, while black areas should remain unchanged."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{mask_base64}"}
                    }
                ]
            }
        ]
        
        # Call chat/completions with image modality
        # OpenRouter requires modalities parameter, which may need to be passed via extra_body
        try:
            # Try with extra_body first (OpenAI SDK >= 1.0)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                extra_body={"modalities": ["image", "text"]},
            )
        except (TypeError, AttributeError):
            # Fallback: use requests directly for OpenRouter-specific parameters
            import requests
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            
            response_data = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "modalities": ["image", "text"],
                }
            )
            response_data.raise_for_status()
            response_json = response_data.json()
            
            # Convert response to match OpenAI SDK format
            class MockResponse:
                def __init__(self, data):
                    self.choices = [MockChoice(data.get("choices", [{}])[0])] if data.get("choices") else []
            
            class MockChoice:
                def __init__(self, choice_data):
                    self.message = MockMessage(choice_data.get("message", {}))
            
            class MockMessage:
                def __init__(self, msg_data):
                    self.content = msg_data.get("content", "")
                    self.images = msg_data.get("images", [])
            
            response = MockResponse(response_json)
        
        # Extract image from response
        message = response.choices[0].message if response.choices else None
        if not message:
            raise RuntimeError("Image edit returned no response")
        
        # Check for images in the response
        images = None
        if hasattr(message, "images"):
            images = message.images
        elif isinstance(message, dict) and "images" in message:
            images = message["images"]
        
        if images:
            # Get first image
            image_data = images[0] if isinstance(images, list) else images
            
            # Extract image URL from various formats
            image_url = None
            if isinstance(image_data, dict):
                if "image_url" in image_data:
                    img_url_obj = image_data["image_url"]
                    if isinstance(img_url_obj, dict) and "url" in img_url_obj:
                        image_url = img_url_obj["url"]
                    elif isinstance(img_url_obj, str):
                        image_url = img_url_obj
                elif "url" in image_data:
                    image_url = image_data["url"]
            elif hasattr(image_data, "image_url"):
                if hasattr(image_data.image_url, "url"):
                    image_url = image_data.image_url.url
                else:
                    image_url = str(image_data.image_url)
            
            if not image_url:
                raise RuntimeError(f"Unexpected image format in response: {type(image_data)}")
            
            # Extract base64 from data URL
            if image_url.startswith("data:image"):
                base64_data = image_url.split(",", 1)[1]
            elif image_url.startswith("http"):
                # If it's a URL, download it
                import requests
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                edited_bytes = img_response.content
                return PILImage.open(io.BytesIO(edited_bytes)).convert("RGB")
            else:
                # Assume it's already base64
                base64_data = image_url
            
            edited_bytes = base64.b64decode(base64_data)
            return PILImage.open(io.BytesIO(edited_bytes)).convert("RGB")
        else:
            # Debug: print response structure
            raise RuntimeError(f"Image edit returned no image data. Response structure: {type(message)}, has images attr: {hasattr(message, 'images')}, content: {getattr(message, 'content', 'N/A')[:200]}")
    
    def _edit_image_openai(self, image: PILImage.Image, mask: PILImage.Image, prompt: str) -> PILImage.Image:
        """Edit image using OpenAI-compatible images.edits API."""
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
