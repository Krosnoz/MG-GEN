import io
import os
import base64

from openai import OpenAI
from PIL import Image as PILImage


class GenerateContentConfig:
    """Compatibility wrapper for GenerateContentConfig to maintain backward compatibility."""
    def __init__(self, temperature=None, response_mime_type=None, **kwargs):
        self.temperature = temperature
        self.response_mime_type = response_mime_type
        self._kwargs = kwargs


class Part:
    """Compatibility wrapper for Part class to maintain backward compatibility."""
    def __init__(self, image_data=None, mime_type='image/png'):
        self.image_data = image_data
        self.mime_type = mime_type
    
    @classmethod
    def from_bytes(cls, data, mime_type='image/png'):
        return cls(image_data=data, mime_type=mime_type)


class ClientWrapper:
    """Wrapper to mimic google.genai.Client interface with models attribute."""
    def __init__(self, openai_client):
        self._openai_client = openai_client
        self.models = ModelsWrapper(openai_client)


class ModelsWrapper:
    """Wrapper to mimic google.genai.Client.models interface."""
    def __init__(self, openai_client):
        self._client = openai_client
    
    def generate_content(self, model, contents, config=None):
        """Generate content using OpenRouter API via OpenAI client."""
        # Build messages array from contents
        # Contents can be a list of strings and Part objects
        message_content = []
        
        for content in contents:
            if isinstance(content, str):
                # Text content
                message_content.append({"type": "text", "text": content})
            elif isinstance(content, Part):
                # Image content
                # Convert image data to base64 if needed
                if isinstance(content.image_data, bytes):
                    image_base64 = base64.b64encode(content.image_data).decode('utf-8')
                else:
                    image_base64 = content.image_data
                
                image_url = f"data:{content.mime_type};base64,{image_base64}"
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
        
        # Create user message
        # If only one text item, use simple string format, otherwise use array format
        if len(message_content) == 1 and message_content[0].get("type") == "text":
            messages = [{"role": "user", "content": message_content[0]["text"]}]
        else:
            messages = [{"role": "user", "content": message_content}]
        
        # Prepare API parameters
        api_params = {
            "model": model,
            "messages": messages,
        }
        
        # Map config parameters
        if config:
            if config.temperature is not None:
                api_params["temperature"] = config.temperature
            if config.response_mime_type == "application/json":
                api_params["response_format"] = {"type": "json_object"}
        
        # Make API call
        response = self._client.chat.completions.create(**api_params)
        
        # Return wrapper with .text attribute for compatibility
        class ResponseWrapper:
            def __init__(self, response):
                self._response = response
                self.text = response.choices[0].message.content if response.choices else ""
        
        return ResponseWrapper(response)


class Gemini:
    def __init__(self, model_name:str=None):
        # Get environment variables with defaults
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        default_model = os.getenv("OPENROUTER_MODEL", "openrouter/polaris-alpha")
        
        self.model_name = model_name if model_name is not None else default_model
        
        # Initialize OpenAI client configured for OpenRouter
        openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Wrap client to mimic google.genai.Client interface with models attribute
        self.client = ClientWrapper(openai_client)

    def upload_image(self, pil_image:PILImage.Image):
        """Convert PIL image to Part object compatible with OpenAI vision API."""
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        image_data = image_bytes.read()
        
        # Encode to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return Part(image_data=image_base64, mime_type='image/png')