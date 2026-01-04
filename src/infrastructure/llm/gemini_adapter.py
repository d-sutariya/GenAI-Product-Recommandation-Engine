import os
from typing import Any
from google import genai
from dotenv import load_dotenv
from src.domain.llm.llm_port import LLMProvider
from src.utils.logger import log
from pydantic import BaseModel

load_dotenv()

class GeminiLLMAdapter(LLMProvider):
    def __init__(self, model_name: str = "gemma-3-27b-it"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            log("llm_adapter", f"Error generating content: {e}")
            raise

    def generate_structured(self, prompt: str, schema: BaseModel) -> BaseModel:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema.model_json_schema(),
            },
        )
        try:
            return schema.model_validate_json(response.text)
        except Exception as e:
            log("llm_adapter", f"Failed to parse output: {e}")
            raise
