import os
from typing import Any
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from src.domain.llm.llm_port import LLMProvider
from src.utils.logger import log
from pydantic import BaseModel
from huggingface_hub import InferenceClient

load_dotenv()

class HFLLMAdapter(LLMProvider):
    """HuggingFace LLM Adapter using LangChain's ChatHuggingFace."""
    
    def __init__(self, repo_id: str = "openai/gpt-oss-20b", temperature: float = 0.7):
        """
        Initialize HuggingFace LLM Adapter.
        
        Args:
            repo_id: HuggingFace model repository ID
            temperature: Temperature for generation (0.0 to 1.0)
            max_length: Maximum length of generated text
        """
        self.api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment")
        
        # Set environment variable for LangChain
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.api_token
        
        self.repo_id = repo_id
        self.temperature = temperature
        self.client = InferenceClient(model=self.repo_id, token=self.api_token, provider="auto")
        # # Initialize HuggingFace Endpoint
        # self.llm = HuggingFaceEndpoint(
        #     repo_id=self.repo_id,
        #     temperature=self.temperature,
        # )
        
        # Wrap with ChatHuggingFace for chat model capabilities
        # self.model = ChatHuggingFace(llm=self.llm,verbose=True)
        
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text response
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(
                messages=messages, 
                temperature=self.temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log("llm_adapter", f"Error generating content: {e}")
            raise

    def generate_structured(self, prompt: str, schema: BaseModel) -> BaseModel:
        """
        Generate structured data from a prompt using Pydantic schema.
        
        Args:
            prompt: Input prompt string
            schema: Pydantic BaseModel schema for structured output
            
        Returns:
            Validated Pydantic model instance
        """
        try:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{schema.__name__}Schema",
                    "schema": schema.model_json_schema(),
                    "strict": False,
                },
            }
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(messages=messages, response_format=response_format)
            result = response.choices[0].message.content
            schema_instance = schema.model_validate_json(result)
            return schema_instance
        except Exception as e:
            log("llm_adapter", f"Failed to parse structured output: {e}")
            raise
