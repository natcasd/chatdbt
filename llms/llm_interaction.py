import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenAIClient:
    """
    Class for interacting with OpenAI's language models.
    """
    def __init__(
        self, 
        model: str = "gpt-3.5-turbo", 
        temperature: float = 0.7
    ):
        """
        Initialize an OpenAI LLM client.
        
        Args:
            model: The specific OpenAI model to use (default: gpt-3.5-turbo)
            temperature: Temperature for generation (higher = more creative)
        """
        self.model = model
        self.temperature = temperature
        self.provider = "openai"
        
        # Get API key from environment
        #API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"OpenAI client initialized with model: {model}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run 'pip install openai' to use OpenAI models.")
    
    def list_models(self):
        """
        Fetch and display all available models from OpenAI API.
        """
        try:
            models = self.client.models.list()
            openai_models = [model.id for model in models.data]
            
            # Group models by category for easier readability
            gpt4_models = [model for model in openai_models if model.startswith("gpt-4")]
            gpt35_models = [model for model in openai_models if model.startswith("gpt-3.5")]
            other_models = [model for model in openai_models if not (model.startswith("gpt-4") or model.startswith("gpt-3.5"))]
            
            print("\n" + "="*50)
            print(f"AVAILABLE OPENAI MODELS ({len(openai_models)} total)")
            print("="*50)
            
            if gpt4_models:
                print("\nGPT-4 Models:")
                for model in sorted(gpt4_models):
                    print(f"  - {model}")
            
            if gpt35_models:
                print("\nGPT-3.5 Models:")
                for model in sorted(gpt35_models):
                    print(f"  - {model}")
                    
            if other_models:
                print("\nOther Models:")
                for model in sorted(other_models):
                    print(f"  - {model}")
            
            print("\n")
        except Exception as e:
            print(f"Error listing OpenAI models: {str(e)}")
            print("\n")
    
    def generate(self, prompt: str, max_tokens: int = 500, system_prompt: str = None, **kwargs) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the model's behavior
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            # For chat models
            if self.model.startswith(("gpt-3.5", "gpt-4")):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content.strip()
            
            # For completion models (older models)
            else:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].text.strip()
                
        except Exception as e:
            print(f"Error generating text with OpenAI: {str(e)}")
            return f"Error: {str(e)}"


class AnthropicClient:
    """
    Class for interacting with Anthropic's Claude language models.
    """
    def __init__(
        self, 
        model: str = "claude-2", 
        temperature: float = 0.7
    ):
        """
        Initialize a Claude LLM client.
        
        Args:
            model: The specific Claude model to use (default: claude-2)
            temperature: Temperature for generation (higher = more creative)
        """
        self.model = model
        self.temperature = temperature
        self.provider = "anthropic"
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        # Initialize Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            print(f"Claude client initialized with model: {model}")
        except ImportError:
            raise ImportError("Anthropic package not installed. Run 'pip install anthropic' to use Claude models.")
    
    def list_models(self):
        """
        Fetch and display all available models from Anthropic API.
        """
        try:
            # Use the actual Anthropic API to list models
            models_response = self.client.models.list()
            models = models_response.data
            model_ids = [model.id for model in models]
            
            print("\n" + "="*50)
            print(f"AVAILABLE ANTHROPIC MODELS ({len(model_ids)} total)")
            print("="*50)
            
            print("\nClaude Models:")
            for model in models:
                current = " (currently selected)" if model.id == self.model else ""
                print(f"  - {model.id}{current}")
                if hasattr(model, 'display_name') and model.display_name:
                    print(f"    Display Name: {model.display_name}")
            
            print("\n")
        except Exception as e:
            print(f"Error listing Anthropic models: {str(e)}")
            print("\n")
    
    def generate(self, prompt: str, max_tokens: int = 500, system_prompt: str = None, **kwargs) -> str:
        """
        Generate text using Claude API.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the model's behavior
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            # Using the messages API (anthropic >= 0.5.0)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.content[0].text
                
        except Exception as e:
            print(f"Error generating text with Claude: {str(e)}")
            return f"Error: {str(e)}"


class GroqClient:
    """
    Class for interacting with Groq's language model APIs.
    """
    def __init__(
        self, 
        model: str = "llama3-8b-8192", 
        temperature: float = 0.7
    ):
        """
        Initialize a Groq client for fast hosted LLM inference. Can query a variety of open source models for free.
        
        Args:
            model: The specific model to use (default: llama3-8b-8192)
            temperature: Temperature for generation (higher = more creative)
        """
        self.model = model
        self.temperature = temperature
        self.provider = "groq"
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq client
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            print(f"Groq client initialized with model: {model}")
        except ImportError:
            raise ImportError("Groq package not installed. Run 'pip install groq' to use Groq-hosted models.")
    
    def list_models(self):
        """
        Fetch and display all available models from Groq API.
        """
        try:
            models = self.client.models.list()
            model_ids = [model.id for model in models.data]
            
            print("\n" + "="*50)
            print(f"AVAILABLE GROQ MODELS ({len(model_ids)} total)")
            print("="*50)
            
            print("\nGroq-hosted Models:")
            for model in models.data:
                current = " (currently selected)" if model.id == self.model else ""
                print(f"  - {model.id}{current}")
                if hasattr(model, 'description') and model.description:
                    print(f"    Description: {model.description}")
            
            print("\n")
        except Exception as e:
            print(f"Error listing Groq models: {str(e)}")
            print("\n")
    
    def generate(self, prompt: str, max_tokens: int = 500, system_prompt: str = None, **kwargs) -> str:
        """
        Generate text using Groq API.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the model's behavior
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error generating text with Groq: {str(e)}")
            return f"Error: {str(e)}"

