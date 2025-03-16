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
        
        # Get API key from environment
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
    
    def generate(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            # For chat models
            if self.model.startswith(("gpt-3.5", "gpt-4")):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
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
    
    def generate(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """
        Generate text using Claude API.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            # Using the messages API (anthropic >= 0.5.0)
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.content[0].text
                
        except Exception as e:
            print(f"Error generating text with Claude: {str(e)}")
            return f"Error: {str(e)}"

