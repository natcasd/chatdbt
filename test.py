#!/usr/bin/env python3
"""
Test script to demonstrate interacting with different LLM clients.
This script tests OpenAI, Anthropic, and Groq LLM clients.
"""

import sys
import os
from pathlib import Path
import time

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import LLM clients
from llms.llm_interaction import OpenAIClient, AnthropicClient, GrokClient

def test_llm(client, name):
    """Test an LLM client with a simple prompt."""
    print(f"\n{'='*50}\nTesting {name} client\n{'='*50}")
    try:
        prompt = "Explain quantum computing in exactly two sentences."
        print(f"Prompt: {prompt}")
        print("\nGenerating response...")
        start_time = time.time()
        response = client.generate(prompt, max_tokens=100)
        elapsed_time = time.time() - start_time
        print(f"\nResponse ({elapsed_time:.2f}s):\n{response}")
        return True
    except Exception as e:
        print(f"Error with {name} client: {str(e)}")
        return False

def main():
    """Test all available LLM clients."""
    print("LLM Client Test Script")
    print("======================\n")
    
    # Record which clients worked
    results = {}
    
    # Test OpenAI client
    try:
        openai_client = OpenAIClient(model="gpt-3.5-turbo")
        results["OpenAI"] = test_llm(openai_client, "OpenAI")
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        results["OpenAI"] = False
    
    # Test Anthropic client
    try:
        anthropic_client = AnthropicClient(model="claude-3-haiku-20240307")
        results["Anthropic"] = test_llm(anthropic_client, "Anthropic")
    except Exception as e:
        print(f"Error initializing Anthropic client: {str(e)}")
        results["Anthropic"] = False
    
    # Test Groq client
    try:
        groq_client = GrokClient(model="llama3-8b-8192")
        results["Groq"] = test_llm(groq_client, "Groq")
    except Exception as e:
        print(f"Error initializing Groq client: {str(e)}")
        results["Groq"] = False
    
    # Print summary
    print("\n\nTest Summary")
    print("============")
    for client_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{client_name}: {status}")

if __name__ == "__main__":
    main()
