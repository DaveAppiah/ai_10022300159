# src/__init__.py
# Author: David Owusu Appiah | Index: 10022300159
"""
Load environment variables from .env file at package initialization.
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify Groq API key is available
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print(
        "[WARNING] GROQ_API_KEY not found in environment. "
        "Please set it in .env file or environment variables."
    )
