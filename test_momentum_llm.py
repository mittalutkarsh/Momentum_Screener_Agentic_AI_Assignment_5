from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
import re

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")

genai.configure(api_key=api_key)

# Use the exact model we found in the list
model = genai.GenerativeModel("models/gemini-2.0-flash")

# Test with a simple finance query
prompt = """
You are a Momentum Screener agent with deep expertise in technical analysis.

Return a JSON object with these keys:
  "reasoning": {
    "parameter_check": "Explanation of parameter validity",
    "strategy_summary": "What this momentum strategy aims to capture"
  }
}

IMPORTANT: Your response must be valid JSON with no narrative text before or after.
"""

try:
    response = model.generate_content(prompt)
    print(f"Response received:\n{response.text}\n")
    
    # Handle markdown code blocks in the response
    json_text = response.text
    # Remove markdown code block if present
    json_text = re.sub(r'^```json\s*', '', json_text)
    json_text = re.sub(r'\s*```$', '', json_text)
    
    print(f"Cleaned text for parsing:\n{json_text}\n")
    
    # Try to parse as JSON
    try:
        data = json.loads(json_text)
        print("Successfully parsed as JSON!")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError as e:
        print(f"Failed to parse as JSON: {e}")
except Exception as e:
    print(f"API call failed: {e}")