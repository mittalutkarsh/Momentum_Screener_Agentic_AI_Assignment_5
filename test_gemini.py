from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")

genai.configure(api_key=api_key)

# Try available models
available_models = [model.name for model in genai.list_models()]
print("Available models:", available_models)

# Try a simple test with an available model
try:
    # Most likely the model is gemini-pro or gemini-1.5-pro
    model = genai.GenerativeModel("gemini-pro")  # Try this model first
    response = model.generate_content("Analyze this stock: AAPL")
    print(f"Test succeeded! Response: {response.text[:100]}...")
except Exception as e:
    print(f"Test failed: {e}")