import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("No API Key")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)
if response.status_code == 200:
    models = response.json().get('models', [])
    print("Available Embedding Models:")
    for model in models:
        if 'embedContent' in model.get('supportedGenerationMethods', []):
            print(f"- {model['name']}")
else:
    print(f"Error fetching models: {response.text}")
