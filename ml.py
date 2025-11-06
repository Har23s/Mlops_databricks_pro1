import os
import mlflow
import requests
import json
import time
import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Load All Necessary API Keys and Endpoints ---
# Google Gemini
os.environ['GOOGLE_API_KEY'] = 'api_key'
os.environ['GOOGLE_API_BASE'] = 'https://generativelanguage.googleapis.com/v1beta'

# Groq
os.environ['GROQ_API_KEY'] = 'api_key'
os.environ['GROQ_API_BASE'] = 'https://api.groq.com/openai/v1'



# --- Set the MLflow Experiment ---
experiment_name = "/Workspace/Users/acct_name"
mlflow.set_experiment(experiment_name=experiment_name)

print(f"✅ Environment variables loaded and MLflow experiment is set to: '{experiment_name}'")

# --- Helper function for fetching Google Gemini models ---
def _get_gemini_models():
    try:
        url = f"{os.environ['GOOGLE_API_BASE']}/models?key={os.environ['GOOGLE_API_KEY']}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Filter for models that support 'generateContent' and are not deprecated
        return [
            model['name'].replace('models/', '')
            for model in data.get('models', [])
            if 'generateContent' in model.get('supportedGenerationMethods', [])
        ]
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch models for Google. Error: {e}")
        return ['gemini-1.5-flash-latest'] # Fallback

# --- Helper function for fetching Groq models ---
def _get_groq_models():
    try:
        url = f"{os.environ['GROQ_API_BASE']}/models"
        headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [model['id'] for model in data.get('data', [])]
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch models for Groq. Error: {e}")
        return ['llama3-70b-8192', 'llama3-8b-8192'] # Fallback

# --- Helper function for fetching Ollama-based models ---
def _get_ollama_models(base_url: str):
    try:
        url = f"{base_url}/api/tags"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [model['name'] for model in data.get('models', [])]
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch models from {base_url}. Error: {e}")
        return [] # Fallback

# --- Helper function for Gemini ---
def _get_gemini_response(model: str, prompt: str) -> str:
    url = f"{os.environ['GOOGLE_API_BASE']}/models/{model}:generateContent?key={os.environ['GOOGLE_API_KEY']}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

# --- Helper function for Groq (OpenAI-compatible API) ---
def _get_groq_response(model: str, prompt: str) -> str:
    url = f"{os.environ['GROQ_API_BASE']}/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# --- Helper function for Ollama ---
def _get_ollama_response(base_url: str, model: str, prompt: str) -> str:
    url = f"{base_url}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json().get("response", "Error: No 'response' key in API output.")

# --- Main Dispatch Function ---
def get_ai_response(provider: str, model: str, prompt: str) -> str:
    """
    Calls the correct AI provider based on the 'provider' key.
    """
    try:
        print(f"Sending prompt to {provider} using model {model}...")
        if provider == 'google':
            return _get_gemini_response(model=model, prompt=prompt)
        elif provider == 'groq':
            return _get_groq_response(model=model, prompt=prompt)
        elif provider == 'ollama':
            return _get_ollama_response(base_url=os.environ['OLLAMA_API_BASE'], model=model, prompt=prompt)
        else:
            return f"Error: Unknown provider '{provider}'"

    except requests.exceptions.RequestException as e:
        return f"Error: A network error occurred for {provider}: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred with {provider}: {e}"

print("✅ Unified AI provider functions are defined.")

# --- Dynamically fetch models for all providers ---
print("Fetching available models from all providers...")
model_options = {
    'google': _get_gemini_models(),
    'groq': _get_groq_models(),
    'ollama': _get_ollama_models(os.environ['OLLAMA_API_BASE']),
    'zinghr_ollama': _get_ollama_models(os.environ['ZINGHR_OLLAMA_API_BASE'])
}
print("✅ Models fetched successfully.")

# Global variables for chat state
chat_active = False
turn_counter = 1
selected_provider = None
selected_model = None

print("\n" + "="*60)
print(" AI CHAT INTERFACE")  
print("="*60)

# --- Create Simple Widgets (display individually to avoid layout issues) ---
print("\n STEP 1: Select Provider and Model")
print("-" * 40)

# Provider dropdown
provider_dropdown = widgets.Dropdown(
    options=[provider for provider, models in model_options.items() if models],
    description='Provider:'
)
display(provider_dropdown)

# Model dropdown  
model_dropdown = widgets.Dropdown(description='Model:')
display(model_dropdown)

def update_models_dropdown(change):
    """Updates the model list when the provider changes."""
    provider = change['new']
    model_dropdown.options = model_options.get(provider, [])

provider_dropdown.observe(update_models_dropdown, names='value')

# Set the initial model options based on the first provider
if provider_dropdown.value:
    update_models_dropdown({'new': provider_dropdown.value})
