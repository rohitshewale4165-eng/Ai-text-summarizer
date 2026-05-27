from flask import Flask, render_template, request, jsonify
import requests
import os
import time
from pathlib import Path

BASE = Path(__file__).parent
env_file = BASE / ".env"
env_example = BASE / ".env.example"

if not env_file.exists() and env_example.exists():
    env_file.write_text(env_example.read_text())

from dotenv import load_dotenv
load_dotenv(env_file, override=True)

app = Flask(__name__)

# HuggingFace Inference Router (newer, more reliable endpoint)
# Falls back to the classic endpoint if router also fails
ENDPOINTS = [
    "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn",
    "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
]

def get_token():
    load_dotenv(env_file, override=True)
    raw = os.getenv("HF_TOKEN", "").strip()
    if raw.startswith("HF_TOKEN="):
        raw = raw[len("HF_TOKEN="):]
    return raw.strip()

def summarize_text(text, token):
    words = text.split()
    if len(words) > 500:
        text = " ".join(words[:500])

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": text,
        "parameters": {"max_length": 150, "min_length": 30, "do_sample": False}
    }

    last_error = None
    for url in ENDPOINTS:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection failed for {url}: {e}"
            continue
        except requests.exceptions.Timeout:
            last_error = "Request timed out. Please try again."
            continue

        if response.status_code == 401:
            return None, "Invalid HF token. Check your .env file."
        if response.status_code == 503:
            return None, "Model is loading on HuggingFace servers. Wait 20 seconds and retry."
        if response.status_code != 200:
            last_error = f"API Error {response.status_code} from {url}: {response.text}"
            continue

        result = response.json()
        if isinstance(result, list) and result:
            return result[0].get("summary_text", ""), None
        last_error = "Unexpected response format from API."

    return None, last_error or "All endpoints failed. Check your internet connection or try a VPN."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    token = get_token()

    if not token or token == "hf_your_token_here":
        return jsonify({"error": (
            "HF_TOKEN not set. Open .env and replace "
            "'hf_your_token_here' with your token from "
            "https://huggingface.co/settings/tokens"
        )}), 400

    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text or len(text.split()) < 10:
        return jsonify({"error": "Please enter at least 10 words."}), 400

    start = time.time()
    summary, error = summarize_text(text, token)
    elapsed = round(time.time() - start, 2)

    if error:
        return jsonify({"error": error}), 500

    return jsonify({
        "summary": summary,
        "original_words": len(text.split()),
        "summary_words": len(summary.split()),
        "time": elapsed
    })

if __name__ == "__main__":
    token = get_token()
    print("\n" + "="*50)
    if not token or token == "hf_your_token_here":
        print("  WARNING: HF_TOKEN not set!")
        print(f"  Open: {env_file}")
        print("  Replace 'hf_your_token_here' with your token")
        print("  Get one FREE at: https://huggingface.co/settings/tokens")
    else:
        print(f"  Token loaded: {token[:8]}{'*' * (len(token)-8)}")
        print("  Endpoints (tried in order):")
        for ep in ENDPOINTS:
            print(f"    - {ep}")
        print("  Ready!")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
