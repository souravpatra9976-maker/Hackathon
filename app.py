from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM


MODEL_DIR = r"C:\Users\3023149\Desktop\models"
MODEL_FILE = "TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf"


model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    model_file=MODEL_FILE,
    model_type="llama",  
    gpu_layers=0         
)

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    
    response = model(prompt, max_new_tokens=100)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
