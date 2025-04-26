import os
from flask import Flask, request, jsonify, send_from_directory
from inference import predict

# Absolute paths
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    text     = request.form.get("text", "")
    file_obj = request.files.get("image")
    img_path = None

    if file_obj and file_obj.filename:
        img_path = os.path.join(PROJECT_DIR, "tmp.png")
        file_obj.save(img_path)

    try:
        probs = predict(img_path, text)
        best  = max(probs.items(), key=lambda kv: kv[1])[0]
        return jsonify({"prediction": f"You likely have {best}."})
    finally:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
