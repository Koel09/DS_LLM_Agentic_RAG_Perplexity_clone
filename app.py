from flask import Flask, request, jsonify, send_from_directory
from rag_pipeline import run_rag_pipeline
import requests
import os
import warnings
warnings.filterwarnings(
    "ignore",
    message=r'.*shadows an attribute in parent "BaseTool".*',
    category=UserWarning,
)

app = Flask(__name__, static_folder="static")

def download_index_html():
    index_url = "https://raw.githubusercontent.com/Koel09/DS_LLM_Agentic_RAG_Perplexity_clone/main/template/index.html"
    if not os.path.exists("template"):
        os.makedirs("template")
    response = requests.get(index_url)
    with open("template/index.html", "w", encoding="utf-8") as f:
        f.write(response.text)

def download_rag_pipeline():
    pass

# -------------------------
# API ENDPOINT
# -------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json

    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        result = run_rag_pipeline(
            question=question
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# FRONTEND
# -------------------------
@app.route("/")
def serve_ui():
    download_index_html()
    return send_from_directory("template", "index.html")


if __name__ == "__main__":
    app.run(debug=True)
