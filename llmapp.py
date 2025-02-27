from flask import Flask, render_template, request, jsonify
import glob
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
import torch
from accelerate import disk_offload
import os

app = Flask(__name__)

# Folder containing scheme text files
SCHEME_FOLDER = "Schemes/"

# Load lightweight embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")  # Secure the token using environment variables

# Model name
model_name = "tiiuae/falcon-7b-instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu",  # Force CPU
    token=HF_TOKEN
)

# Offload the model to disk without unsupported argument
disk_offload(model, offload_dir="offload_folder")

# Create text generation pipeline
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def load_schemes(folder):
    schemes = []
    for state_folder in glob.glob(os.path.join(folder, "*")):
        if os.path.isdir(state_folder):
            for file_path in glob.glob(os.path.join(state_folder, "*.txt")):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    parts = content.split("\n\n")
                    scheme = {"state": os.path.basename(state_folder)}
                    for part in parts:
                        if part.startswith("Scheme Name:"):
                            scheme["name"] = part.replace("Scheme Name:", "").strip()
                        elif part.startswith("Description:"):
                            scheme["description"] = part.replace("Description:", "").strip()
                        elif part.startswith("Eligibility Criteria:"):
                            scheme["eligibility"] = part.replace("Eligibility Criteria:", "").strip()
                    if "name" in scheme and "description" in scheme:
                        schemes.append(scheme)
    return schemes

def improve_query(user_query):
    prompt = f"Refine the following user query for better understanding: {user_query}"
    response = llm_pipeline(prompt, max_new_tokens=50)[0]["generated_text"].strip()
    return response

def find_best_schemes(user_query, schemes):
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
    scheme_embeddings = embedding_model.encode([s["description"] for s in schemes], convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, scheme_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:5]
    return [schemes[i] for i in top_indices.cpu().numpy()]

def generate_detailed_info(top_schemes, user_query):
    selected_schemes = top_schemes[:3]
    schemes_text = "\n\n".join(
        [f"Scheme: {s['name']} (State: {s['state']})\nDescription: {s['description']}\nEligibility: {s.get('eligibility', 'Not specified')}" 
         for s in selected_schemes]
    )
    prompt = f"User query: {user_query}\n\nTop 3 schemes:\n{schemes_text}\n\nProvide detailed information on which scheme is best and why."
    response = llm_pipeline(prompt, max_new_tokens=200)[0]["generated_text"].strip()
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            form_data = {
                "age": request.form.get("age"),
                "gender": request.form.get("gender"),
                "marital": request.form.get("marital"),
                "state": request.form.get("state"),
                "urban": request.form.get("urban"),
                "income": request.form.get("income"),
                "religion": request.form.get("religion"),
                "otherReligion": request.form.get("otherReligion"),
                "caste": request.form.get("caste"),
                "education": request.form.get("education"),
                "citizenship": request.form.get("citizenship"),
                "employment_status": request.form.get("employment-status"),
                "occupation": request.form.get("occupation"),
                "otherOccupation": request.form.get("otherOccupation"),
                "sector": request.form.get("sector"),
                "otherSector": request.form.get("otherSector"),
                "disability": request.form.get("disability")
            }

            schemes = load_schemes(SCHEME_FOLDER)
            refined_query = improve_query(form_data)
            top_schemes = find_best_schemes(refined_query, schemes)
            final_output = generate_detailed_info(top_schemes, refined_query)
            return render_template("result.html", schemes=top_schemes, recommendation=final_output)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)