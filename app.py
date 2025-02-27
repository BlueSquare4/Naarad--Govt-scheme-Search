from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline

app = Flask(__name__)

# Load schemes from CSV. Only drop rows missing essential fields.
CSV_FILE = "NAARAD_Sheet1.csv"

def load_schemes(csv_file):
    df = pd.read_csv(csv_file)
    # Essential fields: State, Scheme Name, and Description must exist.
    df = df.dropna(subset=["State", "Scheme Name", "Description"])
    # For optional fields, fill missing values with an empty string.
    for field in ["Eligibility Criteria", "Age Criteria", "Gender Criteria", "Application Process", "Additional Notes"]:
        if field in df.columns:
            df[field] = df[field].fillna("")
    schemes = df.to_dict(orient="records")
    return schemes

# Load lightweight embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def find_best_schemes(user_query, schemes):
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
    
    composite_texts = []
    valid_schemes = []
    for s in schemes:
        texts = []
        # Always include essential fields: Scheme Name and Description
        texts.append(s["Scheme Name"])
        texts.append(s["Description"])
        # Include Eligibility Criteria only if it is meaningful.
        eligibility = s.get("Eligibility Criteria", "").strip().lower()
        if eligibility and eligibility not in ["na", "none"]:
            texts.append(s["Eligibility Criteria"])
        composite_text = " ".join(texts).strip()
        composite_texts.append(composite_text)
        valid_schemes.append(s)
    
    if not composite_texts:
        return []
    
    scheme_embeddings = embedding_model.encode(composite_texts, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, scheme_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:5]
    return [valid_schemes[i] for i in top_indices.cpu().numpy() if i < len(valid_schemes)]

# Load LLM pipeline for text generation (example using GPT-2)
llm_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=150)

def refine_response(filtered_schemes, user_query):
    scheme_text = "\n\n".join([
        f"Scheme Name: {s['Scheme Name']}\nDescription: {s['Description']}\nEligibility: {s['Eligibility Criteria']}"
        for s in filtered_schemes
    ])
    prompt = (f"User Query: {user_query}\n\nBased on the above government schemes, "
              "provide a detailed and genuine recommendation with insights and reasoning on which scheme "
              "would be best suited for the user.\n\n"
              f"{scheme_text}\n\nResponse:")
    refined = llm_pipeline(prompt)[0]['generated_text']
    return refined

@app.route("/form", methods=["GET", "POST"])
def form():
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
                "caste": request.form.get("caste"),
                "education": request.form.get("education"),
                "citizenship": request.form.get("citizenship"),
                "employment_status": request.form.get("employment-status"),
                "occupation": request.form.get("occupation"),
                "sector": request.form.get("sector"),
                "disability": request.form.get("disability")
            }
            
            schemes = load_schemes(CSV_FILE)
            # Combine non-empty fields into one query string
            user_query = " ".join([str(value) for key, value in form_data.items() if value])
            if not user_query.strip():
                return render_template("result.html", schemes=[], refined_response="No query provided.")
            
            # Filter schemes by the user's state.
            user_state = form_data.get("state", "").strip().lower()
            schemes_by_state = [s for s in schemes if s["State"].strip().lower() == user_state]
            # Fallback: if no scheme matches the state exactly, use all schemes.
            if not schemes_by_state:
                schemes_by_state = schemes
            
            filtered_schemes = find_best_schemes(user_query, schemes_by_state)
            refined_response = refine_response(filtered_schemes, user_query) if filtered_schemes else "No relevant scheme found."
            print(user_query)
            return render_template("result.html", schemes=filtered_schemes, refined_response=refined_response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return render_template("form.html")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
