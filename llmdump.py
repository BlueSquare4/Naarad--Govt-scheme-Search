import os
import glob
import webbrowser
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

SCHEME_FOLDER = "Schemes/"

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

LLM_MODEL = "tiiuae/falcon-7b-instruct"
HF_TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

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
    top_schemes = [schemes[i] for i in top_indices.cpu().numpy()]
    return top_schemes

def generate_detailed_info(top_schemes, user_query):
    selected_schemes = top_schemes[:3]
    schemes_text = "\n\n".join(
        [f"Scheme: {s['name']} (State: {s['state']})\nDescription: {s['description']}\nEligibility: {s.get('eligibility', 'Not specified')}"
         for s in selected_schemes]
    )
    prompt = f"User query: {user_query}\n\nTop 3 schemes:\n{schemes_text}\n\nProvide detailed information on which scheme is best and why."
    response = llm_pipeline(prompt, max_new_tokens=200)[0]["generated_text"].strip()
    return response

def update_result_html(schemes, recommendation):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scheme Results</title>
    </head>
    <body>
        <h2>Recommended Schemes</h2>
        <ul>
    """
    for scheme in schemes:
        html_content += f"""
            <li>
                <strong>{scheme['name']}</strong> (State: {scheme['state']})<br>
                Description: {scheme['description']}<br>
                Eligibility: {scheme.get('eligibility', 'Not specified')}
            </li>
        """
    html_content += f"""
        </ul>
        <h3>Final Recommendation:</h3>
        <p>{recommendation}</p>
        <a href="index.html">Go Back</a>
    </body>
    </html>
    """
    
    with open("result.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    schemes = load_schemes(SCHEME_FOLDER)
    user_input = input("Enter your details (query): ")
    refined_query = improve_query(user_input)
    top_schemes = find_best_schemes(refined_query, schemes)
    final_output = generate_detailed_info(top_schemes, refined_query)
    update_result_html(top_schemes, final_output)

    print("Results saved in result.html. Opening in browser...")
    webbrowser.open("result.html")
