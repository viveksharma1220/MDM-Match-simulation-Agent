import hashlib
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Embedding Model (MiniLM is fast and efficient)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- SECTION 1: ROBUST JSON REPAIR (From llm.py reference) ---
def _clean_json_text(text):
    """Fixes common LLM JSON errors like trailing commas or Python literals."""
    text = re.sub(r'```json\s*|```', '', text) # Remove fences
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r',\s*([}\]])', r'\1', text) # Trailing commas
    return text.strip()

# --- SECTION 2: VECTOR DISCOVERY ENGINE (From vectorizer.py reference) ---
def _get_row_text(df):
    """Converts rows to descriptive strings for embedding."""
    return df.fillna('').astype(str).apply(lambda x: " | ".join([f"{c}: {v}" for c, v in x.items() if v]), axis=1).tolist()

def find_semantic_clusters(df, threshold=0.85):
    """Finds record pairs that are 'semantically' similar but might fail exact match."""
    if len(df) < 2: return []
    
    texts = _get_row_text(df)
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    
    # Compute Cosine Similarity
    sim_matrix = embeddings @ embeddings.T
    
    # Extract pairs above threshold (excluding self-match)
    pairs = []
    rows, cols = np.where((sim_matrix >= threshold) & (np.triu(np.ones(sim_matrix.shape), k=1) > 0))
    
    for r, c in zip(rows, cols):
        pairs.append({
            "record_a": df.iloc[r].to_dict(),
            "record_b": df.iloc[c].to_dict(),
            "similarity": round(float(sim_matrix[r, c]), 4)
        })
    return pairs[:5] # Send top 5 most similar clusters to AI as evidence

# --- SECTION 3: THE TWO-PASS ARCHITECT ---
def discover_match_rules(df, threshold=0.85):
    """
    Main Entry Point.
    Pass 1: Cluster Analysis (Vector)
    Pass 2: Semantic Logic Induction (AI)
    """
    if df is None or df.empty: return pd.DataFrame()

    # 1. Evidence Gathering (Vector Pass)
    evidence = find_semantic_clusters(df, threshold)
    
    # 2. Preparation for AI
    groq_key = st.secrets.get("groq", {}).get("matcher_api_key")
    if not groq_key:
        return pd.DataFrame([{"rule_name": "Error", "logic": "API Key Missing"}])
    
    client = Groq(api_key=groq_key)

    # THE MASTER PROMPT (Combines features of your code and the reference)
    prompt = f"""
    Act as a Senior MDM Architect. Analyze the following evidence clusters found via Vector Similarity.
    
    EVIDENCE:
    {json.dumps(evidence, indent=2)}

    TASK:
    Generate a JSON Match Configuration. Follow these Reltio-inspired rules:
    - Use 'Automatic' for high-confidence IDs (SSN, NPI, Email).
    - Use 'Suspect' for Fuzzy names and addresses.
    - Create 'Composite Rules' using AND/OR logic.
    - Use MDM syntax like Exact(field), Fuzzy(field), and ExactOrNull(field).

    OUTPUT FORMAT (Strict JSON Array):
    [{{
        "rule_name": "Descriptive Name",
        "type": "Automatic | Suspect",
        "logic": "(Exact(Email)) OR (Fuzzy(Name) AND Exact(DOB))",
        "confidence": 0.0-1.0,
        "rule_reasoning": "Detailed MDM explanation",
        "score_justification": "Mathematical basis for the confidence score"
    }}]
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an MDM architect. Output only raw JSON arrays. Do not truncate."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        content = _clean_json_text(completion.choices[0].message.content)
        data = json.loads(content)
        
        # Flatten dictionary if LLM wraps it in a key
        rules = data if isinstance(data, list) else list(data.values())[0]
        return pd.DataFrame(rules)

    except Exception as e:
        return pd.DataFrame([{"rule_name": "Pipeline Error", "logic": str(e), "confidence": 0}])