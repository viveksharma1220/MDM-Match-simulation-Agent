import json
import re
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# Initialize Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# --- SECTION 1: UTILITIES ---
def _clean_json_text(text):
    """Fixes common LLM JSON errors like trailing commas or Python literals."""
    if not text: return ""
    text = re.sub(r'```json\s*|```', '', text) 
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r',\s*([}\]])', r'\1', text) 
    return text.strip()

# --- SECTION 2: AI SCHEMA FILTER (The "All-In" Pass) ---
def get_identity_columns(df):
    """
    Pass 0: Provide ALL columns to the AI.
    The AI filters out non-identity columns to prevent 'vector noise'.
    """
    all_columns = df.columns.tolist()
    # Provide a significant sample (5 rows) so AI sees the data patterns
    sample_json = df.head(5).to_json()

    groq_key = st.secrets.get("groq", {}).get("matcher_api_key")
    if not groq_key: return all_columns[:5] # Fallback
    
    client = Groq(api_key=groq_key)

    prompt = f"""
    Act as a Master Data Management (MDM) Expert. 
    I am providing you with ALL the columns from a dataset. 
    Your task is to FILTER these columns and only keep those that represent 'Identity' 
    (data that defines who/what a record is).

    FULL COLUMN LIST: {all_columns}
    DATA PREVIEW: {sample_json}

    FILTERING RULES:
    1. KEEP: Names, Emails, Phones, Physical Addresses, SSN, NPI, DOB, Account IDs.
    2. DISCARD: Timestamps (Created_At), Row IDs, Booleans (is_active), 
       Generic Notes, Prices, or system-generated UUIDs that don't help in matching.
    
    Return a JSON object: {{"identity_columns": ["col_name_1", "col_name_2"]}}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        res = json.loads(completion.choices[0].message.content)
        filtered_cols = res.get('identity_columns', [])
        
        # Verify columns actually exist in the dataframe
        valid_cols = [c for c in filtered_cols if c in df.columns]
        
        if not valid_cols:
            st.warning("AI filtered out all columns. Falling back to first 5 columns.")
            return all_columns[:5]
            
        return valid_cols
    except Exception as e:
        st.error(f"AI Filter Error: {e}")
        return all_columns[:5]

# --- SECTION 3: DICTIONARY VECTOR ENGINE ---
def find_semantic_clusters(df, threshold=0.80):
    """
    Uses Dictionary-Based Embedding on AI-filtered columns.
    """
    if len(df) < 2: return []
    
    # AI decides which of ALL columns are useful
    target_cols = get_identity_columns(df)
    
    st.write(f"🛡️ **AI filtered the schema.** Matching based on: `{', '.join(target_cols)}`")

    # 1. Field-Level Embedding
    field_matrices = {}
    for col in target_cols:
        texts = df[col].fillna('').astype(str).tolist()
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
        field_matrices[col] = embeddings @ embeddings.T

    # 2. Candidate Discovery (Unique Pairs)
    num_rows = len(df)
    pairs = []
    rows, cols = np.where(np.triu(np.ones((num_rows, num_rows)), k=1) > 0)
    
    for r, c in zip(rows, cols):
        profile = {col: round(float(field_matrices[col][r, c]), 4) for col in target_cols}
        max_score = max(profile.values())
        
        if max_score > threshold:
            pairs.append({
                "record_a": df.iloc[r].to_dict(),
                "record_b": df.iloc[c].to_dict(),
                "similarity_profile": profile,
                "max_score": max_score
            })

    return sorted(pairs, key=lambda x: x['max_score'], reverse=True)[:5]

# --- SECTION 4: THE RULE INDUCTOR ---
def discover_match_rules(df, threshold=0.80):
    if df is None or df.empty: return pd.DataFrame()

    evidence = find_semantic_clusters(df, threshold)
    if not evidence:
        return pd.DataFrame([{"rule_name": "No Duplicates Found", "logic": "No similarity detected", "confidence": 0}])

    groq_key = st.secrets.get("groq", {}).get("matcher_api_key")
    client = Groq(api_key=groq_key)

    prompt = f"""
    Act as a Senior MDM Architect. Review the following field-level similarity clusters.
    CRITICAL BUSINESS RULES:
    - 'npi_id', 'ssn', and 'email' are UNIQUE identifiers. 
    - NEVER use 'Fuzzy' for these IDs. If the similarity score is high (e.g. > 0.9), always use 'Exact()'.
    - Use 'Fuzzy' only for Names, Addresses, and Cities.
    EVIDENCE:
    {json.dumps(evidence, indent=2)}

    TASK:
    Generate Match Rules in JSON.
    1. 'evidence_score_summary': Brief score breakdown (e.g. Email: 1.0, Name: 0.9).
    2. 'logic': MDM syntax (e.g. Exact(Email) OR Fuzzy(Name)).
    3. Use 'Automatic' for high-confidence matches.

    OUTPUT FORMAT (Strict JSON Array):
    [{{
      [{{
        "rule_name": "Short Name",
        "type": "Automatic | Suspect",
        "logic": "Exact(column_name)",
        "confidence": 0.95
    }}]
    }}]
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Output only JSON arrays."}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = _clean_json_text(completion.choices[0].message.content)
        data = json.loads(content)
        rules = data if isinstance(data, list) else list(data.values())[0]
        return pd.DataFrame(rules)
    except Exception as e:
        return pd.DataFrame([{"rule_name": "Error", "logic": str(e), "confidence": 0}])