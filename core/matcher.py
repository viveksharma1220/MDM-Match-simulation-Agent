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
def discover_match_rules(df, selected_columns):
    """Generates Reltio-formatted match rules using Groq based on selected columns."""
    # Note: Ensure "matcher_api_key" or "profiler_api_key" matches what is in your secrets.toml
    api_key = st.secrets.get("groq", {}).get("matcher_api_key") 
    if not api_key:
        return pd.DataFrame([{"rule_name": "Error", "logic": "API Key Missing", "confidence": 0.0, "selected": False}])

    client = Groq(api_key=api_key)

    # Filter dataframe to ONLY include user-selected columns
    filtered_df = df[selected_columns]

    # Get sample data as CSV string to send to Groq
    sample_data = filtered_df.head(10).to_csv(index=False)

    # Strict prompt to force Reltio JSON Output
    prompt = f"""
    You are an MDM Reltio Match Rule Configuration Expert.
    Analyze the following data sample and generate match rules strictly formatted as Reltio Match Rule JSON objects.

    DATA SAMPLE:
    {sample_data}

    REQUIREMENTS:
    1. Create 2 to 3 distinct Reltio match rules based ONLY on the columns provided in the sample. Do not invent columns.
    2. Output a JSON object containing an array called "rules".
    3. Each object in the "rules" array must have these exact keys:
       - "rule_name": String (e.g., "Exact Name and Fuzzy Address")
       - "confidence": Float between 0.0 and 1.0 (e.g., 0.95)
       - "logic": A stringified JSON representing the actual Reltio match rule config.
       - "rule_reasoning": String explaining why this rule is appropriate for this data.

    CRITICAL RELTIO STRUCTURE: 
    The "logic" field MUST contain a stringified JSON. Inside the "rule" object of this JSON, you MUST include "matchTokenClasses" and "comparatorClasses" that map the chosen attributes to standard Reltio Java classes (e.g., com.reltio.match.comparator.BasicStringComparator, com.reltio.match.token.ExactMatchToken, com.reltio.match.comparator.DoubleMetaphoneComparator, etc.).

    Example of the exact string format expected inside the "logic" field:
    '{{"matchGroups": [{{"uri": "configuration/entityTypes/Contact/matchGroups/Rule1", "label": "Rule 1", "type": "suspect", "rule": {{"matchTokenClasses": {{"mapping": [{{"attribute": "configuration/entityTypes/Contact/attributes/FirstName", "class": "com.reltio.match.token.FuzzyTextMatchToken"}}]}}, "comparatorClasses": {{"mapping": [{{"attribute": "configuration/entityTypes/Contact/attributes/FirstName", "class": "com.reltio.match.comparator.DoubleMetaphoneComparator"}}]}}, "fuzzy": ["configuration/entityTypes/Contact/attributes/FirstName"]}}}}]}}'

    RETURN ONLY VALID JSON.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a Data Architect. Respond ONLY with valid JSON containing a 'rules' array."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        res = json.loads(completion.choices[0].message.content)
        rules_list = res.get("rules", [])
        
        # Add default 'selected' boolean for the Streamlit UI
        for r in rules_list:
            r['selected'] = True
            
        return pd.DataFrame(rules_list)

    except Exception as e:
        return pd.DataFrame([{"rule_name": "Error", "logic": str(e), "confidence": 0.0, "selected": False}])