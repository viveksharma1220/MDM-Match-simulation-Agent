import pandas as pd
import numpy as np
import json
import re
import streamlit as st
from collections import Counter
from groq import Groq

# Optional: For phonetic simulation
try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

# --- Format Pattern Detectors ---
_ALL_DIGITS    = re.compile(r"^\d+$")
_ALPHA_ONLY    = re.compile(r"^[A-Za-z\s\-'.]+$")
_DATE_LIKE     = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}")
_EMAIL_LIKE    = re.compile(r"@[\w.\-]+\.[a-z]{2,}", re.I)
_PHONE_LIKE    = re.compile(r"^\+?[\d\s\-().]{7,15}$")
_ZIP_LIKE      = re.compile(r"^\d{5}(-\d{4})?$")

def detect_format_hint(values):
    """Deep pattern analysis of sample values."""
    sample = [str(v).strip() for v in values if str(v).strip()]
    if not sample: return "empty/null"

    n = len(sample)
    def pct(k): return k / n

    cnt_email = sum(1 for v in sample if _EMAIL_LIKE.search(v))
    cnt_phone = sum(1 for v in sample if _PHONE_LIKE.match(v))
    cnt_date  = sum(1 for v in sample if _DATE_LIKE.search(v))
    cnt_digits = sum(1 for v in sample if _ALL_DIGITS.match(v))
    
    if pct(cnt_email) > 0.7: return "Email Address"
    if pct(cnt_phone) > 0.7: return "Phone Number"
    if pct(cnt_date) > 0.7: return "Date/Timestamp"
    if pct(cnt_digits) > 0.85: return "Numeric ID/Code"
    
    # Structural Pattern Detection (e.g., CA-1234 -> XX-NNNN)
    patterns = [re.sub(r"[A-Za-z]", "X", re.sub(r"\d", "N", v)) for v in sample[:20]]
    if patterns:
        top_pat, top_cnt = Counter(patterns).most_common(1)[0]
        if top_cnt / len(patterns) > 0.6:
            return f"Structured Pattern: {top_pat}"
            
    return "Mixed Text/Unstructured"

def profile_data(df):
    """Generates a deep statistical profile with format hints."""
    stats = []
    total_rows = len(df)
    
    for col in df.columns:
        series = df[col]
        # Clean data for analysis
        clean_series = series.dropna().astype(str).str.strip()
        clean_series = clean_series[clean_series != ""]
        
        n_unique = series.nunique()
        missing_perc = (series.isna().sum() / total_rows * 100) if total_rows > 0 else 0
        unique_perc = (n_unique / total_rows * 100) if total_rows > 0 else 0
        
        # Deep Format Analysis
        samples = clean_series.value_counts().head(30).index.tolist()
        fmt_hint = detect_format_hint(samples)
        
        # Length Stats
        lens = clean_series.str.len()
        avg_len = round(lens.mean(), 1) if not lens.empty else 0
        
        # Basic Health Logic
        health = "✅ High" if missing_perc < 5 else "⚠️ Review" if missing_perc < 50 else "❌ Poor"
        
        stats.append({
            "Attribute": col,
            "Format Hint": fmt_hint,
            "Quality Health": health,
            "Missing %": round(missing_perc, 2),
            "Uniqueness %": round(unique_perc, 2),
            "Distinct": n_unique,
            "Avg Len": avg_len,
            "Is Numeric": clean_series.str.match(r"^\d+$").sum() / max(len(clean_series), 1) > 0.8,
            "Samples": samples[:5]  # Keep 5 for the UI table
        })
        
    return pd.DataFrame(stats), samples # Returning samples for AI reasoning

def get_ai_reasoning(profile_df):
    """Uses Groq to recommend MDM strategies based on deep profile."""
    api_key = st.secrets.get("groq", {}).get("profiler_api_key")
    if not api_key:
        return [{"column": "Error", "reasoning": "API Key Missing"}]
    
    client = Groq(api_key=api_key)

    # Convert profile to a concise prompt summary
    analysis_payload = []
    for _, row in profile_df.iterrows():
        analysis_payload.append({
            "column": row['Attribute'],
            "format": row['Format Hint'],
            "uniqueness": f"{row['Uniqueness %']}%",
            "missing": f"{row['Missing %']}%",
            "is_numeric": row['Is Numeric'],
            "avg_len": row['Avg Len']
        })

    prompt = f"""
    Act as an MDM (Master Data Management) Architect. 
    Analyze these dataset columns and recommend a matching strategy.
    
    Strategies:
    - 'Exact': Use for unique identifiers, emails, or codes with high uniqueness.
    - 'Fuzzy': Use for names, addresses, or city fields where typos occur.
    - 'None': Use for descriptions, timestamps, or low-uniqueness noise.

    Data Profile:
    {json.dumps(analysis_payload, indent=2)}

    Return a JSON array of objects:
    {{"column": "...", "match_type": "Exact|Fuzzy|None", "confidence": "High|Med|Low", "reasoning": "..."}}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a Data Architect. Respond ONLY with a JSON array."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        res = json.loads(completion.choices[0].message.content)
        # Flatten if Groq wraps in a key
        return res if isinstance(res, list) else list(res.values())[0]

    except Exception as e:
        return [{"column": "Error", "match_type": "None", "reasoning": str(e)}]