import pandas as pd
import re
from typing import Optional

# --- Guarded Optional Imports ---
try:
    import phonenumbers
    _HAS_PHONENUMBERS = True
except ImportError:
    _HAS_PHONENUMBERS = False

try:
    from email_validator import validate_email, EmailNotValidError
    _HAS_EMAIL_VALIDATOR = True
except ImportError:
    _HAS_EMAIL_VALIDATOR = False

try:
    import usaddress
    _HAS_USADDRESS = True
except ImportError:
    _HAS_USADDRESS = False

try:
    from nameparser import HumanName
    _HAS_NAMEPARSER = True
except ImportError:
    _HAS_NAMEPARSER = False

# --- Address Abbreviation Dictionary ---
_ADDR_ABBR = {
    r"\bSt\b": "Street", r"\bAve\b": "Avenue", r"\bBlvd\b": "Boulevard",
    r"\bRd\b": "Road", r"\bLn\b": "Lane", r"\bDr\b": "Drive",
    r"\bApt\b": "Apartment", r"\bSte\b": "Suite", r"\bPO Box\b": "PO Box"
}

# --- Validator Functions ---

def audit_phone(df, col):
    """Detects invalid phones, float noise, or formatting issues."""
    samples = df[col].dropna().astype(str).str.strip()
    if samples.empty: return None

    # Check 1: Float Noise (216.0)
    if samples.str.contains(r'\.0$', na=False).any():
        bad = samples[samples.str.contains(r'\.0$', na=False)].iloc[0]
        return {
            "attribute": col, "issue_detected": "Numeric Float Noise",
            "example_incorrect": bad, "example_corrected": re.sub(r'\.0$', '', bad), "impact": "High"
        }

    # Check 2: Global Validity (using phonenumbers)
    if _HAS_PHONENUMBERS:
        for val in samples.head(20):
            try:
                parsed = phonenumbers.parse(val, "US")
                if not phonenumbers.is_valid_number(parsed): raise Exception()
            except:
                return {
                    "attribute": col, "issue_detected": "Invalid Phone Pattern",
                    "example_incorrect": val, "example_corrected": "+1 (XXX) XXX-XXXX", "impact": "High"
                }
    return None

def audit_email(df, col):
    """Detects malformed emails using email-validator."""
    samples = df[col].dropna().astype(str).str.strip()
    if samples.empty: return None

    for val in samples.head(20):
        if _HAS_EMAIL_VALIDATOR:
            try:
                validate_email(val, check_deliverability=False)
            except EmailNotValidError:
                return {
                    "attribute": col, "issue_detected": "Malformed Email",
                    "example_incorrect": val, "example_corrected": "user@domain.com", "impact": "High"
                }
    return None

def audit_address(df, col):
    """Checks for abbreviations and structural issues."""
    samples = df[col].dropna().astype(str)
    if samples.empty: return None

    # Check if address contains abbreviations that should be expanded
    has_abbr = any(re.search(pat, str(val), re.I) for val in samples.head(20) for pat in _ADDR_ABBR.keys())
    
    if has_abbr:
        bad = samples.iloc[0]
        corrected = bad
        for pat, rep in _ADDR_ABBR.items():
            corrected = re.sub(pat, rep, corrected, flags=re.I)
        return {
            "attribute": col, "issue_detected": "Unexpanded Abbreviations",
            "example_incorrect": bad, "example_corrected": corrected, "impact": "Low"
        }
    return None

def audit_casing(df, col):
    """Detects systemic casing issues and proposes smart Title Case."""
    if any(skip in col.lower() for skip in ['id', 'key', 'code']): return None
    
    series = df[col].dropna().astype(str)
    if series.empty: return None

    is_mostly_caps = (series.str.isupper().sum() / len(series)) > 0.6
    if is_mostly_caps:
        val = series.iloc[0]
        # Smart correction for Mc/Mac/O'
        corrected = val.title()
        corrected = re.sub(r"\bMc([a-z])", lambda m: "Mc" + m.group(1).upper(), corrected)
        corrected = re.sub(r"O'([a-z])", lambda m: "O'" + m.group(1).upper(), corrected)
        
        return {
            "attribute": col, "issue_detected": "Systemic UPPERCASE",
            "example_incorrect": val, "example_corrected": corrected, "impact": "Med"
        }
    return None

def audit_date(df, col):
    """Detects inconsistent date formats."""
    samples = df[col].dropna().astype(str)
    if samples.empty: return None
    
    # Try parsing first 5 samples with pandas
    try:
        pd.to_datetime(samples.head(5), errors='raise')
    except:
        return {
            "attribute": col, "issue_detected": "Inconsistent Date Format",
            "example_incorrect": str(samples.iloc[0]), "example_corrected": "YYYY-MM-DD", "impact": "High"
        }
    return None

# --- Main Orchestrator ---

def ai_cleanse_and_enrich(df_sample):
    """Performs a global quality audit on the dataframe sample."""
    findings = []
    
    for col in df_sample.columns:
        col_lower = col.lower()
        result = None
        
        # Routing Logic
        if any(k in col_lower for k in ['phone', 'tel', 'mob']):
            result = audit_phone(df_sample, col)
        elif 'email' in col_lower or 'mail' in col_lower:
            result = audit_email(df_sample, col)
        elif any(k in col_lower for k in ['addr', 'street', 'road']):
            result = audit_address(df_sample, col)
        elif any(k in col_lower for k in ['date', 'time', 'created', 'updated']):
            result = audit_date(df_sample, col)
        elif any(k in col_lower for k in ['name', 'city', 'state', 'country', 'desc']):
            result = audit_casing(df_sample, col)
            
        if result:
            findings.append(result)

    if not findings:
        return [{
            "attribute": "Dataset", "issue_detected": "Healthy: No systemic issues found", 
            "example_incorrect": "N/A", "example_corrected": "N/A", "impact": "None"
        }]
        
    return findings