import streamlit as st
import pandas as pd
import json
from core.ingestor import process_file
from core.connectors import fetch_from_url, fetch_from_db, fetch_from_s3
from core.profiler import profile_data, get_ai_reasoning
from core.enricher import ai_cleanse_and_enrich
from core.matcher import discover_match_rules
from core.simulator import parse_and_simulate

# --- 1. UI Configuration ---
st.set_page_config(page_title="Semantic MDM Studio", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .ai-box { background-color: #161b22; border: 1px solid #ff4b4b; padding: 20px; border-radius: 10px; margin-bottom: 25px; }
    .stButton>button { border-radius: 5px; background-color: #ff4b4b; color: white; font-weight: bold; width: 100%; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; color: #ff4b4b; }
    .rule-editor { background: #1c2128; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Initialize Session State ---
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = "01 Data Ingestion"
if 'master_data' not in st.session_state:
    st.session_state['master_data'] = None
if 'discovered_rules' not in st.session_state:
    st.session_state['discovered_rules'] = None
if 'ai_profile_analysis' not in st.session_state:
    st.session_state['ai_profile_analysis'] = None

# --- 3. Sidebar Navigation ---
with st.sidebar:
    st.title("🛡️ MDM Workflow")
    steps = ["01 Data Ingestion", "02 Data Profiling", "03 Quality & Enrichment", "04 Match Rule Creator", "05 Data Simulation"]
    curr_idx = steps.index(st.session_state['current_step'])
    
    for i, s in enumerate(steps):
        if i == curr_idx: st.markdown(f"🔵 **{s}**")
        elif i < curr_idx: st.markdown(f"✅ {s}")
        else: st.markdown(f"⚪ {s}")
    
    st.divider()
    if 'master_data' in st.session_state:
        nav = st.selectbox("Quick Jump", steps, index=curr_idx)
        if nav != st.session_state['current_step']:
            st.session_state['current_step'] = nav
            st.rerun()

    if st.button("Reset Agent Memory"):
        st.session_state.clear()
        st.rerun()

st.title("🤖 MDM Simulation Studio")

# --- STEP 1: DATA INGESTION ---
if st.session_state['current_step'] == "01 Data Ingestion":
    st.header("📥 Data Ingestion")
    src = st.segmented_control("Connector", ["File", "S3", "DB"], default="File")
    
    if st.session_state['master_data'] is None:
        df_in = None
        if src == "File":
            f = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
            if f: df_in = process_file(f)
        elif src == "S3":
            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    b = st.text_input("Bucket Name")
                    ak = st.text_input("Access Key", type="password")
                with c2:
                    k = st.text_input("Key Path")
                    sk = st.text_input("Secret Key", type="password")
                if st.button("Connect to S3"): df_in = fetch_from_s3(b, k, ak, sk, "us-east-1")
        elif src == "DB":
            with st.container(border=True):
                conn = st.text_input("Conn String")
                q = st.text_area("SQL Query", "SELECT * FROM users LIMIT 1000")
                if st.button("Execute"): df_in = fetch_from_db(conn, q)
        
        if df_in is not None:
            st.session_state['master_data'] = df_in
            st.rerun()
    else:
        st.success("✅ Data Loaded Successfully")
        st.dataframe(st.session_state['master_data'].head(10), use_container_width=True)
        if st.button("➔ Proceed to Data Profiling", type="primary"):
            st.session_state['current_step'] = "02 Data Profiling"
            st.rerun()

# --- STEP 2: DATA PROFILING (WITH AI STRATEGY) ---
elif st.session_state['current_step'] == "02 Data Profiling":
    st.header("📊 Deep Data Profiler")
    p_df, samples = profile_data(st.session_state['master_data'])
    
    st.markdown('<div class="ai-box">', unsafe_allow_html=True)
    if st.session_state['ai_profile_analysis'] is None:
        st.info("The AI hasn't analyzed your data yet. Click below for an MDM Strategy Recommendation.")
        if st.button("🧠 Induce MDM Strategy with Groq"):
            with st.spinner("Analyzing data patterns..."):
                st.session_state['ai_profile_analysis'] = get_ai_reasoning(p_df)
                st.rerun()
    else:
        st.subheader("✨ AI Architectural Recommendations")
        ai_rec_df = pd.DataFrame(st.session_state['ai_profile_analysis'])
        
        def highlight_match(val):
            color = '#1f6e2e' if val == 'Exact' else '#b58900' if val == 'Fuzzy' else '#444'
            return f'background-color: {color}; color: white; font-weight: bold; border-radius: 5px;'

        st.dataframe(ai_rec_df.style.applymap(highlight_match, subset=['match_type']), use_container_width=True)
        
        if st.button("Re-Run AI Analysis 🔄"):
            st.session_state['ai_profile_analysis'] = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📋 Statistical Inventory")
    st.dataframe(p_df, use_container_width=True, hide_index=True)
    
    if st.button("➔ Proceed to Quality Audit", type="primary"):
        st.session_state['current_step'] = "03 Quality & Enrichment"
        st.rerun()

# --- STEP 3: QUALITY & ENRICHMENT ---
elif st.session_state['current_step'] == "03 Quality & Enrichment":
    st.header("✨ Deep Quality Audit")
    if st.button("🚀 Run Systemic Audit", type="primary"):
        with st.spinner("Auditing structural integrity..."):
            sample = st.session_state['master_data'].head(100)
            st.session_state['quality_audit'] = ai_cleanse_and_enrich(sample)
            st.rerun()

    if st.session_state.get('quality_audit'):
        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
        st.table(pd.DataFrame(st.session_state['quality_audit']))
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Proceed to Match Rule Creator ➔", type="primary"):
            st.session_state['current_step'] = "04 Match Rule Creator"
            st.rerun()

# --- STEP 4: RULE CREATOR (HUMAN-IN-THE-LOOP WORKBENCH) ---
elif st.session_state['current_step'] == "04 Match Rule Creator":
    st.header("🧬 AI Match Rule Designer")
    
    if st.button("🔍 Induce AI Rule Proposals", type="primary"):
        with st.spinner("AI Architect is designing rules based on vector clusters..."):
            st.session_state['discovered_rules'] = discover_match_rules(st.session_state['master_data'])

    rules = st.session_state.get('discovered_rules')
    if rules is not None and not rules.empty:
        st.subheader("🛠️ Architectural Review Board")
        st.info("The Agent has proposed the following rules. You can edit the logic directly below.")
        
        edited_list = []
        for i, row in rules.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    name = st.text_input(f"Rule Name {i}", value=row['rule_name'], key=f"n_{i}")
                    # HUMAN LOOP: The logic text area is now fully editable
                    logic = st.text_area(f"Logic {i}", value=row['logic'], key=f"l_{i}")
                with c2:
                    conf = st.slider(f"Confidence {i}", 0.0, 1.0, float(row['confidence']), key=f"c_{i}")
                    cat = "🔵 AUTO" if conf >= 0.90 else "🟡 POTENTIAL"
                    st.markdown(f"**Strategy:** {cat}")
                
                edited_list.append({
                    "rule_name": name, 
                    "logic": logic, 
                    "confidence": conf, 
                    "rule_reasoning": row.get('rule_reasoning', '')
                })
        
        if st.button("💾 Save Rules & Run Simulation", type="primary"):
            st.session_state['discovered_rules'] = pd.DataFrame(edited_list)
            st.session_state['current_step'] = "05 Data Simulation"
            st.rerun()

# --- STEP 5: SIMULATION ---
elif st.session_state['current_step'] == "05 Data Simulation":
    st.header("🧪 MDM Impact Simulation")
    
    if st.button("🚀 Execute Final Simulation", type="primary"):
        with st.spinner("Processing data merges..."):
            report, auto_tot, pot_tot = parse_and_simulate(
                st.session_state['master_data'], 
                st.session_state['discovered_rules']
            )
            st.session_state['sim_results'] = report
            st.session_state['auto_tot'] = auto_tot
            st.session_state['pot_tot'] = pot_tot

    if 'sim_results' in st.session_state:
        colA, colB = st.columns(2)
        colA.metric("Auto-Merged Records", st.session_state['auto_tot'], help="Logic Confidence >= 90%")
        colB.metric("Potential Matches", st.session_state['pot_tot'], help="Logic Confidence < 90%")
        
        for res in st.session_state['sim_results']:
            label = "✅ [AUTO]" if "Automatic" in res['category'] else "⚠️ [POTENTIAL]"
            with st.expander(f"{label} {res['rule']} - {res['count']} Records Affected"):
                st.markdown(f"**Applied Logic:** `{res['logic']}`")
                st.dataframe(res['data'], use_container_width=True)