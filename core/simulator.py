import pandas as pd
import re

def parse_and_simulate(df, rules_df):
    if df is None or rules_df is None or rules_df.empty:
        return [], 0, 0

    simulation_report = []
    total_auto = 0
    total_potential = 0

    # Process only the rules that the user checked 'True'
    active_rules = rules_df[rules_df['selected'] == True].copy()

    for _, rule in active_rules.iterrows():
        logic_string = str(rule.get('logic', ''))
        rule_name = rule.get('rule_name', 'Match Rule')
        conf = float(rule.get('confidence', 0))
        
        # 1. EXTRACT THE COLUMN NAME
        # This regex looks for 'Exact(npi_id)' and pulls out 'npi_id'
        match = re.search(r'Exact\((.*?)\)', logic_string, re.IGNORECASE)
        
        if not match:
            continue
            
        target_col = match.group(1).strip().replace("'", "").replace('"', "")

        # 2. VALIDATE COLUMN EXISTS
        if target_col not in df.columns:
            continue

        # 3. FIND THE TWINS (EXACT MATCHES)
        # We find rows where this specific column has the same value
        is_duplicate = df.duplicated(subset=[target_col], keep=False)
        matched_data = df[is_duplicate].copy()

        if not matched_data.empty:
            # Sort by the ID so twins appear together
            matched_data = matched_data.sort_values(by=target_col)
            count = len(matched_data)
            
            if conf >= 0.90: total_auto += count
            else: total_potential += count
            
            simulation_report.append({
                "rule": f"{rule_name} (on {target_col})",
                "category": "Automatic Merge" if conf >= 0.90 else "Potential Match",
                "logic": f"Exact Match on '{target_col}'",
                "count": count,
                "data": matched_data
            })

    return simulation_report, total_auto, total_potential