import pandas as pd
import re

def parse_and_simulate(df, rules_df):
    if df is None or rules_df is None or rules_df.empty:
        return [], 0, 0

    simulation_report = []
    total_auto = 0
    total_potential = 0

    for _, rule in rules_df.iterrows():
        logic_string = str(rule.get('logic', ''))
        rule_name = rule.get('rule_name', 'Unnamed Rule')
        conf = float(rule.get('confidence', 0))
        
        match_category = "Automatic Merge" if conf >= 0.90 else "Potential Match"
        
        # Extract columns and clean them of whitespace/quotes
        matches = re.findall(r'(?:Exact|Fuzzy)\((.*?)\)', logic_string)
        target_cols = [c.strip().replace("'", "").replace('"', "") for c in matches if c.strip() in df.columns]

        if not target_cols:
            continue

        duplicate_mask = df.duplicated(subset=target_cols, keep=False)
        matched_data = df[duplicate_mask].copy()

        if not matched_data.empty:
            count = len(matched_data)
            if conf >= 0.90: total_auto += count
            else: total_potential += count
            
            simulation_report.append({
                "rule": rule_name,
                "category": match_category,
                "logic": logic_string,
                "confidence": conf,
                "count": count,
                "data": matched_data.sort_values(by=target_cols)
            })

    return simulation_report, total_auto, total_potential