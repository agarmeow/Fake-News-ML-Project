import pandas as pd
import numpy as np

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)

def clean_v(v):
    if pd.isna(v):
        return "UNVERIFIABLE"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    v = str(v).strip().upper()
    return v if v in ["TRUE", "FALSE", "UNVERIFIABLE"] else "UNVERIFIABLE"

df = pd.read_csv("results/verifier_relation_pattern.csv")
df['gold'] = df['gold'].apply(clean_v)

n = len(df)
print(f"Total rows: {n}")

for c in ['c2', 'c3', 'c4']:
    v_col = f'{c}_verdict'
    df[v_col] = df[v_col].apply(clean_v)
    corr = (df[v_col] == df['gold']).sum()
    unv = (df[v_col] == 'UNVERIFIABLE').sum()
    p, lo, hi = wilson_ci(corr, n)
    print(f"{c.upper()}: Correct={corr}/{n}, Acc={p*100:.1f}% [Wilson 95% CI: {lo*100:.1f}%, {hi*100:.1f}%], UNVERIFIABLE={unv}/{n} ({unv/n*100:.1f}%)")

# Paired C3 vs C2
diff = df[df['c3_verdict'] != df['c2_verdict']]
print(f"\nPaired C3 vs C2 differences (count={len(diff)}):")
for _, r in diff.iterrows():
    print(f"  id={r['id']}, gold={r['gold']}, C2={r['c2_verdict']}, C3={r['c3_verdict']}, claim='{r['claim_n']}'")

summary_text = f"""Pattern Relation Evaluation Summary (N={n})
--------------------------------------------------
C2: Accuracy = {(df['c2_verdict'] == df['gold']).sum()}/{n} ({(df['c2_verdict'] == df['gold']).mean()*100:.1f}%), UNVERIFIABLE = {(df['c2_verdict'] == 'UNVERIFIABLE').sum()}/{n} ({(df['c2_verdict'] == 'UNVERIFIABLE').mean()*100:.1f}%)
C3: Accuracy = {(df['c3_verdict'] == df['gold']).sum()}/{n} ({(df['c3_verdict'] == df['gold']).mean()*100:.1f}%), UNVERIFIABLE = {(df['c3_verdict'] == 'UNVERIFIABLE').sum()}/{n} ({(df['c3_verdict'] == 'UNVERIFIABLE').mean()*100:.1f}%)
C4: Accuracy = {(df['c4_verdict'] == df['gold']).sum()}/{n} ({(df['c4_verdict'] == df['gold']).mean()*100:.1f}%), UNVERIFIABLE = {(df['c4_verdict'] == 'UNVERIFIABLE').sum()}/{n} ({(df['c4_verdict'] == 'UNVERIFIABLE').mean()*100:.1f}%)

Rows where C3 != C2:
{diff[['id', 'gold', 'c2_verdict', 'c3_verdict', 'claim_n']].to_string(index=False)}
"""
with open("results/verifier_relation_pattern_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)
