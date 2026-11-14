import pandas as pd

def clean_v(v):
    if pd.isna(v):
        return "UNVERIFIABLE"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    v = str(v).strip().upper()
    return v if v in ["TRUE", "FALSE", "UNVERIFIABLE"] else "UNVERIFIABLE"

df = pd.read_csv("results/verifier_ablation.csv")

# REPORT set: exclude T01-T06, F01-F06
exclude_ids = [f"T{i:02d}" for i in range(1, 7)] + [f"F{i:02d}" for i in range(1, 7)]
df = df[~df['id'].isin(exclude_ids)].copy()

n = len(df)
print(f"REPORT set count: {n}")

df['gold'] = df['gold'].apply(clean_v)
df['c4_verdict'] = df['c4_verdict'].apply(clean_v)
df['c4_conflict'] = df['c4_conflict'].fillna(False).astype(bool)

# C4 metrics
c4_corr = (df['c4_verdict'] == df['gold']).sum()
c4_acc = c4_corr / n
c4_decisive_wrong = ((df['c4_verdict'].isin(['TRUE', 'FALSE'])) & (df['c4_verdict'] != df['gold'])).sum()
c4_abstentions = (df['c4_verdict'] == 'UNVERIFIABLE').sum()

# C4c: set conflict == True rows to UNVERIFIABLE
df['c4c_verdict'] = df['c4_verdict']
df.loc[df['c4_conflict'] == True, 'c4c_verdict'] = 'UNVERIFIABLE'

c4c_corr = (df['c4c_verdict'] == df['gold']).sum()
c4c_acc = c4c_corr / n
c4c_decisive_wrong = ((df['c4c_verdict'].isin(['TRUE', 'FALSE'])) & (df['c4c_verdict'] != df['gold'])).sum()
c4c_abstentions = (df['c4c_verdict'] == 'UNVERIFIABLE').sum()

print("\n--- post hoc, not pre-registered ---")
print(f"Metric\t\t\tC4\t\tC4c")
print(f"Accuracy\t\t{c4_corr}/{n} ({c4_acc*100:.1f}%)\t{c4c_corr}/{n} ({c4c_acc*100:.1f}%)")
print(f"Decisive-wrong\t\t{c4_decisive_wrong}\t\t{c4c_decisive_wrong}")
print(f"Abstentions\t\t{c4_abstentions}\t\t{c4c_abstentions}")

# Let's inspect the conflict rows
conflicts = df[df['c4_conflict'] == True]
print(f"\nConflict rows (count={len(conflicts)}):")
for _, r in conflicts.iterrows():
    print(f"  id={r['id']}, gold={r['gold']}, c4={r['c4_verdict']}, c4c={r['c4c_verdict']}, claim='{r['claim_n']}'")
