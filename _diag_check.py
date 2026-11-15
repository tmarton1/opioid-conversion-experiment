from common.config import RESULTS_DIR
import pandas as pd

df = pd.read_csv(f'{RESULTS_DIR}/gemini_diagnostic_results.csv')
print(f"Total rows: {len(df)}")
print(f"  Success: {(df['status']=='success').sum()}")
print(f"  Error:   {(df['status']=='error').sum()}")

errs = df[df['status'] == 'error']
print(f"\n=== Error types ===")
print(errs['error_type'].value_counts().to_string())

print(f"\n=== Error messages (unique, truncated) ===")
for msg in errs['error_message'].unique():
    print(f"  {str(msg)[:200]}")

print(f"\n=== Error rate by opioid ===")
by_op = df.groupby('from_opioid').agg(
    total=('status','size'),
    errors=('status', lambda x: (x=='error').sum())
)
by_op['error_rate'] = (by_op['errors'] / by_op['total'] * 100).round(1)
print(by_op[by_op['errors'] > 0].to_string())

print(f"\n=== Error rate by explain_units ===")
by_eu = df.groupby('explain_units').agg(
    total=('status','size'),
    errors=('status', lambda x: (x=='error').sum())
)
by_eu['error_rate'] = (by_eu['errors'] / by_eu['total'] * 100).round(1)
print(by_eu.to_string())

err_with_resp = errs[errs['response'].notna()]
print(f"\n=== Errors with response text: {len(err_with_resp)} / {len(errs)} ===")
if len(err_with_resp) > 0:
    print("  (model responded but dose extraction failed)")
    for _, r in err_with_resp.head(3).iterrows():
        print(f"  {r['from_opioid']} explain={r['explain_units']}: {str(r['response'])[:120]}")
