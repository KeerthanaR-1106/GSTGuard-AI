"""Quick correctness spot-check for Stage 2 output."""
import pandas as pd, numpy as np

feat = pd.read_csv('data/ml_features.csv')
gt   = pd.read_csv('data/ground_truth.csv')

print("=== ml_features.csv checks ===")
print(f"Rows  : {len(feat)}")
print(f"Cols  : {len(feat.columns)}  (invoice_no + 43 features)")
print(f"NaNs  : {feat.isnull().sum().sum()}")
print(f"Infs  : {np.isinf(feat.select_dtypes(include='number')).sum().sum()}")
print()

# Verify no label columns leaked
bad_cols = [c for c in feat.columns if c in ('actual_anomaly','anomaly_type')]
print(f"Label columns leaked: {bad_cols if bad_cols else 'NONE (good)'}")

# Spot-check: anomaly invoices should have higher deviations
merged = feat.merge(gt, on='invoice_no')

anom   = merged[merged['actual_anomaly']==1]
normal = merged[merged['actual_anomaly']==0]

print()
print("=== Deviation features (mean) normal vs anomalous ===")
for col in ['total_amount_difference','gst_difference','total_tax_difference',
            'amount_to_taxable_ratio','duplicate_transaction_flag',
            'supplier_daily_frequency','future_date_flag']:
    nm = normal[col].mean()
    am = anom[col].mean()
    ratio = am/nm if nm > 0 else float('inf')
    print(f"  {col:<40}: normal={nm:>10.3f}  anomaly={am:>10.3f}  ratio={ratio:>6.1f}x")

print()
print("=== Anomaly type breakdown on key features ===")
for atype in anom['anomaly_type'].unique():
    sub = anom[anom['anomaly_type']==atype]
    r = sub['amount_to_taxable_ratio'].mean()
    d = sub['total_amount_difference'].mean()
    print(f"  {atype:<35}: ratio={r:.2f}, total_amt_diff={d:.0f}")

print()
print("[DONE] Spot-check complete.")
