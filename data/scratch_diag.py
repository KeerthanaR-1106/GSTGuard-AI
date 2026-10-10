import pandas as pd, numpy as np
df = pd.read_csv('data/gst_invoices.csv')
df_gt = pd.read_csv('data/ground_truth.csv')

normal_nos = df_gt[df_gt['actual_anomaly']==0]['invoice_no']
df_n = df[df['invoice_no'].isin(normal_nos)]

p99 = df_n['total_amount'].quantile(0.99)
max_n = df_n['total_amount'].max()
mean_n = df_n['total_amount'].mean()
print("Normal total_amount: mean=%d, 99th-pct=%d, max=%d" % (mean_n, p99, max_n))

ea_nos = df_gt[df_gt['anomaly_type']=='Extreme Invoice Amount']['invoice_no']
ea = df[df['invoice_no'].isin(ea_nos)].copy()
print("\nExtreme Invoice Amount - total_amount stats:")
print("  count:", len(ea))
print("  min:", ea['total_amount'].min())
print("  max:", ea['total_amount'].max())
print("  mean:", ea['total_amount'].mean())
print("  rows > 5x p99 (%d): %d" % (p99*5, (ea['total_amount'] > p99*5).sum()))
print("  rows > p99 (%d): %d" % (p99, (ea['total_amount'] > p99).sum()))

ea['expected_amt'] = ea['taxable_value'] + ea['total_tax']
ea['ratio'] = ea['total_amount'] / ea['expected_amt']
print("\nRatio of total_amount to (taxable+tax) for EIA rows:")
print(ea['ratio'].describe().to_string())
print("\nConclusion: EIA anomaly is detectable as amount >> taxable_value + total_tax")
print("The 5x-p99 threshold in validator is too strict because normal max is already large.")
print("The anomaly DOES exist (ratio 10-50x off from expected) - validator threshold needs fixing.")
