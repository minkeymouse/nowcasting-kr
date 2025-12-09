#!/usr/bin/env python3
"""Verify experiment results and generate summary statistics."""
import pandas as pd

df = pd.read_csv('outputs/experiments/aggregated_results.csv')

print('=== Summary Statistics ===')
print(f'\nTotal rows: {len(df)}')
print(f'\nBy Model (valid results):')
for model in ['VAR', 'DFM', 'DDFM', 'ARIMA']:
    model_df = df[df['model']==model]
    valid = model_df['n_valid'].sum()
    print(f'  {model}: {valid} valid results')

print(f'\nBy Target (average sMAE):')
for target in ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']:
    target_df = df[df['target']==target]
    print(f'\n  {target}:')
    for model in ['VAR', 'DFM', 'DDFM']:
        model_target = target_df[target_df['model']==model]
        if len(model_target) > 0 and model_target['n_valid'].sum() > 0:
            avg_smae = model_target['sMAE'].mean()
            print(f'    {model}: sMAE={avg_smae:.4f}')

print(f'\nKOEQUIPTE DDFM vs DFM comparison:')
koe_ddfm = df[(df['target']=='KOEQUIPTE') & (df['model']=='DDFM')]
koe_dfm = df[(df['target']=='KOEQUIPTE') & (df['model']=='DFM')]
if len(koe_ddfm) > 0 and len(koe_dfm) > 0:
    ddfm_avg = koe_ddfm['sMAE'].mean()
    dfm_avg = koe_dfm['sMAE'].mean()
    diff = abs(ddfm_avg - dfm_avg)
    print(f'  DDFM avg sMAE: {ddfm_avg:.6f}')
    print(f'  DFM avg sMAE: {dfm_avg:.6f}')
    print(f'  Difference: {diff:.6f} ({diff/dfm_avg*100:.4f}%)')
