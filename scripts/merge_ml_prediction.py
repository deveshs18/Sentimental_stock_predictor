import pandas as pd
import os

# Paths
predict_growth_path = os.path.join('data', 'processed', 'predict_growth.csv')
ml_predictions_path = os.path.join('data', 'ml_predictions.csv')
output_path = predict_growth_path  # Overwrite or update in place

# Load data
if os.path.exists(predict_growth_path):
    growth_df = pd.read_csv(predict_growth_path)
else:
    print(f"File not found: {predict_growth_path}")
    exit(1)

if os.path.exists(ml_predictions_path):
    ml_df = pd.read_csv(ml_predictions_path)
else:
    print(f"File not found: {ml_predictions_path}")
    exit(1)

# Merge on 'company' (ticker), only if ml_prediction column doesn't exist
if 'ml_prediction' not in growth_df.columns:
    merged_df = pd.merge(growth_df, ml_df, on='company', how='left')
else:
    # If it exists, update it with the new predictions
    growth_df.set_index('company', inplace=True)
    ml_df.set_index('company', inplace=True)
    growth_df.update(ml_df)
    growth_df.reset_index(inplace=True)
    merged_df = growth_df

# Save updated file
merged_df.to_csv(output_path, index=False)
print(f"✅ Merged ML predictions into {output_path}")
