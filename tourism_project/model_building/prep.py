import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Load Data
df = pd.read_csv('tourism_project/data/tourism.csv')

# Dropping unnecessary columns
df.drop(columns=['Unnamed: 0', 'CustomerID'], axis=1, inplace=True, errors='ignore')

# Convert Columns to Categorical explicitly
cat_mapping_cols = [
    'ProdTaken', 'TypeofContact', 'CityTier', 'Occupation', 'Gender',
    'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'Designation'
]
for col in cat_mapping_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Create Age Buckets and drop the original continuous Age column
df['Age_bucket'] = pd.cut(df['Age'], bins=[0, 18, 25, 35, 50, 100], labels=['0-18', '19-25', '26-35', '36-50', '51+'])
df.drop(['Age'], axis=1, inplace=True, errors='ignore')

# Handle the Gender categorical replacement safely
df['Gender'] = df['Gender'].astype(str).replace({'Fe Male': 'Female'})
df['Gender'] = df['Gender'].astype('category')

# Drop Duplicates
df.drop_duplicates(inplace=True)

# Drop Gender and OwnCar based on Chi-Square statistical results
df.drop(['Gender', 'OwnCar'], inplace=True, axis=1, errors='ignore')

# =====================================================================
# PHASE 1 FEATURE ENGINEERING (ROW-LEVEL)
# =====================================================================

# 1. Interaction Efficiency (Cast to float first to fix Numpy division TypeError)
df['PitchSatisfactionScore'] = df['PitchSatisfactionScore'].astype(float)
df['DurationOfPitch'] = df['DurationOfPitch'].astype(float)
df['Interaction_Efficiency'] = df['PitchSatisfactionScore'] / (df['DurationOfPitch'] + 0.1)

# 2. Total Group Size
df['NumberOfPersonVisiting'] = df['NumberOfPersonVisiting'].astype(float)
df['NumberOfChildrenVisiting'] = df['NumberOfChildrenVisiting'].astype(float)
df['Total_Group_Size'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']

# 3. Life Stage Label
df['Life_Stage'] = df['MaritalStatus'].astype(str) + "_" + df['Age_bucket'].astype(str)

# 4. Golden Demographic Flag (Young Executives)
df['Is_Young_Executive'] = np.where(
    (df['Designation'].astype(str) == 'Executive') & (df['Age_bucket'].astype(str).isin(['19-25', '26-35'])),
    1, 0
)

# 5. Pitch Value Alignment
premium_products = ['King', 'Deluxe', 'Super Deluxe']
premium_designations = ['VP', 'Director', 'Senior Manager']
is_premium_pitch = df['ProductPitched'].astype(str).isin(premium_products)
is_premium_customer = df['Designation'].astype(str).isin(premium_designations)
df['Pitch_Alignment'] = np.where(is_premium_pitch == is_premium_customer, 1, 0)

# =====================================================================
# DROPPING REDUNDANT RAW SOURCE COLUMNS
# =====================================================================
columns_to_remove_after_engineering = [
    'Designation',
    'MaritalStatus',
    'NumberOfPersonVisiting',
    'NumberOfChildrenVisiting'
]
df.drop(columns=columns_to_remove_after_engineering, axis=1, inplace=True, errors='ignore')

# Separate Features and Target
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

# Ensure target is loaded as numeric binary integers for modeling
y = y.astype(int)

# =====================================================================
# EXACT 3-WAY SPLIT SYSTEM (64 / 16 / 20 Ratio)
# =====================================================================
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

# Save Splits to CSV files
output_dir = "data-splits"
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(os.path.join(output_dir, "Xtrain.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "Xtest.csv"), index=False)
X_val.to_csv(os.path.join(output_dir, "Xval.csv"), index=False) 
y_train.to_csv(os.path.join(output_dir, "ytrain.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "ytest.csv"), index=False)
y_val.to_csv(os.path.join(output_dir, "yval.csv"), index=False)



print("Data successfully prepared: train/val/test splits written.")
