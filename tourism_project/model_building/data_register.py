
import pandas as pd
raw_path = 'tourism_project/data/tourism.csv'
# Load raw dataset
df = pd.read_csv(raw_path)
expected_columns = ['CustomerID','ProdTaken','Age','TypeofContact','CityTier','DurationOfPitch','Occupation','Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
                    'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']
missing = [col for col in expected_columns if col not in df.columns]
if missing:
  raise ValueError("Missing columns:" + ', '.join(missing) )
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
print("Summary")
df.describe()

