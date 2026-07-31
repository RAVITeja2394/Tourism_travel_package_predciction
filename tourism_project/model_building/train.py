import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import joblib  
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
import warnings, sys,os
from sklearn.metrics import classification_report



warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
print("--- STARTING TOURISM CONVERSION ML PIPELINE ---")
# =====================================================================
# 1. LOAD THE PREPARED SPLITS (From prep.py)
# =====================================================================
try:
    Xtrain = pd.read_csv("data-splits/Xtrain.csv")
    Xval = pd.read_csv("data-splits/Xval.csv")
    ytrain = pd.read_csv("data-splits/ytrain.csv").squeeze()
    yval = pd.read_csv("data-splits/yval.csv").squeeze()
except FileNotFoundError as e:
    print(f"Critical Data split files missing. {e}")
    sys.exit(1)

# Force data types to string/object so encoders can read them cleanly
categorical_data_types = ['TypeofContact', 'ProductPitched', 'Occupation', 'Life_Stage',
                          'CityTier', 'PreferredPropertyStar', 'PitchSatisfactionScore', 'Age_bucket']

for col in categorical_data_types:
    if col in Xtrain.columns:
        Xtrain[col] = Xtrain[col].astype(str)
    if col in Xval.columns:
        Xval[col] = Xval[col].astype(str)

# Ensure numeric columns are strictly float/int
numeric_data_types = ['DurationOfPitch', 'MonthlyIncome', 'NumberOfFollowups', 'NumberOfTrips', 'Interaction_Efficiency']
for col in numeric_data_types:
    if col in Xtrain.columns:
        Xtrain[col] = Xtrain[col].astype(float)
    if col in Xval.columns:
        Xval[col] = Xval[col].astype(float)

# =====================================================================
# 2. DEFINE PIPELINE ARCHITECTURE & VARIABLE ROUTING
# =====================================================================
age_order = [["0-18", "19-25", "26-35", "36-50", "51+"]]
numeric_features = ['DurationOfPitch', 'MonthlyIncome', 'NumberOfFollowups', 'NumberOfTrips', 'Interaction_Efficiency']
nominal_features = ['TypeofContact', 'ProductPitched', 'Occupation', 'Life_Stage']
ordinal_features = ['Age_bucket']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('nom', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_features),
        ('ord', OrdinalEncoder(
            categories=age_order, 
            handle_unknown='use_encoded_value',
            unknown_value=-1
         ), ordinal_features)
    ],
    remainder='passthrough'  # Keeps Passport, Total_Group_Size, Is_Young_Executive, Pitch_Alignment intact
)

# Handle class imbalance based on your target distribution
class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# Base XGBoost model matching your architectural constraints
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric='logloss')

# Construct the uniform model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgbclassifier', xgb_model)
])

# =====================================================================
# 3. DEFINE EXPERIMENT TUNING GRID
# =====================================================================
param_grid = { "xgbclassifier__n_estimators": [50, 75, 80,100], "xgbclassifier__max_depth": [2,3,4,5,6,7,8], "xgbclassifier__learning_rate": [0.05, 0.08,0.1,0.15]}

# Configure Local MLflow Repository Space
# if "MLFLOW_TRACKING_URI" in os.environ and os.environ["MLFLOW_TRACKING_URI"].strip():
#     tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
#     print(f"📡 Routing parameter matrix over internet to active Ngrok destination: {tracking_uri}")
#     mlflow.set_tracking_uri(tracking_uri)
# else:
#     print("⚠️ Ngrok URL missing. Logging locally inside repository context: ./mlruns")
#     mlflow.set_tracking_uri("file:./mlruns")
tracking_uri = os.getenv["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Tourism_Package_XGBoost_V4")

# =====================================================================
# 4. HYPERPARAMETER TUNING & MANUAL MLFLOW NESTED LOGGING
# =====================================================================

with mlflow.start_run() as parent_run:
    print("Executing hyperparameter space grid optimization...")
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Artifact URI:", mlflow.get_artifact_uri())
    # Grid Search using 5-Fold cross-validation matching your coding style
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Extract results array to iterate over manually
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

       
        clean_params = {k.replace("xgbclassifier__", ""): v for k, v in param_set.items()}

        # Log each combination cleanly as a separate child/nested MLflow run
        with mlflow.start_run(nested=True, run_name=f"Iteration_{i}"):
            mlflow.log_params(clean_params)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log optimal model parameters to the root parent execution
    clean_best_params = {k.replace("xgbclassifier__", ""): v for k, v in grid_search.best_params_.items()}
    mlflow.log_params(clean_best_params)

    # Store and isolate the absolute best performing configuration model
    best_model = grid_search.best_estimator_

    # Custom inference probability thresholding (0.45)
    classification_threshold = 0.45

    # Probability extraction and threshold assignment for the Train partition
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    # Probability extraction and threshold assignment for the Validation partition
    y_pred_val_proba = best_model.predict_proba(Xval)[:, 1]
    y_pred_val = (y_pred_val_proba >= classification_threshold).astype(int)

    # Compute dictionary-based metric matrices
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    val_report = classification_report(yval, y_pred_val, output_dict=True)

    # Log comprehensive system performance parameters to the root path
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "val_accuracy": val_report["accuracy"],
        "val_precision": val_report["1"]["precision"],
        "val_recall": val_report["1"]["recall"],
        "val_f1-score": val_report["1"]["f1-score"]
    })

    print("\n--- Model Training & Manual Nested Logging Completed Successfully ---")
    print(f"Optimal Hyperparameters: {clean_best_params}")
    print(f"Final Validation F1-Score (at 0.45 threshold): {val_report['1']['f1-score']:.4f}")
    
    # Save model and log artifact safely
    model_dir = "tourism_project/deployment"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model_tourism_package_prediction_v1.joblib")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
