import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime

# --- Load Data ---
try:
    df = pd.read_csv('processed_data/mlb_modeling_dataset.csv')
except FileNotFoundError:
    print("ERROR: 'mlb_modeling_dataset.csv' not found. Please run the creation script first.")
    exit()

print("Dataset loaded successfully.")

# --- Create Binary Target & Features ---
df['HomeTeamWon'] = (df['RunDifferential'] > 0).astype(int)
df['off_wOBA_diff'] = df['off_wOBA_home'] - df['off_wOBA_away']
df['off_K%_diff'] = df['off_K%_home'] - df['off_K%_away']
df['off_BB%_diff'] = df['off_BB%_home'] - df['off_BB%_away']
df['off_Barrel%_diff'] = df['off_Barrel%_home'] - df['off_Barrel%_away']
df['pch_FIP_diff'] = df['pch_FIP_home'] - df['pch_FIP_away']
df['pch_K-BB%_diff'] = df['pch_K-BB%_home'] - df['pch_K-BB%_away']
df['pch_Barrel%_diff'] = df['pch_Barrel%_home'] - df['pch_Barrel%_away']
df['matchup_woba_fip_home'] = df['off_wOBA_home'] - df['pch_FIP_away']
df['matchup_woba_fip_away'] = df['off_wOBA_away'] - df['pch_FIP_home']
print("Feature engineering complete.")

# --- Define Features (X) and Target (y) ---
features = [col for col in df.columns if col.startswith(('off_', 'pch_', 'matchup_'))]
X = df[features]
y = df['HomeTeamWon']

# --- Time-Based Train/Predict Split ---
# Convert game_date string to datetime object for comparison
df['game_date'] = pd.to_datetime(df['game_date'])
today = datetime.strptime('2025-07-30', '%Y-%m-%d')

train_df = df[df['game_date'] < today]
prediction_df = df[df['game_date'] == today].copy() # Use .copy() to avoid SettingWithCopyWarning

if prediction_df.empty:
    print(f"\nNo games found for today ({today.strftime('%Y-%m-%d')}). Exiting.")
    exit()

X_train = train_df[features]
y_train = train_df['HomeTeamWon']
X_predict = prediction_df[features]

print(f"\nTraining on {len(X_train)} games before {today.strftime('%Y-%m-%d')}.")
print(f"Predicting on {len(X_predict)} games scheduled for today.")

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_predict_scaled = scaler.transform(X_predict)

# --- Train a Calibrated Model ---
print("\nTraining calibrated XGBoost classifier...")
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- Make Predictions for Today's Games ---
print("Generating predictions for today's games...")
todays_probs = calibrated_model.predict_proba(X_predict_scaled)[:, 1]

# --- Format and Display Predictions ---
predictions_output = prediction_df[['HomeTeam_Name', 'AwayTeam_Name','Home Opener Odds', 'Away Opener']].copy()
predictions_output['Home_Win_Probability'] = [f"{prob:.1%}" for prob in todays_probs]

print("\n--- MLB Predictions for 2025-07-30 ---")
print(predictions_output.to_string(index=False))