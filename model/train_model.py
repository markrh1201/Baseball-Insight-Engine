import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
import re
import sys # Used for exiting gracefully
import matplotlib.pyplot as plt # Import for plotting

def rename_columns_for_modeling(df):
    """
    Parses column names like 'wOBA_home_hitting' and renames them to 'off_wOBA_home'.
    This version uses a corrected regular expression to handle the STAT_TEAM_CATEGORY format
    and sanitizes special characters from stat names.
    """
    # Regex to capture: (1: Stat Name)_(2: Team Type)_(3: Category)
    # Example: wOBA_home_hitting -> ('wOBA', 'home', 'hitting')
    pattern = re.compile(r"(.+)_(home|away)_(hitting|pitching)$")

    category_map = {
        'hitting': 'off_',
        'pitching': 'pch_'
    }

    new_cols = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            stat_name, team_type, category = match.groups()

            # Sanitize the stat name: replace all special chars with '_'
            sanitized_stat = stat_name.replace('%', '_Pct').replace('/', '_').replace('-', '_')

            # Assemble the new, clean column name
            new_prefix = category_map[category]
            new_col_name = f"{new_prefix}{sanitized_stat}_{team_type}"
            new_cols[col] = new_col_name

    # Rename all columns at once
    df.rename(columns=new_cols, inplace=True)

    # Handle the non-stat columns separately for clarity
    df.rename(columns={
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'home_team_won': 'HomeTeamWon',
    }, inplace=True)

    return df

def create_features(df):
    """Applies feature engineering using the new, sanitized column names."""
    try:
        df['off_wOBA_diff'] = df['off_wOBA_home'] - df['off_wOBA_away']
        df['off_K_Pct_diff'] = df['off_K_Pct_home'] - df['off_K_Pct_away']
        df['off_BB_Pct_diff'] = df['off_BB_Pct_home'] - df['off_BB_Pct_away']
        df['off_Barrel_Pct_diff'] = df['off_Barrel_Pct_home'] - df['off_Barrel_Pct_away']
        df['pch_FIP_diff'] = df['pch_FIP_home'] - df['pch_FIP_away']
        df['pch_K_BB_Pct_diff'] = df['pch_K_BB_Pct_home'] - df['pch_K_BB_Pct_away']
        df['pch_Barrel_Pct_diff'] = df['pch_Barrel_Pct_home'] - df['pch_Barrel_Pct_away']
        df['matchup_woba_fip_home'] = df['off_wOBA_home'] - df['pch_FIP_away']
        df['matchup_woba_fip_away'] = df['off_wOBA_away'] - df['pch_FIP_home']
    except KeyError as e:
        print(f"--- FATAL ERROR during feature creation ---")
        print(f"A required column is missing: {e}")
        print("This likely means the column renaming failed. Please check the column names in your CSV files.")
        # Optional: Print all available columns for debugging
        # print("\nAvailable columns after renaming:")
        # print(sorted(df.columns))
        sys.exit(1) # Exit the script
    return df

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Load Pre-Split Data ---
    try:
        train_df = pd.read_csv('modeling_data/training_dataset.csv')
        predict_df = pd.read_csv('modeling_data/testing_dataset.csv')
    except FileNotFoundError as e:
        print(f"ERROR: Could not find data files in 'modeling_data/' folder. {e}")
        print("Please run the data creation script first.")
        sys.exit(1)

    print("Training and testing datasets loaded successfully.")

    # Strip whitespace from column headers
    train_df.columns = train_df.columns.str.strip()
    predict_df.columns = predict_df.columns.str.strip()
    print("Column headers cleaned.")

    if predict_df.empty:
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"\nNo games found for today ({today_str}) in the testing dataset. Exiting.")
        sys.exit(0)

    # --- 2. Rename and Feature Engineer ---
    train_df = rename_columns_for_modeling(train_df)
    predict_df = rename_columns_for_modeling(predict_df)
    print("Column names have been standardized.")

    train_df = create_features(train_df)
    predict_df = create_features(predict_df)
    print("Feature engineering complete for both datasets.")

    # --- 3. Define Features (X) and Target (y) ---
    features = [col for col in train_df.columns if col.startswith(('off_', 'pch_', 'matchup_'))]
    X_train = train_df[features]
    y_train = train_df['HomeTeamWon']
    X_predict = predict_df[features]

    print(f"\nTraining on {len(X_train)} historical games.")
    print(f"Predicting on {len(X_predict)} games scheduled for today.")

    # --- 4. Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    # --- 5. Train a Calibrated Model ---
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
    
    # --- 6. Make and Display Predictions ---
    print("Generating predictions for today's games...")
    todays_probs = calibrated_model.predict_proba(X_predict_scaled)[:, 1]

    predictions_output = predict_df[['HomeTeam', 'AwayTeam', 'Home Opener Odds', 'Away Opener Odds']].copy()
    predictions_output['Home_Win_Probability'] = [f"{prob:.1%}" for prob in todays_probs]

    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n--- MLB Predictions for {today_str} ---")
    print(predictions_output.to_string(index=False))
    
    # --- 7. Plot Feature Importance ---
    print("\nDisplaying feature importance graph...")
    
    # Extract the first base estimator from the calibrated model
    # (assuming cv=3, there are 3 base models trained)
    # CORRECT
    xgb_model = calibrated_model.calibrated_classifiers_[0].estimator
    
    # Create a pandas Series for easy plotting
    feature_importances = pd.Series(
        data=xgb_model.feature_importances_,
        index=features # The list of feature names
    ).sort_values(ascending=True)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances.index, feature_importances.values) # type: ignore
    plt.title('XGBoost Feature Importance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout() # Adjust layout to make room for feature names
    plt.show()