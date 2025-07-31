import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
import re
import sys
import matplotlib.pyplot as plt
import numpy as np # Imported for odds calculations

# --- Helper Functions for Betting Calculations ---

def convert_american_to_decimal(american_odds):
    """Converts American odds to decimal odds."""
    if american_odds >= 100:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def calculate_implied_probability(american_odds):
    """Converts American odds to implied probability."""
    if american_odds >= 100:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def rename_columns_for_modeling(df):
    """
    Parses column names like 'wOBA_home_hitting' and renames them to 'off_wOBA_home'.
    This version uses a corrected regular expression to handle the STAT_TEAM_CATEGORY format
    and sanitizes special characters from stat names.
    """
    # Regex to capture: (1: Stat Name)_(2: Team Type)_(3: Category)
    pattern = re.compile(r"(.+)_(home|away)_(hitting|pitching)$")
    category_map = {'hitting': 'off_', 'pitching': 'pch_'}
    
    new_cols = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            stat_name, team_type, category = match.groups()
            sanitized_stat = stat_name.replace('%', '_Pct').replace('/', '_').replace('-', '_')
            new_prefix = category_map[category]
            new_col_name = f"{new_prefix}{sanitized_stat}_{team_type}"
            new_cols[col] = new_col_name

    df.rename(columns=new_cols, inplace=True)
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
        print(f"--- FATAL ERROR during feature creation: A required column is missing: {e} ---")
        sys.exit(1)
    return df

def generate_betting_card(predictions_df, kelly_fraction=0.25):
    """
    Analyzes model predictions against betting odds to find value bets.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing teams, odds, and model probabilities.
        kelly_fraction (float): The fraction of the Kelly Criterion to use for stake sizing (e.g., 0.25 for quarter Kelly).
                                This is a risk-management strategy to be more conservative.

    Returns:
        pd.DataFrame: A DataFrame formatted as a "betting card" showing only +EV bets.
    """
    betting_opportunities = []

    for index, row in predictions_df.iterrows():
        # --- Home Team Bet ---
        home_prob = row['Home_Win_Probability']
        home_odds = row['Home Opener Odds']
        home_decimal_odds = convert_american_to_decimal(home_odds)
        home_edge = (home_prob * home_decimal_odds) - 1

        if home_edge > 0:
            kelly_stake = (home_edge / (home_decimal_odds - 1)) * kelly_fraction
            betting_opportunities.append({
                'Team': row['HomeTeam'],
                'Opponent': row['AwayTeam'],
                'Bet Type': 'Moneyline',
                'Odds (American)': f"{home_odds:+.0f}",
                'Model Probability': f"{home_prob:.1%}",
                'Implied Probability': f"{calculate_implied_probability(home_odds):.1%}",
                'Edge (+EV)': f"{home_edge:+.2%}",
                'Kelly Stake': f"{kelly_stake:.2%}",
            })

        # --- Away Team Bet ---
        away_prob = 1 - home_prob # Probability for the away team is 1 minus home team prob
        away_odds = row['Away Opener Odds']
        away_decimal_odds = convert_american_to_decimal(away_odds)
        away_edge = (away_prob * away_decimal_odds) - 1

        if away_edge > 0:
            kelly_stake = (away_edge / (away_decimal_odds - 1)) * kelly_fraction
            betting_opportunities.append({
                'Team': row['AwayTeam'],
                'Opponent': row['HomeTeam'],
                'Bet Type': 'Moneyline',
                'Odds (American)': f"{away_odds:+.0f}",
                'Model Probability': f"{away_prob:.1%}",
                'Implied Probability': f"{calculate_implied_probability(away_odds):.1%}",
                'Edge (+EV)': f"{away_edge:+.2%}",
                'Kelly Stake': f"{kelly_stake:.2%}",
            })

    if not betting_opportunities:
        return pd.DataFrame() # Return empty dataframe if no value bets

    # Create and sort the final betting card DataFrame
    betting_card = pd.DataFrame(betting_opportunities)
    return betting_card.sort_values(by='Edge (+EV)', ascending=False)

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Load Data ---
    try:
        train_df = pd.read_csv('modeling_data/training_dataset.csv')
        predict_df = pd.read_csv('modeling_data/testing_dataset.csv')
    except FileNotFoundError as e:
        print(f"ERROR: Could not find data files. {e}")
        sys.exit(1)

    train_df.columns = train_df.columns.str.strip()
    predict_df.columns = predict_df.columns.str.strip()

    if predict_df.empty:
        print(f"\nNo games found in the testing dataset. Exiting.")
        sys.exit(0)

    # --- 2. Rename, Feature Engineer & Define Features ---
    train_df = rename_columns_for_modeling(train_df)
    predict_df = rename_columns_for_modeling(predict_df)
    train_df = create_features(train_df)
    predict_df = create_features(predict_df)
    
    features = [col for col in train_df.columns if col.startswith(('off_', 'pch_', 'matchup_'))]
    X_train = train_df[features]
    y_train = train_df['HomeTeamWon']
    X_predict = predict_df[features]

    # --- 3. Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    # --- 4. Train a Calibrated Model ---
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
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5) # Using 5-fold CV
    calibrated_model.fit(X_train_scaled, y_train)
    print("Model training complete.")
    
    # --- 5. Generate Predictions and Identify Value Bets ---
    print("Generating predictions and identifying value bets...")
    todays_probs = calibrated_model.predict_proba(X_predict_scaled)[:, 1]

    # Create a clean DataFrame for analysis
    predictions_output = predict_df[['HomeTeam', 'AwayTeam', 'Home Opener Odds', 'Away Opener Odds']].copy()
    predictions_output['Home_Win_Probability'] = todays_probs

    # Generate the actionable betting card
    betting_card = generate_betting_card(predictions_output, kelly_fraction=0.25)
    
    # --- 6. Display Actionable Betting Card ---
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n--- MLB Value Betting Card for {today_str} ---")
    print("Based on model-identified edge against market odds.")
    print("Stake is recommended as a percentage of your total bankroll (Quarter Kelly).")

    if not betting_card.empty:
        print(betting_card.to_string(index=False))
    else:
        print("\nNo value bets identified for today's games. It's wise to pass.")

    # --- 7. Plot Feature Importance ---
    print("\nDisplaying feature importance graph...")
    # To get a more robust measure, we can average the feature importances
    # across all models trained during cross-validation in CalibratedClassifierCV.
    importances = [clf.estimator.feature_importances_ for clf in calibrated_model.calibrated_classifiers_]
    avg_importances = np.mean(importances, axis=0)

    feature_importances = pd.Series(
        data=avg_importances,
        index=features
    ).sort_values(ascending=True)
    
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importances.index, feature_importances.values)
    plt.title('Average XGBoost Feature Importance (from Calibrated CV)', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()