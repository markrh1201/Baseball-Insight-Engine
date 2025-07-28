import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to remove columns with all missing values
class DropAllMissingColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_to_drop_ = [col for col in X.columns if X[col].isnull().all()]
        return self
    
    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop_)

# Load the datasets
training_df = pd.read_csv('Training.csv')
X_test = pd.read_csv('Testing.csv')
team_names = pd.read_csv('Team_ID_Map.csv')

# Separate features and target
X_train = training_df.drop(columns='Home_Win')
y_train = training_df['Home_Win']

# Define the numeric transformer pipeline
numeric_transformer = Pipeline(steps=[
    ('drop_missing_cols', DropAllMissingColumns()),  # Custom step to drop all missing columns
    ('imputer', SimpleImputer(strategy='median'))  # Impute remaining missing values with median
])

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X_train.columns)
    ]
)

# Build the full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=100)),
    ('xgb', XGBClassifier(eval_metric='logloss', scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])))
])

param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 6, 9],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Predict using the best found parameters
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)

# Merge the test data with team names
X_test = X_test.merge(team_names, left_on='Home_teamIDfg', right_on='teamIDfg', how='left').rename(columns={'Team': 'Home_Team'})
X_test = X_test.merge(team_names, left_on='Away_teamIDfg', right_on='teamIDfg', how='left').rename(columns={'Team': 'Away_Team'})

# Create a DataFrame with the predictions and teams
results_df = pd.DataFrame({
    'Home_Team': X_test['Home_Team'],
    'Away_Team': X_test['Away_Team'],
    'Home_Win_Prediction': y_pred
})

# Add a column to indicate the winning team
results_df['Winning_Team'] = results_df.apply(
    lambda row: row['Home_Team'] if row['Home_Win_Prediction'] == 1 else row['Away_Team'],
    axis=1
)

# Save the results to a CSV file
results_df.to_csv('Predicted_Matchups.csv', index=False)

print("Predictions saved to Predicted_Matchups.csv successfully!")
print(results_df.head())
