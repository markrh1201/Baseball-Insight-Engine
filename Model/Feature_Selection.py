import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('data/2024_Training_Data.csv')
    X = data.drop(columns=['Home_Win'])
    y = data['Home_Win']
    
    X_new, selected_features = select_features(X, y)
    print("Selected features:", X.columns[selected_features])
