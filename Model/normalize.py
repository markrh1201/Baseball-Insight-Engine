import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_non_binary_features(X_train, X_test):
    
    # Identify binary columns
    binary_columns = [col for col in X_train.columns if len(X_train[col].unique()) == 2]
    
    # Identify non-binary columns
    non_binary_columns = [col for col in X_train.columns if col not in binary_columns and col != 'Home_Win']
    
    # Normalize the non-binary features
    scaler = StandardScaler()
    X_train[non_binary_columns] = scaler.fit_transform(X_train[non_binary_columns])
    X_test[non_binary_columns] = scaler.transform(X_test[non_binary_columns])

    print(f"Data Normalized!")
    
    return X_train, X_test
    

