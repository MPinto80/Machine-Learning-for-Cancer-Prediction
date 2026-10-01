from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os

def train_model(x_train_selected, y_train, max_features):
    print('Training Random Forest Classifier...')
    rf = OneVsRestClassifier(RandomForestClassifier(max_features=max_features))
    rf.fit(x_train_selected, y_train)
    return rf

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f'Model saved to {filepath}')

def load_model(filepath):
    if os.path.exists(filepath):
        print(f'Loading model from {filepath}...')
        return joblib.load(filepath)
    return None
