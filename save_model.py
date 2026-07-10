import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def save_model():
    print("Training final model for deployment...")
    df = pd.read_csv('processed_dataset.csv')
    
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        
    X = df.drop('STATUS_BIN', axis=1)
    y = df['STATUS_BIN']
    
    # Train the Random Forest Model as it had the best accuracy (87%)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
        
    print("Model saved successfully as 'model.pkl'.")

if __name__ == "__main__":
    save_model()
