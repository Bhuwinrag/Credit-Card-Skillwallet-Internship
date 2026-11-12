import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def save_model():
    print("Training production-grade balanced Random Forest model...")
    df = pd.read_csv('processed_dataset.csv')
    
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        
    X = df.drop('STATUS_BIN', axis=1)
    y = df['STATUS_BIN']
    
    # Train a Random Forest with balanced class weights so all 20 features are actively evaluated
    rf_model = RandomForestClassifier(
        n_estimators=40,
        max_depth=10,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
        
    print("Balanced Random Forest model saved successfully as 'model.pkl'.")
    print(f"Number of features: {len(rf_model.feature_names_in_)}")
    print("Top 5 features by importance:")
    for f, imp in sorted(zip(rf_model.feature_names_in_, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {f}: {imp*100:.2f}%")

if __name__ == "__main__":
    save_model()
