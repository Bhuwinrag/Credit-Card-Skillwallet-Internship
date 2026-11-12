import pandas as pd
import pickle

def find_inputs():
    df = pd.read_csv('processed_dataset.csv')
    model = pickle.load(open('model.pkl', 'rb'))
    
    # Get a row where the model predicts 1 (Approved)
    X = df.drop(['ID', 'STATUS_BIN'], axis=1)
    preds = model.predict(X)
    
    # Find an approved one
    approved_idx = (preds == 1).argmax()
    rejected_idx = (preds == 0).argmax()
    
    print("--- APPROVED APPLICANT INPUTS ---")
    print(X.iloc[approved_idx])
    
    print("\n--- REJECTED APPLICANT INPUTS ---")
    print(X.iloc[rejected_idx])

if __name__ == '__main__':
    find_inputs()
