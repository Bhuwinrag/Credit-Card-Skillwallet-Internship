import pickle
import numpy as np

def test_predictions():
    # Load model
    model = pickle.load(open('model.pkl', 'rb'))
    
    # 20 features:
    # CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, AMT_INCOME_TOTAL, NAME_INCOME_TYPE, 
    # NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, DAYS_BIRTH, DAYS_EMPLOYED, 
    # FLAG_MOBIL, FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL, CNT_FAM_MEMBERS, FAMILY_DEPENDENCY, 
    # open_month, end_month, window

    # 1. Prime Applicant: High Income ($150,000), 12 yrs employed, owns car/realty, clean recent credit window
    prime_features = [1, 1, 1, 0, 150000, 0, 0, 0, 0, 40*365, 12*365, 1, 1, 1, 1, 2, 2, -6, 0, 6]
    
    # 2. Risky Applicant: Low Income ($22,000), 0.8 yrs employed, 7 dependents, rented, old delinquent window
    risky_features = [1, 0, 0, 3, 22000, 2, 3, 4, 3, 24*365, 0.8*365, 1, 0, 0, 0, 4, 7, -33, 0, 33]

    prime_pred = model.predict([np.array(prime_features)])[0]
    prime_prob = model.predict_proba([np.array(prime_features)])[0]

    risky_pred = model.predict([np.array(risky_features)])[0]
    risky_prob = model.predict_proba([np.array(risky_features)])[0]

    print(f"Prime Applicant Prediction: {prime_pred} (Approval Probability: {prime_prob[1]*100:.2f}%)")
    print(f"Risky Applicant Prediction: {risky_pred} (Rejection Probability: {risky_prob[0]*100:.2f}%)")

if __name__ == '__main__':
    test_predictions()
