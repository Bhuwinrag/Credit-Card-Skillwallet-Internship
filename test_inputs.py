import pandas as pd
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

    # Good Applicant
    # Male(1), Car Yes(1), Realty Yes(1), 0 children, 150000 income, Working(0), Higher edu(0),
    # Married(0), House(0), Age(40)*365, Emp(10)*365, Mobil(1), WorkPh(1), Ph(1), Email(1),
    # 2 fam, 2 dep, -24, 0, 24
    good_features = [1, 1, 1, 0, 150000, 0, 0, 0, 0, 40*365, 10*365, 1, 1, 1, 1, 2, 2, -24, 0, 24]
    
    # Bad Applicant
    # Female(0), Car No(0), Realty No(0), 3 children, 20000 income, Pensioner(2), Lower Sec(3),
    # Widow(4), Rented(3), Age(25)*365, Emp(1)*365, Mobil(1), WorkPh(0), Ph(0), Email(0),
    # 5 fam, 8 dep, -5, -2, 3
    bad_features = [0, 0, 0, 3, 20000, 2, 3, 4, 3, 25*365, 1*365, 1, 0, 0, 0, 5, 8, -5, -2, 3]

    good_pred = model.predict([np.array(good_features)])
    bad_pred = model.predict([np.array(bad_features)])

    print(f"Good Profile Prediction: {good_pred[0]}")
    print(f"Bad Profile Prediction: {bad_pred[0]}")

if __name__ == '__main__':
    test_predictions()
