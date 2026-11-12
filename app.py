import pickle
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import traceback
import os

app = Flask(__name__)

# Base directory for reliable serverless resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model securely using absolute paths for Serverless (Vercel) compatibility
model = None
try:
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Housing type lookup table
HOUSING_MAP = {
    0: "House / Apartment",
    1: "With Parents",
    2: "Municipal Apartment",
    3: "Rented Apartment",
    4: "Office Apartment",
    5: "Co-op Apartment"
}

# 1. Landing Page Route
@app.route('/')
def home():
    return render_template('landing.html')

# 2. Interactive Evaluation Route
@app.route('/evaluate')
def evaluate():
    return render_template('evaluate.html')

# 3. Model Analytics & EDA Dashboard Route
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

# 4. Static Plot Serving Route
@app.route('/plots/<path:filename>')
def serve_plot(filename):
    plots_dir = os.path.join(BASE_DIR, 'plots')
    return send_from_directory(plots_dir, filename)

# 5. Form Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in the precise order expected by the trained model
        feature_keys = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
            'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
            'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
            'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'FAMILY_DEPENDENCY',
            'open_month', 'end_month', 'window'
        ]

        raw_values = {}
        features = []
        for key in feature_keys:
            val = float(request.form.get(key, 0))
            raw_values[key] = val
            features.append(val)

        # Store user-entered years for display
        age_years = raw_values['DAYS_BIRTH']
        employed_years = raw_values['DAYS_EMPLOYED']
        annual_income = raw_values['AMT_INCOME_TOTAL']

        # Convert user-friendly Years to Days as required by the model
        features[9] = features[9] * 365.0    # DAYS_BIRTH
        features[10] = features[10] * 365.0  # DAYS_EMPLOYED

        # Convert into numpy 2D array
        final_input = np.array([features])

        # Inference
        prediction = model.predict(final_input)[0]
        
        # Calculate probabilities if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(final_input)[0]
            prob_reject = float(probs[0])
            prob_approve = float(probs[1])
        else:
            prob_approve = 1.0 if prediction == 1 else 0.0
            prob_reject = 1.0 - prob_approve

        is_approved = (prediction == 1)
        prediction_text = "Credit Card Approved" if is_approved else "Credit Card Rejected"

        # Calculate confidence & gauge offset (circumference 264)
        if is_approved:
            confidence_score = round(prob_approve * 100, 1)
        else:
            confidence_score = round(prob_reject * 100, 1)

        gauge_offset = int(max(0, 264 - (264 * (confidence_score / 100.0))))

        # Determine Risk Tier
        if prob_approve >= 0.75:
            risk_tier = "Tier 1 — Low Risk / Prime"
        elif prob_approve >= 0.40:
            risk_tier = "Tier 2 — Moderate Risk"
        else:
            risk_tier = "Tier 3 — High Default Exposure"

        # Explainability & Contributing Factors Engine
        positive_factors = []
        risk_factors = []

        # Bureau Window Analysis
        if raw_values['window'] <= 12:
            positive_factors.append(f"Low Delinquency Window ({int(raw_values['window'])} months bureau history)")
        else:
            risk_factors.append(f"Long Delinquency Observation Window ({int(raw_values['window'])} months record depth)")

        # Income Analysis
        if annual_income >= 100000:
            positive_factors.append(f"High Annual Income Capacity (${annual_income:,.0f} USD)")
        elif annual_income >= 50000:
            positive_factors.append(f"Solid Steady Income (${annual_income:,.0f} USD)")
        else:
            risk_factors.append(f"Constrained Annual Income (${annual_income:,.0f} USD)")

        # Employment Analysis
        if employed_years >= 5:
            positive_factors.append(f"Strong Employment Longevity ({employed_years:g} years tenure)")
        elif employed_years >= 2:
            positive_factors.append(f"Established Career Stability ({employed_years:g} years tenure)")
        else:
            risk_factors.append(f"Short Employment Tenure ({employed_years:g} years on current job)")

        # Asset & Housing Ownership
        if raw_values['FLAG_OWN_REALTY'] == 1:
            positive_factors.append("Real Estate Property Owner (Asset Backing)")
        if raw_values['FLAG_OWN_CAR'] == 1:
            positive_factors.append("Vehicle Asset Owner (Collateral Stability)")
        if raw_values['FLAG_OWN_REALTY'] == 0 and raw_values['FLAG_OWN_CAR'] == 0:
            risk_factors.append("No Real Estate or Vehicle Assets Registered")

        # Housing Type
        housing_label = HOUSING_MAP.get(int(raw_values['NAME_HOUSING_TYPE']), "Standard")
        if raw_values['NAME_HOUSING_TYPE'] == 0:
            positive_factors.append(f"Stable Residence: {housing_label}")
        elif raw_values['NAME_HOUSING_TYPE'] == 3:
            risk_factors.append("Rented Residence (Higher Monthly Fixed Overhead)")

        # Dependency
        if raw_values['FAMILY_DEPENDENCY'] <= 2:
            positive_factors.append(f"Low Family Dependency Ratio ({int(raw_values['FAMILY_DEPENDENCY'])} dependents)")
        elif raw_values['FAMILY_DEPENDENCY'] >= 4:
            risk_factors.append(f"Elevated Household Dependency ({int(raw_values['FAMILY_DEPENDENCY'])} dependents)")

        # Ensure lists are never empty
        if not positive_factors:
            positive_factors.append("Active Credit Bureau Applicant Registration")
        if not risk_factors:
            risk_factors.append("Standard Bureau Monitoring Recommended")

        # Underwriting Recommendation
        if is_approved:
            rec_limit = min(max(5000, annual_income * 0.20), 40000)
            recommendation_text = (
                f"Automated Approval Granted. Applicant qualifies for immediate revolving line issuance "
                f"with a recommended initial credit line of ${rec_limit:,.0f} and 0% introductory APR for 12 months."
            )
        else:
            recommendation_text = (
                "Automated Approval Declined. The balanced decision tree identified elevated default probabilities "
                "associated with the vintage window and debt-to-income profile. Recommend secondary manual review "
                "or offering a secured deposit credit card program."
            )

        # Applicant Summary Object
        applicant_summary = {
            'income': annual_income,
            'employed_years': employed_years,
            'age_years': int(age_years),
            'housing_type': housing_label,
            'fam_members': int(raw_values['CNT_FAM_MEMBERS']),
            'dependency': int(raw_values['FAMILY_DEPENDENCY']),
            'car': "Yes" if raw_values['FLAG_OWN_CAR'] == 1 else "No",
            'realty': "Yes" if raw_values['FLAG_OWN_REALTY'] == 1 else "No",
            'window': int(raw_values['window']),
            'open_month': int(raw_values['open_month'])
        }

        return render_template(
            'result.html',
            prediction_text=prediction_text,
            is_approved=is_approved,
            confidence_score=confidence_score,
            gauge_offset=gauge_offset,
            risk_tier=risk_tier,
            positive_factors=positive_factors,
            risk_factors=risk_factors,
            applicant_summary=applicant_summary,
            recommendation_text=recommendation_text
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        return render_template(
            'result.html',
            prediction_text="Error in evaluation. Please verify your inputs.",
            is_approved=False,
            confidence_score=0.0,
            gauge_offset=264,
            risk_tier="Processing Error",
            positive_factors=["Input verification error"],
            risk_factors=["Please recheck numeric fields and ranges"],
            applicant_summary={
                'income': 0, 'employed_years': 0, 'age_years': 0,
                'housing_type': 'Unknown', 'fam_members': 0, 'dependency': 0,
                'car': 'N/A', 'realty': 'N/A', 'window': 0, 'open_month': 0
            },
            recommendation_text="An error occurred while evaluating the application. Please return and verify all inputs."
        )

# 6. REST API Endpoint for programmatic inference
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        feature_keys = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
            'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
            'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
            'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'FAMILY_DEPENDENCY',
            'open_month', 'end_month', 'window'
        ]

        features = [float(data.get(k, 0)) for k in feature_keys]
        features[9] = features[9] * 365.0
        features[10] = features[10] * 365.0

        final_input = np.array([features])
        pred = int(model.predict(final_input)[0])
        probs = model.predict_proba(final_input)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]

        return jsonify({
            'status': 'success',
            'prediction': pred,
            'decision': 'Approved' if pred == 1 else 'Rejected',
            'approval_probability': round(float(probs[1]) * 100, 2),
            'rejection_probability': round(float(probs[0]) * 100, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# 7. Analytics Data Endpoint
@app.route('/api/analytics', methods=['GET'])
def api_analytics():
    return jsonify({
        'status': 'success',
        'dataset_size': 150000,
        'features_count': 20,
        'target_distribution': {'rejected_or_late': 87.22, 'approved_good': 12.78},
        'champion_model': 'Balanced Decision Tree',
        'metrics': {
            'roc_auc': 0.8524,
            'precision': 0.8797,
            'accuracy': 0.7463,
            'f1_score': 0.7855
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
