
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="BankWise | Marketing Intelligence",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    return (
        joblib.load("balanced_logistic_regression.pkl"),
        joblib.load("decision_tree.pkl"),
        joblib.load("random_forest.pkl"),
        joblib.load("balanced_random_forest.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("model_columns.pkl")
    )

try:
    balanced_lr, decision_tree, random_forest, balanced_rf, scaler, model_columns = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# --------------------------------------------------
# CUSTOM HEADER
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }

    .main-header {
        background: linear-gradient(120deg, #12304A, #087EAA);
        padding: 30px 32px;
        border-radius: 15px;
        border-bottom: 5px solid #E6B84A;
        margin-bottom: 25px;
    }

    .brand {
        color: #B9E5F0;
        font-size: 13px;
        letter-spacing: 2px;
        font-weight: 600;
    }

    .main-header h1 {
        color: white;
        font-size: 32px;
        margin: 10px 0;
    }

    .main-header p {
        color: #E2EDF5;
        font-size: 16px;
        margin-bottom: 0;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 22px;
        border: 1px solid #D5E0E9;
        border-radius: 12px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background-color: #087EAA;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #075E80;
        color: white;
        border: none;
    }

    h2, h3 {
        color: #12304A;
    }
</style>

<div class="main-header">
    <div class="brand">BANKWISE · MARKETING INTELLIGENCE</div>
    <h1>Bank Marketing Prediction</h1>
    <p>
        Understand customer behaviour. Predict term-deposit
        subscription using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Enter customer details below to predict their likelihood "
    "of subscribing to a bank term deposit."
)

# --------------------------------------------------
# MODEL EVALUATION RESULTS
# --------------------------------------------------
st.subheader("Model Performance")

st.caption(
    "Evaluation on the held-out test set. "
    "Precision, recall and F1-score refer to the YES class."
)

results = pd.DataFrame({
    "Model": [
        "Balanced Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Balanced Random Forest"
    ],
    "Accuracy": [0.7557, 0.8435, 0.8800, 0.8416],
    "Precision": [0.2669, 0.3202, 0.4843, 0.3837],
    "Recall": [0.6229, 0.3006, 0.3932, 0.5832],
    "F1-score": [0.3737, 0.3101, 0.4340, 0.4629]
})

st.dataframe(
    results.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-score": "{:.2%}"
    }).highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1-score"],
        color="#D9F2E6"
    ),
    use_container_width=True,
    hide_index=True
)

st.info(
    "Best F1-score: Balanced Random Forest (46.29%). "
    "Highest accuracy: Random Forest (88.00%). "
    "Highest recall: Balanced Logistic Regression (62.29%)."
)

st.divider()


# --------------------------------------------------
# CUSTOMER INPUT FORM
# --------------------------------------------------
st.subheader("Customer Information")

with st.form("customer_form"):

    left, right = st.columns(2)

    with left:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35
        )

        job = st.selectbox(
            "Job",
            [
                "management", "technician", "entrepreneur",
                "blue-collar", "retired", "admin.", "services",
                "student", "self-employed", "unemployed",
                "housemaid", "unknown"
            ]
        )

        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )

        education = st.selectbox(
            "Education",
            ["secondary", "tertiary", "primary", "unknown"]
        )

        balance = st.number_input(
            "Account Balance (€)", value=1000
        )

        default = st.selectbox(
            "Credit in Default", ["no", "yes"]
        )

        housing = st.selectbox(
            "Housing Loan", ["yes", "no"]
        )

    with right:
        loan = st.selectbox(
            "Personal Loan", ["no", "yes"]
        )

        contact = st.selectbox(
            "Contact Method",
            ["cellular", "telephone", "unknown"]
        )

        day = st.number_input(
            "Last Contact Day",
            min_value=1, max_value=31, value=15
        )

        month = st.selectbox(
            "Last Contact Month",
            [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ]
        )

        campaign = st.number_input(
            "Contacts in Current Campaign",
            min_value=1, max_value=100, value=2
        )

        pdays = st.number_input(
            "Days Since Previous Campaign Contact (-1 if none)",
            min_value=-1, max_value=999, value=-1
        )

        previous = st.number_input(
            "Previous Campaign Contacts",
            min_value=0, value=0
        )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "success", "failure", "other"]
        )

    submitted = st.form_submit_button(
        "Predict Subscription",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------
if submitted:
    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    try:
        # Encode and align columns with training data
        customer = pd.get_dummies(customer, dtype=int)
        customer = customer.reindex(
            columns=model_columns,
            fill_value=0
        )

        # The scaler was fitted on the training features.
        # Use the same transformed inputs for all models.
        customer_scaled = scaler.transform(customer)

        predictions = {
            "Balanced Logistic Regression":
                balanced_lr.predict(customer_scaled)[0],
            "Decision Tree":
                decision_tree.predict(customer_scaled)[0],
            "Random Forest":
                random_forest.predict(customer_scaled)[0],
            "Balanced Random Forest":
                balanced_rf.predict(customer_scaled)[0]
        }

        st.divider()
        st.subheader("Prediction Results")
        st.caption(
            "These are model predictions, not guarantees of "
            "a customer's actual behaviour."
        )

        cols = st.columns(2)

        for i, (model_name, prediction) in enumerate(
            predictions.items()
        ):
            with cols[i % 2]:
                if prediction == "yes":
                    st.success(
                        f"**{model_name}**\n\n"
                        "Likely to subscribe: YES"
                    )
                else:
                    st.info(
                        f"**{model_name}**\n\n"
                        "Likely to subscribe: NO"
                    )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption(
            "Check that the saved feature columns, scaler, "
            "and models belong to the same training run."
        )
