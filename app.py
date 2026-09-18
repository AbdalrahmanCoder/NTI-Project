from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# Keep the model files in the same folder as this app.
folder = Path(__file__).parent

with open(folder / "loan_data_model.pkl", "rb") as file:
    loan_data_model = pickle.load(file)

with open(folder / "loan_data_preprocessing.pkl", "rb") as file:
    loan_data_assets = pickle.load(file)

with open(folder / "loan_approval_model.pkl", "rb") as file:
    loan_approval_model = pickle.load(file)

with open(folder / "loan_approval_scaler.pkl", "rb") as file:
    loan_approval_scaler = pickle.load(file)


st.title("Loan Approval Prediction")
st.write("Try both models by entering the applicant information.")


def show_result(model, values):
    """Show a small result message for either model."""
    # The two saved models were fitted with different pandas settings.
    values = values.to_numpy()
    prediction = model.predict(values)[0]
    probability = model.predict_proba(values)[0][1]

    if prediction == 1:
        st.success("Loan Accepted")
    else:
        st.error("Loan Not Accepted")

    st.write(f"Estimated approval chance: {probability:.1%}")


# The saved loan_data_model.pkl also expects the 22 one-hot columns.
def prepare_loan_data(age, gender, education, income, work_years, home,
                      amount, intent, interest, income_percent,
                      credit_history, credit_score, previous_default):
    return prepare_approval_data(
        age, income, work_years, amount, interest, income_percent,
        credit_history, credit_score, gender, education, home, intent,
        previous_default
    )


def prepare_approval_data(age, income, work_years, amount, interest,
                          income_percent, credit_history, credit_score,
                          gender, education, home, intent, previous_default):
    # This model uses one-hot columns, like processed_loan_approval.csv.
    values = pd.DataFrame({
        "person_age": [age],
        "person_income": [income],
        "person_emp_exp": [work_years],
        "loan_amnt": [amount],
        "loan_int_rate": [interest],
        "loan_percent_income": [income_percent],
        "cb_person_cred_hist_length": [credit_history],
        "credit_score": [credit_score],
        "person_gender_male": [int(gender == "male")],
        "person_education_Bachelor": [int(education == "Bachelor")],
        "person_education_Doctorate": [int(education == "Doctorate")],
        "person_education_High School": [int(education == "High School")],
        "person_education_Master": [int(education == "Master")],
        "person_home_ownership_OTHER": [int(home == "OTHER")],
        "person_home_ownership_OWN": [int(home == "OWN")],
        "person_home_ownership_RENT": [int(home == "RENT")],
        "loan_intent_EDUCATION": [int(intent == "EDUCATION")],
        "loan_intent_HOMEIMPROVEMENT": [int(intent == "HOMEIMPROVEMENT")],
        "loan_intent_MEDICAL": [int(intent == "MEDICAL")],
        "loan_intent_PERSONAL": [int(intent == "PERSONAL")],
        "loan_intent_VENTURE": [int(intent == "VENTURE")],
        "previous_loan_defaults_on_file_Yes": [int(previous_default == "Yes")]
    })

    return pd.DataFrame(
        loan_approval_scaler.transform(values),
        columns=values.columns
    )


tab1, tab2 = st.tabs(["Loan Data Model", "Loan Approval Model"])

with tab1:
    st.header("Loan Data Model")
    st.write("This uses the model trained from loan_data.csv.")

    with st.form("loan_data_form"):
        age = st.number_input("Age", 18, 100, 25, key="data_age")
        gender = st.selectbox("Gender", ["female", "male"], key="data_gender")
        education = st.selectbox(
            "Education", ["High School", "Bachelor", "Master", "Doctorate"],
            key="data_education"
        )
        income = st.number_input("Annual Income", 0.0, value=50000.0,
                                 key="data_income")
        work_years = st.number_input("Employment Experience", 0, 50, 2,
                                     key="data_work")
        home = st.selectbox("Home Ownership",
                            ["RENT", "OWN", "MORTGAGE", "OTHER"],
                            key="data_home")
        amount = st.number_input("Loan Amount", 0.0, value=10000.0,
                                 key="data_amount")
        intent = st.selectbox(
            "Loan Intent",
            ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL",
             "VENTURE", "DEBTCONSOLIDATION"],
            key="data_intent"
        )
        interest = st.number_input("Interest Rate (%)", 0.0, 50.0, 10.0,
                                   key="data_interest")
        income_percent = st.number_input("Loan Percent of Income", 0.0, 1.0,
                                         0.2, key="data_percent")
        credit_history = st.number_input("Credit History Length", 0.0, 50.0,
                                         3.0, key="data_history")
        credit_score = st.number_input("Credit Score", 300, 850, 650,
                                       key="data_score")
        previous_default = st.selectbox("Previous Loan Default", ["No", "Yes"],
                                       key="data_default")

        accept_data = st.form_submit_button("Accept Loan Data Application")

    if accept_data:
        input_values = prepare_loan_data(
            age, gender, education, income, work_years, home, amount, intent,
            interest, income_percent, credit_history, credit_score,
            previous_default
        )
        show_result(loan_data_model, input_values)

with tab2:
    st.header("Loan Approval Model")
    st.write("This uses the one-hot encoded loan approval dataset.")

    with st.form("loan_approval_form"):
        approval_age = st.number_input("Age", 18, 100, 25, key="approval_age")
        approval_income = st.number_input("Annual Income", 0.0, value=50000.0,
                                          key="approval_income")
        approval_work = st.number_input("Employment Experience", 0, 50, 2,
                                        key="approval_work")
        approval_amount = st.number_input("Loan Amount", 0.0, value=10000.0,
                                          key="approval_amount")
        approval_interest = st.number_input("Interest Rate (%)", 0.0, 50.0,
                                            10.0, key="approval_interest")
        approval_percent = st.number_input("Loan Percent of Income", 0.0, 1.0,
                                           0.2, key="approval_percent")
        approval_history = st.number_input("Credit History Length", 0.0, 50.0,
                                           3.0, key="approval_history")
        approval_score = st.number_input("Credit Score", 300, 850, 650,
                                         key="approval_score")
        approval_gender = st.selectbox("Gender", ["female", "male"],
                                       key="approval_gender")
        approval_education = st.selectbox(
            "Education", ["High School", "Bachelor", "Master", "Doctorate"],
            key="approval_education"
        )
        approval_home = st.selectbox("Home Ownership",
                                     ["RENT", "OWN", "MORTGAGE", "OTHER"],
                                     key="approval_home")
        approval_intent = st.selectbox(
            "Loan Intent",
            ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL",
             "VENTURE"],
            key="approval_intent"
        )
        approval_default = st.selectbox("Previous Loan Default", ["No", "Yes"],
                                        key="approval_default")

        accept_approval = st.form_submit_button("Accept Loan Approval Application")

    if accept_approval:
        input_values = prepare_approval_data(
            approval_age, approval_income, approval_work, approval_amount,
            approval_interest, approval_percent, approval_history,
            approval_score, approval_gender, approval_education, approval_home,
            approval_intent, approval_default
        )
        show_result(loan_approval_model, input_values)
