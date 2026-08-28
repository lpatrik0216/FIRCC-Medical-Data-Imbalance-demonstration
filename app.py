import streamlit as st
import pandas as pd
import numpy as np
from keras.src.metrics.accuracy_metrics import accuracy
from numpy.ma.extras import average
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="INPACE Living Lab - AI Data Drift", layout="wide")
st.title("Medical AI Data Drift demonstration")
st.markdown(
    "What are the risks of deploying an AI model trained on a single population to a new demographic, and how can we mitigate them?")

tsh_shift = st.slider("TSH Shift Value (Research suggests 1.25 is a realistic multiplier for Asian populations)", min_value=1.0, max_value=2.0, value=1.25, step=0.05)
st.write("Note that the Asian population data is synthetic, created from the Western dataset with realistic (25% increase) shifts in TSH and age based on research.")
@st.cache_data
def load_and_prepare_data(shift_val):
    population_a = pd.read_csv('archive/cleaned_dataset_Thyroid1.csv')
    target_col = 'binaryClass'
    healthy_label = 0

    population_a['TSH'] = pd.to_numeric(population_a['TSH'], errors='coerce')
    population_a = population_a.dropna(subset=['TSH', 'age'])

    population_b = population_a.copy()
    healthy_mask = population_b[target_col] == healthy_label
    sick_mask = population_b[target_col] != healthy_label

    # Itt használjuk a paraméterként kapott értéket
    population_b.loc[healthy_mask, 'TSH'] = population_b.loc[healthy_mask, 'TSH'] * shift_val + 0.5
    population_b.loc[sick_mask, 'age'] = population_b.loc[sick_mask, 'age'] * 1.15

    for col in population_a.columns:
        if population_a[col].dtype == 'object':
            le = LabelEncoder()
            combined = pd.concat([population_a[col], population_b[col]]).astype(str)
            le.fit(combined)
            population_a[col] = le.transform(population_a[col].astype(str))
            population_b[col] = le.transform(population_b[col].astype(str))

    X_A = population_a.drop(columns=[target_col])
    y_A = population_a[target_col]
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

    X_B = population_b.drop(columns=[target_col])
    y_B = population_b[target_col]
    X_test_B = X_B.loc[X_test_A.index]
    y_test_B = y_B.loc[y_test_A.index]

    return X_train_A, X_test_A, y_train_A, y_test_A, X_test_B, y_test_B, population_b

# 3. A függvény meghívásakor átadjuk a slider aktuális értékét
X_train_A, X_test_A, y_train_A, y_test_A, X_test_B, y_test_B, pop_b = load_and_prepare_data(tsh_shift)


@st.cache_resource
def train_models(current_pop_b):
    target_col = 'binaryClass'

    model_base = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_base.fit(X_train_A, y_train_A)

    X_train_B = current_pop_b.drop(columns=['binaryClass']).loc[X_train_A.index]
    y_train_B = current_pop_b['binaryClass'].loc[y_train_A.index]

    X_train_A_recal = X_train_A.copy()
    X_train_A_recal['is_asian'] = 0
    X_train_B_recal = X_train_B.copy()
    X_train_B_recal['is_asian'] = 1

    X_train_combined = pd.concat([X_train_A_recal, X_train_B_recal])
    y_train_combined = pd.concat([y_train_A, y_train_B])

    model_recal = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_recal.fit(X_train_combined, y_train_combined)

    return model_base, model_recal


model_base, model_recalibrated = train_models(pop_b)

tab1, tab2, tab3, tab4 = st.tabs(["1. Base model (Western population)", "2. Domain Shift (Asian population)", "3. Recalibrated model", "4. Patient simulation"])

with tab1:
    st.header("1. Step: Base model")
    st.write("This model was trained on a Western population. Let's see how it performs on its own demographic.")

    preds_A = model_base.predict(X_test_A)
    f1_A = f1_score(y_test_A, preds_A, average='macro')

    st.metric(label="F1-Score (Western Population)", value=f"{f1_A * 100:.2f}%")

with tab2:
    st.header("Step 2: Data Shift and the AI's Collapse")
    st.write(
        "We will now test the same Western model on the Asian demographic.")

    preds_B = model_base.predict(X_test_B)
    f1_B = f1_score(y_test_B, preds_B, average='macro')

    delta_val = f"{(f1_B - f1_A) * 100:.2f}%"
    st.metric(label="F1-Score (Asian Data)", value=f"{f1_B * 100:.2f}%", delta=delta_val, delta_color="red")

    st.subheader("Why did the model fail? (SHAP Analysis)")
    st.write(
        "The figures below show how the higher healthy TSH levels confused the decision logic.")

    col1, col2 = st.columns(2)
    explainer = shap.TreeExplainer(model_base)

    with col1:
        st.write("**Western Patients**")
        vals_A = explainer.shap_values(X_test_A)
        vals_A = vals_A[1] if isinstance(vals_A, list) else (vals_A[:, :, 1] if len(vals_A.shape) == 3 else vals_A)

        plt.figure(figsize=(6, 4))
        shap.summary_plot(vals_A, X_test_A, show=False, sort=False)
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.write("**Asian patients**")
        vals_B = explainer.shap_values(X_test_B)
        vals_B = vals_B[1] if isinstance(vals_B, list) else (vals_B[:, :, 1] if len(vals_B.shape) == 3 else vals_B)

        plt.figure(figsize=(6, 4))
        shap.summary_plot(vals_B, X_test_B, show=False, sort=False)
        st.pyplot(plt.gcf())
        plt.clf()

with tab3:
    st.header("3. Step: AI Recalibration és Fair Decision-Making")
    st.write(
        "The model was retrained on a comprehensive dataset incorporating both Western and Asian populations, explicitly providing demographic context via an is_asian flag.")

    X_test_A_recal = X_test_A.copy()
    X_test_A_recal['is_asian'] = 0
    X_test_B_recal = X_test_B.copy()
    X_test_B_recal['is_asian'] = 1

    # Életszerű zaj hozzáadása
    np.random.seed(42)
    X_test_B_recal_noisy = X_test_B_recal.copy()
    X_test_B_recal_noisy['TSH'] = X_test_B_recal_noisy['TSH'] * np.random.uniform(0.98, 1.02,
                                                                                  size=len(X_test_B_recal_noisy))

    preds_A_recal = model_recalibrated.predict(X_test_A_recal)
    preds_B_recal = model_recalibrated.predict(X_test_B_recal_noisy)

    f1_A_recal = f1_score(y_test_A, preds_A_recal, average='macro')
    f1_B_recal = f1_score(y_test_B, preds_B_recal, average='macro')

    A_accuracy = accuracy_score(y_test_A, preds_A_recal)
    B_accuracy = accuracy_score(y_test_B, preds_B_recal)

    col3, col4 = st.columns(2)
    col3.metric(label="Accuracy (Western population)", value=f"{A_accuracy * 100:.2f}%")
    col4.metric(label="Accuracy (Asian population)", value=f"{B_accuracy * 100:.2f}%")

    col5, col6 = st.columns(2)
    col5.metric(label="New F1-Score (Western data)", value=f"{f1_A_recal * 100:.2f}%")
    col6.metric(label="New F1-Score (Asian data)", value=f"{f1_B_recal * 100:.2f}%", delta="+ Javítva",
                delta_color="normal")

    st.write(
        "But what are these scores, and why are they important?")
    st.write(
        "Accuracy shows us the big picture: out of all patients, how many did the model correctly identify as either healthy or sick?")
    st.write(
        "But in the world of medical AI, accuracy isn't everything. Let's say the model predicts 'unhealthy' only if it is incredibly sure. In this case, our precision will be excellent, since almost every patient it flagged as unhealthy truly is. However, by being so overly cautious, it will completely miss dozens of early-stage patients.")
    st.write(
        "To solve this problem, we use another metric: recall, which tells us how many unhealthy patients our model actually successfully found out of all the truly unhealthy ones.")
    st.write(
        "And finally, our F1 score. Instead of just validating our model on a single, easily skewed metric, the F1 score evaluates it based on a balance of both precision and recall.")

    with tab4:
        st.header("Patient Simulator")
        st.write("Input a hypothetical patient's data to see how the two models differ in their diagnosis.")

        # Checkbox for full simulation mode
        full_sim = st.checkbox("Full Simulation")

        input_col, result_col = st.columns([1, 1])

        with input_col:
            st.subheader("Patient Vitals")
            patient_data = {}

            patient_data['age'] = st.number_input("Age", min_value=1, max_value=100, value=45)

            gender_col = next((col for col in X_train_A.columns if col.lower() in ['sex', 'gender']), None)
            if gender_col:
                patient_data[gender_col] = st.selectbox(
                    f"Gender ({gender_col})",
                    options=[0, 1],
                    format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
                )

            # Conditionally render TSH based on simulation mode
            if full_sim:
                patient_data['TSH'] = st.number_input("TSH Level", min_value=0.0, value=6.2, step=0.1)
            else:
                patient_data['TSH'] = st.slider("TSH Level", min_value=0.0, max_value=15.0, value=6.2, step=0.1)

            demographic = st.selectbox(
                "Demographic Context",
                options=[0, 1],
                format_func=lambda x: "Asian Population" if x == 1 else "Western Population"
            )

            if full_sim:
                st.markdown("---")
                st.write("**Detailed datapoints:**")
                for col in X_train_A.columns:
                    if col not in patient_data:
                        unique_vals = X_train_A[col].dropna().unique()
                        is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})

                        median_val = X_train_A[col].median()

                        if is_binary:
                            patient_data[col] = st.selectbox(
                                f"{col}",
                                options=[0, 1],
                                index=0 if median_val == 0 else 1,
                                format_func=lambda x: "True" if x == 1 else "False"
                            )
                        else:
                            patient_data[col] = st.number_input(f"{col}", value=float(median_val))
            else:
                # Silently fill the rest with median values
                for col in X_train_A.columns:
                    if col not in patient_data:
                        patient_data[col] = X_train_A[col].median()

        # Create DataFrames for prediction
        patient_df_base = pd.DataFrame([patient_data])

        # Reorder columns to exactly match the base model's training data
        patient_df_base = patient_df_base[X_train_A.columns]

        patient_df_recal = patient_df_base.copy()
        patient_df_recal['is_asian'] = demographic

        # Reorder columns to exactly match the recalibrated model's training data
        expected_recal_cols = list(X_train_A.columns) + ['is_asian']
        patient_df_recal = patient_df_recal[expected_recal_cols]

        # Run predictions
        pred_base = model_base.predict(patient_df_base)[0]
        pred_recal = model_recalibrated.predict(patient_df_recal)[0]

        with result_col:
            st.subheader("AI Diagnosis Results")

            st.write("**1. Western Base Model:**")
            if pred_base == 0:
                st.success("Diagnosis: Healthy")
            else:
                st.error("Diagnosis: Sick")

            st.write("---")

            st.write("**2. Recalibrated Model:**")
            if pred_recal == 0:
                st.success("Diagnosis: Healthy")
            else:
                st.error("Diagnosis: Sick")