import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="INPACE Living Lab - AI Data Drift", layout="wide")
st.title("Orvosi AI Adateltolódás (Data Drift) Demonstráció")
st.markdown(
    "Hogyan válik veszélyessé egy nyugati adatokon betanított orvosi AI egy ázsiai populáción, és hogyan javíthatjuk ki?")


@st.cache_data
def load_and_prepare_data():
    population_a = pd.read_csv('archive/cleaned_dataset_Thyroid1.csv')
    target_col = 'binaryClass'
    healthy_label = 0

    population_a['TSH'] = pd.to_numeric(population_a['TSH'], errors='coerce')
    population_a = population_a.dropna(subset=['TSH', 'age'])

    population_b = population_a.copy()
    healthy_mask = population_b[target_col] == healthy_label
    sick_mask = population_b[target_col] != healthy_label

    population_b.loc[healthy_mask, 'TSH'] = population_b.loc[healthy_mask, 'TSH'] * 1.25 + 0.5
    population_b.loc[sick_mask, 'age'] = population_b.loc[sick_mask, 'age'] * 1.15

    for col in population_a.columns:
        if population_a[col].dtype == 'object':
            le = LabelEncoder()
            combined = pd.concat([population_a[col], population_b[col]]).astype(str)
            le.fit(combined)
            population_a[col] = le.transform(population_a[col].astype(str))
            population_b[col] = le.transform(population_b[col].astype(str))

    # 4. Train/Test Split
    X_A = population_a.drop(columns=[target_col])
    y_A = population_a[target_col]
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

    X_B = population_b.drop(columns=[target_col])
    y_B = population_b[target_col]
    X_test_B = X_B.loc[X_test_A.index]
    y_test_B = y_B.loc[y_test_A.index]

    return X_train_A, X_test_A, y_train_A, y_test_A, X_test_B, y_test_B, population_b


X_train_A, X_test_A, y_train_A, y_test_A, X_test_B, y_test_B, pop_b = load_and_prepare_data()


@st.cache_resource
def train_models():
    target_col = 'binaryClass'

    model_base = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_base.fit(X_train_A, y_train_A)

    X_train_B = pop_b.drop(columns=[target_col]).loc[X_train_A.index]
    y_train_B = pop_b[target_col].loc[y_train_A.index]

    X_train_A_recal = X_train_A.copy()
    X_train_A_recal['is_asian'] = 0
    X_train_B_recal = X_train_B.copy()
    X_train_B_recal['is_asian'] = 1

    X_train_combined = pd.concat([X_train_A_recal, X_train_B_recal])
    y_train_combined = pd.concat([y_train_A, y_train_B])

    model_recal = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_recal.fit(X_train_combined, y_train_combined)

    return model_base, model_recal


model_base, model_recalibrated = train_models()

tab1, tab2, tab3 = st.tabs(["1. Alapmodell (Nyugat)", "2. Domain Shift (Ázsia)", "3. Újrakalibrált Modell"])

with tab1:
    st.header("1. Lépés: Alapmodell")
    st.write("A modellt a nyugati klinikai adatokon tanítottuk be. Lássuk, hogyan teljesít a saját demográfiáján.")

    preds_A = model_base.predict(X_test_A)
    f1_A = f1_score(y_test_A, preds_A, average='macro')

    st.metric(label="F1-Score (Nyugati Adatokon)", value=f"{f1_A * 100:.2f}%")

with tab2:
    st.header("2. Lépés: Adateltolódás és Modell Összeomlás")
    st.write(
        "Ugyanezt a nyugati modellt most a távol-keleti pácienseken teszteljük.")

    preds_B = model_base.predict(X_test_B)
    f1_B = f1_score(y_test_B, preds_B, average='macro')

    delta_val = f"{(f1_B - f1_A) * 100:.2f}%"
    st.metric(label="F1-Score (Ázsiai Adatokon)", value=f"{f1_B * 100:.2f}%", delta=delta_val, delta_color="inverse")

    st.subheader("Miért hibázott az AI? (SHAP Analízis)")
    st.write(
        "Az alábbi ábrák mutatják, hogy a magasabb egészséges TSH-szint hogyan zavarta össze a döntési logikát.")

    col1, col2 = st.columns(2)
    explainer = shap.TreeExplainer(model_base)

    with col1:
        st.write("**Nyugati páciensek (Tiszta logika)**")
        vals_A = explainer.shap_values(X_test_A)
        vals_A = vals_A[1] if isinstance(vals_A, list) else (vals_A[:, :, 1] if len(vals_A.shape) == 3 else vals_A)

        plt.figure(figsize=(6, 4))
        shap.summary_plot(vals_A, X_test_A, show=False, sort=False)
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.write("**Ázsiai páciensek (Összezavarodott döntések)**")
        vals_B = explainer.shap_values(X_test_B)
        vals_B = vals_B[1] if isinstance(vals_B, list) else (vals_B[:, :, 1] if len(vals_B.shape) == 3 else vals_B)

        plt.figure(figsize=(6, 4))
        shap.summary_plot(vals_B, X_test_B, show=False, sort=False)
        st.pyplot(plt.gcf())
        plt.clf()

with tab3:
    st.header("3. Lépés: AI Újrakalibráció és Fair Döntéshozatal")
    st.write(
        "A modellt újra tanítottuk egy kibővített adathalmazon, megadva a demográfiai kontextust (`is_asian` flag).")

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

    col3, col4 = st.columns(2)
    col3.metric(label="Új F1-Score (Nyugati Adatok)", value=f"{f1_A_recal * 100:.2f}%")
    col4.metric(label="Új F1-Score (Ázsiai Adatok)", value=f"{f1_B_recal * 100:.2f}%", delta="+ Javítva",
                delta_color="normal")