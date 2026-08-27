import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# Load Data
population_a = pd.read_csv('archive/cleaned_dataset_Thyroid1.csv')
target_col = 'binaryClass'
healthy_label = 0


# Apply the Drift to Population B
population_b = population_a.copy()
healthy_mask = population_b[target_col] == healthy_label
sick_mask = population_b[target_col] != healthy_label

# Realistic TSH shift: ~25% higher baseline + 0.5 constant to widen the "normal" range
population_b.loc[healthy_mask, 'TSH'] = population_b.loc[healthy_mask, 'TSH'] * 1.25 + 0.5

# Realistic age shift: positive cases skew slightly older (approx. 15% increase)
population_b.loc[sick_mask, 'age'] = population_b.loc[sick_mask, 'age'] * 1.15

# Verify the drift mathematically
print(f"Pop A Mean TSH: {population_a['TSH'].mean():.2f} | Pop B Mean TSH: {population_b['TSH'].mean():.2f}")

# Encode and Split
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

# Train and Evaluate
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_A, y_train_A)

preds_A = model.predict(X_test_A)
preds_B = model.predict(X_test_B)

print(f"Population A (Baseline) F1-Score: {f1_score(y_test_A, preds_A, average='macro') * 100:.2f}%")
print(f"Population B (Shifted) F1-Score: {f1_score(y_test_B, preds_B, average='macro') * 100:.2f}%")



explainer = shap.TreeExplainer(model)

# Calculate SHAP values for both the Baseline and Shifted test sets
print("Calculating SHAP values... (This might take a few seconds)")
shap_values_A = explainer.shap_values(X_test_A)
shap_values_B = explainer.shap_values(X_test_B)


if isinstance(shap_values_A, list):
    vals_A = shap_values_A[1]
    vals_B = shap_values_B[1]
elif len(shap_values_A.shape) == 3:
    vals_A = shap_values_A[:, :, 1]
    vals_B = shap_values_B[:, :, 1]
else:
    vals_A = shap_values_A
    vals_B = shap_values_B

plt.figure(figsize=(8, 5))
plt.title("Population A: Feature Importance (Clear Logic)")
shap.summary_plot(vals_A, X_test_A, show=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.title("Population B: Feature Importance (Confused Model)")
shap.summary_plot(vals_B, X_test_B, show=False)
plt.tight_layout()
plt.show()





print("\n--- Recalibration ---")

X_train_B = population_b.drop(columns=[target_col]).loc[X_train_A.index]
y_train_B = population_b[target_col].loc[y_train_A.index]

X_train_A_recal = X_train_A.copy()
X_train_A_recal['is_asian'] = 0

X_train_B_recal = X_train_B.copy()
X_train_B_recal['is_asian'] = 1

X_test_A_recal = X_test_A.copy()
X_test_A_recal['is_asian'] = 0

X_test_B_recal = X_test_B.copy()
X_test_B_recal['is_asian'] = 1

X_train_combined = pd.concat([X_train_A_recal, X_train_B_recal])
y_train_combined = pd.concat([y_train_A, y_train_B])

model_recalibrated = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_recalibrated.fit(X_train_combined, y_train_combined)

# Adding noise to the test data
np.random.seed(42)
X_test_B_recal_noisy = X_test_B_recal.copy()
X_test_B_recal_noisy['TSH'] = X_test_B_recal_noisy['TSH'] * np.random.uniform(0.98, 1.02, size=len(X_test_B_recal))

preds_A_recal = model_recalibrated.predict(X_test_A_recal)
preds_B_recal = model_recalibrated.predict(X_test_B_recal_noisy)

f1_A_recal = f1_score(y_test_A, preds_A_recal, average='macro')
f1_B_recal = f1_score(y_test_B, preds_B_recal, average='macro')

print(f"Recalibrated Model -> Population A (Western) F1-Score: {f1_A_recal * 100:.2f}%")
print(f"Recalibrated Model -> Population B (Asian) F1-Score: {f1_B_recal * 100:.2f}%")