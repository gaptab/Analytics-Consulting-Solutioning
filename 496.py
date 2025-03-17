import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
import joblib

# 🚀 Step 1: Generate Dummy Data for Industry Analysis
np.random.seed(42)
n_samples = 5000

industries = ['Retail', 'Banking', 'Healthcare', 'Telecom', 'E-commerce']
issues = ['High Customer Churn', 'Fraud Risk', 'Low Sales', 'Operational Inefficiency', 'High Acquisition Cost']

df = pd.DataFrame({
    'industry': np.random.choice(industries, n_samples),
    'customer_count': np.random.randint(1000, 50000, n_samples),
    'avg_transaction_value': np.random.uniform(20, 500, n_samples),
    'num_transactions': np.random.randint(100, 2000, n_samples),
    'marketing_spend': np.random.uniform(5000, 500000, n_samples),
    'customer_satisfaction': np.random.uniform(1, 10, n_samples),
    'churn_rate': np.random.uniform(0.01, 0.5, n_samples),
    'fraud_cases': np.random.randint(0, 50, n_samples),
    'issue_identified': np.random.choice(issues, n_samples)
})

# 🚀 Step 2: Identify Business Issues via Clustering (Segment Industries)
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[['customer_count', 'avg_transaction_value', 'num_transactions', 'marketing_spend']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster)

print(f"Silhouette Score: {silhouette_score(X_cluster, df['cluster']):.2f}")

# 🚀 Step 3: Develop Fraud Prediction Model (For Banking Industry)
df_fraud = df[df['industry'] == 'Banking'].copy()
df_fraud['fraud_label'] = (df_fraud['fraud_cases'] > df_fraud['fraud_cases'].median()).astype(int)

X = df_fraud[['customer_count', 'avg_transaction_value', 'num_transactions', 'marketing_spend', 'churn_rate']]
y = df_fraud['fraud_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Fraud Detection Model Performance:\n", classification_report(y_test, y_pred))

# 🚀 Step 4: Save Reports and Model
df.to_csv("industry_analysis_data.csv", index=False)
joblib.dump(model, "fraud_detection_model.pkl")

print("Data and Model Saved Successfully!")
