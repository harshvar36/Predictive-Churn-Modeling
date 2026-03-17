import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('Telco-Customer-Churn.csv')

# 2. Data Cleaning: The "TotalCharges" issue
# In this dataset, some TotalCharges are empty strings " ", which Python doesn't see as null.
# errors='coerce' turns those spaces into NaN (Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Check for missing values created by the coercion
print(f"Missing values found: {df.isnull().sum()['TotalCharges']}")

# 4. Drop those missing values (only 11 rows out of 7000+)
df.dropna(inplace=True)

# 5. Visualizing the Target (Churn)
sns.countplot(x='Churn', data=df)
plt.title('How many customers actually left?')
plt.show()

# %%
# 1. Plotting Tenure vs Churn
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple="stack", palette='magma')
plt.title('Tenure Distribution by Churn')
plt.show()

# %%
# 1. Reset: Go back to the original df but with the TotalCharges fix
df_clean = df.drop(['customerID'], axis=1)

# 2. Encoding
# Label encode the simple Yes/No columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for col in binary_cols:
    df_clean[col] = le.fit_transform(df_clean[col])

# 3. One-Hot Encode the remaining categories
# This will now create roughly 30 columns, NOT 7,000!
df_final = pd.get_dummies(df_clean, drop_first=True)

print("Clean Data Shape:", df_final.shape) 
# It should show something like (7032, 31)
print(df_final.head())
# %%# %%# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Define Features (X) and Target (y) using our clean df_final
X = df_final.drop('Churn', axis=1)
y = df_final['Churn']

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Generate predictions
y_pred = model.predict(X_test)

print("Model re-trained successfully with clean data!")
# %%
from sklearn.metrics import classification_report, confusion_matrix

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
# %%
# 1. Get feature importance from our trained model
importances = model.feature_importances_
feature_names = X.columns

# 2. Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 3. Plot the top 10 most important features
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title('Top 10 Drivers of Customer Churn')
plt.show()
# %%
def predict_churn(customer_data):
    """
    Takes a dictionary of customer traits and predicts churn.
    """
    # 1. Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # 2. Re-apply the same encoding we used for the training data
    # We must ensure the input has the EXACT same 31 columns as X_train
    input_encoded = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    
    # 3. Make the prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1] # Probability of Churn (Class 1)
    
    status = "CHURN" if prediction == 1 else "STAY"
    print(f"Prediction: {status}")
    print(f"Churn Probability: {probability:.2%}")

# --- TEST CASE 1: High Risk (New customer, high charges, month-to-month) ---
new_customer_at_risk = {
    'tenure': 1,
    'MonthlyCharges': 95.50,
    'TotalCharges': 95.50,
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic',
    'PaymentMethod': 'Electronic check',
    'PaperlessBilling': 'Yes'
}

print("Testing High Risk Customer:")
predict_churn(new_customer_at_risk)

# --- TEST CASE 2: Low Risk (Long-term loyal customer, low charges) ---
loyal_customer = {
    'tenure': 72,
    'MonthlyCharges': 20.00,
    'TotalCharges': 1440.00,
    'Contract': 'Two year',
    'InternetService': 'No',
    'PaymentMethod': 'Credit card (automatic)',
    'PaperlessBilling': 'No'
}

print("\nTesting Loyal Customer:")
predict_churn(loyal_customer)