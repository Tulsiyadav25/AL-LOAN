import pandas as pd
import pickle
from flask import Flask, request, jsonify
import mysql.connector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ========================== MODEL TRAINING ==========================

# Load Dataset
data = pd.read_csv("loan_data.csv")  # Ensure this dataset is available

# Feature Selection
X = data[['age', 'income', 'loan_amount', 'credit_score']]
y = data['loan_status']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save Model & Scaler
pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model and Scaler saved successfully.")

# ========================== FLASK APP ==========================

app = Flask(_name_)

# Load Model & Scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# MySQL Connection
conn = mysql.connector.connect(host="localhost", user="root", password="", database="loan_db")
cursor = conn.cursor()

@app.route('/')
def home():
    return '''
    <h2>Loan Approval Prediction</h2>
    <form action="/predict" method="post">
        <label>Name:</label>
        <input type="text" name="name" required><br><br>
        
        <label>Age:</label>
        <input type="number" name="age" required><br><br>
        
        <label>Income:</label>
        <input type="number" name="income" required><br><br>
        
        <label>Loan Amount:</label>
        <input type="number" name="loan_amount" required><br><br>
        
        <label>Credit Score:</label>
        <input type="number" name="credit_score" required><br><br>

        <button type="submit">Check Approval</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    name = data['name']
    age = int(data['age'])
    income = float(data['income'])
    loan_amount = float(data['loan_amount'])
    credit_score = int(data['credit_score'])

    # Data Preprocessing
    input_data = scaler.transform([[age, income, loan_amount, credit_score]])
    prediction = model.predict(input_data)[0]
    status = "Approved" if prediction == 1 else "Rejected"

    # Save to MySQL
    query = "INSERT INTO loan_applicants (name, age, income, loan_amount, credit_score, approval_status) VALUES (%s, %s, %s, %s, %s, %s)"
    values = (name, age, income, loan_amount, credit_score, status)
    cursor.execute(query, values)
    conn.commit()

    return jsonify({"Loan Status": status})

if _name_ == '_main_':
    app.run(debug=True)
