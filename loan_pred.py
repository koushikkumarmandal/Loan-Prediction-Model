import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib


data = pd.read_csv("LoanData.csv")

data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median())

data = data.drop(['Loan_ID'], axis=1)

data['Gender'] = data['Gender'].replace(('Male', 'Female'), (1, 0))
data['Married'] = data['Married'].replace(('Yes', 'No'), (1, 0))
data['Education'] = data['Education'].replace(('Graduate', 'Not Graduate'), (1, 0))
data['Self_Employed'] = data['Self_Employed'].replace(('Yes', 'No'), (1, 0))
data['Loan_Status'] = data['Loan_Status'].replace(('Y', 'N'), (1, 0))
data['Property_Area'] = data['Property_Area'].replace(('Urban', 'Semiurban', 'Rural'), (1, 1, 0))
data['Dependents'] = data['Dependents'].replace(('0', '1', '2', '3+'), (0, 1, 1, 1))

data['LoanAmount'] *= 1000

y = data['Loan_Status']
x = data.drop(['Loan_Status'], axis=1)

# Balance dataset
x_resample, y_resample = SMOTE().fit_resample(x, y.values.ravel())
x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, random_state=0)



models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True)
}

best_model = None
best_accuracy = 0
best_model_name = ""

accuracies = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_accuracy:
        best_model = model
        best_model_name = name
        best_accuracy = acc

print(f"\n Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")


model_filename = "best_loan_model.pkl"
joblib.dump(best_model, model_filename)





app = tk.Tk()
app.title(" Loan Prediction System")
app.geometry("700x800")
app.configure(bg="#eef2f3")

labels = ["Gender (1=Male, 0=Female)", "Married (1=Yes, 0=No)", "Dependents (0/1)",
          "Education (1=Graduate, 0=Not)", "Self Employed (1=Yes, 0=No)",
          "Applicant Income", "Coapplicant Income", "Loan Amount",
          "Loan Amount Term", "Credit History (1/0)",
          "Property Area (1=Urban/Semiurban, 0=Rural)"]

entries = []

for label in labels:
    tk.Label(app, text=label, bg="#eef2f3", font=("Arial", 10, "bold")).pack()
    entry = tk.Entry(app, width=30)
    entry.pack()
    entries.append(entry)

history = []

def predict():
    try:
        model = joblib.load(model_filename)
        values = [float(e.get()) for e in entries]
        values[7] /= 1000  
        sample = np.array([values])

        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1] * 100
        result = "Loan Approved" if pred == 1 else "Loan Not Approved"

        history.append((values, result, f"{prob:.2f}%"))

        messagebox.showinfo("Prediction Result",
                            f"{result}\nConfidence: {prob:.2f}%\nModel: {best_model_name}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

def reset_fields():
    for e in entries:
        e.delete(0, tk.END)

def show_performance_graph():
    plt.figure(figsize=(8, 5))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title("Model Performance Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.show()

def visualize_data():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Loan_Status', data=data)
    plt.title("Loan Status Distribution")
    plt.show()


tk.Button(app, text="Predict Loan Status", command=predict, bg="green", fg="white", width=30).pack(pady=10)
tk.Button(app, text="Reset Fields", command=reset_fields, bg="orange", fg="white", width=30).pack(pady=5)
tk.Button(app, text="Show Model Performance", command=show_performance_graph, bg="blue", fg="white", width=30).pack(pady=5)
tk.Button(app, text="Visualize Data", command=visualize_data, bg="purple", fg="white", width=30).pack(pady=5)

app.mainloop()