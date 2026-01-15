import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/Koushik Mandal/Desktop/loan/LoanData.csv")
data.head(5)

data.info()

data.describe()

data.describe(include = 'object')

data['Loan_Status'].value_counts()
data.isnull().sum()
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

data.isnull().sum()

data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median())

data.isnull().sum()

#outlayers & handling
'''plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 6)

plt.subplot(1, 3, 1)
sns.boxplot(data['ApplicantIncome'])

plt.subplot(1, 3, 2)
sns.boxplot(data['CoapplicantIncome'])

plt.subplot(1, 3, 3)
sns.boxplot(data['LoanAmount'])

plt.suptitle("Outliers Detection")
plt.show()'''
print("Before Removing the outliers", data.shape)

data = data[data['ApplicantIncome']<25000]

print("After Removing the outliers", data.shape)


print("Before Removing the outliers", data.shape)

data = data[data['CoapplicantIncome']<12000]

print("After Removing the outliers", data.shape)

print("Before Removing the outliers", data.shape)

data = data[data['LoanAmount']<400]

print("After Removing the outliers", data.shape)


#analysis
'''plt.subplot(1, 3, 1)
sns.distplot(data['ApplicantIncome'], color = 'green')

plt.subplot(1, 3, 2)
sns.distplot(data['CoapplicantIncome'], color = 'green')

plt.subplot(1, 3, 3)
sns.distplot(data['LoanAmount'], color = 'green')

data['ApplicantIncome'] = np.log(data['ApplicantIncome'])
data['CoapplicantIncome'] = np.log1p(data['CoapplicantIncome'])

plt.subplot(1, 3, 1)
sns.distplot(data['ApplicantIncome'], color = 'green')

plt.subplot(1, 3, 2)
sns.distplot(data['CoapplicantIncome'], color = 'green')

plt.subplot(1, 3, 3)
sns.distplot(data['LoanAmount'], color = 'green')

plt.suptitle("After Log Transformation data")
plt.show()'''

num = data.select_dtypes('number').columns.tolist()
cat = data.select_dtypes('object').columns.tolist()

print(num)
print(cat)


'''for i in cat[:-1]:
    plt.figure(figsize = (15,10))
    plt.subplot(2,3,1)
    sns.countplot(x =i, hue = 'Loan_Status', data = data, palette = 'plasma' )
    plt.xlabel(i, fontsize = 15)
    plt.show()'''

print(data.columns)

print(pd.crosstab(data['Loan_Status'], data['Married']))

print(pd.crosstab(data['Loan_Status'], data['Education']))

print(pd.crosstab(data['Loan_Status'], data['Property_Area']))

# Plt Categporial with Target Data

print(pd.crosstab(data['Loan_Status'], data['Self_Employed']))

# Data Prepration

print(data.select_dtypes('object').head())

data = data.drop(['Loan_ID'], axis = 1)

print(data.select_dtypes('object').head())

data['Gender'] = data['Gender'].replace(('Male', 'Female'),(1,0))
data['Married'] = data['Married'].replace(('Yes', 'No'),(1,0))
data['Education'] = data['Education'].replace(('Graduate', 'Not Graduate'),(1,0))

print(data.head())

data['Dependents'].value_counts()
data['Self_Employed'] = data['Self_Employed'].replace(('Yes', 'No'),(1,0))
data['Loan_Status'] = data['Loan_Status'].replace(('Y', 'N'),(1,0))
data['Property_Area'] = data['Property_Area'].replace(('Urban', 'Semiurban','Rural'),(1,1,0))

data['Dependents'] = data['Dependents'].replace(('0', '1','2', '3+'),(0,1,1,1))

print(data.head())

y = data['Loan_Status']
x = data.drop(['Loan_Status'], axis = 1)

print(x.shape)

print(x.columns)
print(y.shape)
print(y)

# Handle Imbalance data

from imblearn.over_sampling import SMOTE

x_rasmple, y_rasmple = SMOTE().fit_resample(x, y.values.ravel())

print(x_rasmple.shape)
print(y_rasmple.shape)

y.shape

#Train test Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x_rasmple, y_rasmple, test_size = 0.2, random_state = 0)

print(x_train.shape)

print(y_test.shape)

#Model Building
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

LR = LogisticRegression()
LR.fit(x_train, y_train)



LogisticRegression()

y_pred = LR.predict(x_test)

print("Traning Accuracy", LR.score(x_train, y_train))
print("Test Accuracy", LR.score(x_test, y_test))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

print("Our Model Accuracy is",accuracy_score(y_pred, y_test) )
print(data.columns)

print(data.head())

model_pred = np.array([[1,1,1, 1, 1,3924, 1733, 148.0, 360, 1, 1]])

prediction = LR.predict(model_pred)
#print(prediction[0])
print("Predicted Loan Status:", "Approved" if prediction[0] == 1 else "Not Approved")









import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load and preprocess the data as done earlier
# (Assuming the model has already been trained)
data = pd.read_csv("C:/Users/Koushik Mandal/Desktop/loan/LoanData.csv")

# Preprocess the data
data['Gender'] = data['Gender'].replace(('Male', 'Female'), (1, 0))
data['Married'] = data['Married'].replace(('Yes', 'No'), (1, 0))
data['Education'] = data['Education'].replace(('Graduate', 'Not Graduate'), (1, 0))
data['Self_Employed'] = data['Self_Employed'].replace(('Yes', 'No'), (1, 0))
data['Loan_Status'] = data['Loan_Status'].replace(('Y', 'N'), (1, 0))
data['Property_Area'] = data['Property_Area'].replace(('Urban', 'Semiurban', 'Rural'), (1, 1, 0))
data['Dependents'] = data['Dependents'].replace(('0', '1', '2', '3+'), (0, 1, 1, 1))

# Handle missing values (as per your preprocessing)
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median())

# Preparing the data
x = data.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = data['Loan_Status']

# Handling imbalanced data using SMOTE
x_resample, y_resample = SMOTE().fit_resample(x, y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, random_state=0)

# Logistic Regression model
LR = LogisticRegression()
LR.fit(x_train, y_train)

# Function to predict loan status based on user input
def predict_loan_status():
    try:
        # Fetching input values from the GUI
        gender = int(gender_var.get())
        married = int(married_var.get())
        education = int(education_var.get())
        self_employed = int(self_employed_var.get())
        applicant_income = float(applicant_income_var.get())
        coapplicant_income = float(coapplicant_income_var.get())
        loan_amount = float(loan_amount_var.get())
        loan_amount_term = int(loan_amount_term_var.get())
        credit_history = float(credit_history_var.get())
        property_area = int(property_area_var.get())
        dependents = int(dependents_var.get())

        # Creating a 2D array with user inputs for prediction
        user_input = np.array([[gender, married, education, self_employed, applicant_income,
                                coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area, dependents]])

        # Predicting loan status
        prediction = LR.predict(user_input)
        result = "Approved" if prediction[0] == 1 else "Not Approved"

        # Showing the result in a message box
        messagebox.showinfo("Prediction Result", f"Loan Status: {result}")

    except Exception as e:
        messagebox.showerror("Input Error", f"An error occurred: {e}")

# Creating the main window
root = tk.Tk()
root.title("Loan Prediction")

# Creating labels and entry widgets for user input
tk.Label(root, text="Gender (1 = Male, 0 = Female):").grid(row=0, column=0)
gender_var = tk.StringVar()
tk.Entry(root, textvariable=gender_var).grid(row=0, column=1)

tk.Label(root, text="Married (1 = Yes, 0 = No):").grid(row=1, column=0)
married_var = tk.StringVar()
tk.Entry(root, textvariable=married_var).grid(row=1, column=1)

tk.Label(root, text="Education (1 = Graduate, 0 = Not Graduate):").grid(row=2, column=0)
education_var = tk.StringVar()
tk.Entry(root, textvariable=education_var).grid(row=2, column=1)

tk.Label(root, text="Self Employed (1 = Yes, 0 = No):").grid(row=3, column=0)
self_employed_var = tk.StringVar()
tk.Entry(root, textvariable=self_employed_var).grid(row=3, column=1)

tk.Label(root, text="Applicant Income:").grid(row=4, column=0)
applicant_income_var = tk.StringVar()
tk.Entry(root, textvariable=applicant_income_var).grid(row=4, column=1)

tk.Label(root, text="Coapplicant Income:").grid(row=5, column=0)
coapplicant_income_var = tk.StringVar()
tk.Entry(root, textvariable=coapplicant_income_var).grid(row=5, column=1)

tk.Label(root, text="Loan Amount:").grid(row=6, column=0)
loan_amount_var = tk.StringVar()
tk.Entry(root, textvariable=loan_amount_var).grid(row=6, column=1)

tk.Label(root, text="Loan Amount Term (in months):").grid(row=7, column=0)
loan_amount_term_var = tk.StringVar()
tk.Entry(root, textvariable=loan_amount_term_var).grid(row=7, column=1)

tk.Label(root, text="Credit History:").grid(row=8, column=0)
credit_history_var = tk.StringVar()
tk.Entry(root, textvariable=credit_history_var).grid(row=8, column=1)

tk.Label(root, text="Property Area (1 = Urban, 1 = Semiurban, 0 = Rural):").grid(row=9, column=0)
property_area_var = tk.StringVar()
tk.Entry(root, textvariable=property_area_var).grid(row=9, column=1)

tk.Label(root, text="Dependents (0, 1, 2, 3+):").grid(row=10, column=0)
dependents_var = tk.StringVar()
tk.Entry(root, textvariable=dependents_var).grid(row=10, column=1)

# Button to trigger prediction
tk.Button(root, text="Predict Loan Status", command=predict_loan_status).grid(row=11, columnspan=2)

# Run the main loop
root.mainloop()
