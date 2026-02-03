from flask import Flask,request
import pickle
import pandas as pd

app = Flask(__name__)
model_pkl = open("classifier.pkl","rb")
clf = pickle.load(model_pkl)

@app.route("/")
def hello_ping():
    return {'Message ':'Hi , Welcome to Loan Prediction Model 3.0!'}

@app.route("/predict",methods=['POST'])
def prediction():
    loan_req = request.get_json()
    print(loan_req)

    if loan_req['Gender'] == "Male":
        Gender = 0
    else:
        Gender = 1

    if loan_req['Married'] == "Unmarried":
        Married = 0
    else:
        Married = 1

    if loan_req['Credit_History'] == "Uncleared Debts":
        Credit_History = 0
    else:
        Credit_History = 1

    ApplicantIncome = loan_req ['ApplicantIncome']
    LoanAmount = loan_req['LoanAmount']

    result = clf.predict([[Gender,Married,ApplicantIncome,LoanAmount,Credit_History]])

    if result == 0:
        pred = "Rejected"
    else:
        
        pred = "Approved"

    return {"Loan Approval Status for you is ":pred}



# loan_application = {
#     'Gender': "Male",
#     'Married': "Unmarried",
#     'ApplicantIncome': 50000,
#     'Credit_Mistory': "Cleared Debts"
#     'LoanAmount': 500000
# }