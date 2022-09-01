from pydantic import BaseModel, Field
import joblib
import pickle
import re
from fastapi import FastAPI
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import os
import uvicorn


class ChargeOff(BaseModel):
    loan_amnt:float = Field(example=2000)
    purpose:str = Field(example='wedding')
    fico:float = Field(example = 800)
    dti:float = Field(example = 1)
    addr_state:str = Field(example = 'CA')
    emp_length:float = Field(example = 10)
    
class Grade(BaseModel):
    loan_amnt:float = Field(example = 4575)
    term:int = Field(example = 36)
    emp_length:float = Field(example = 10)
    home_ownership:str = Field(example = 'MORTGAGE')
    annual_inc:float = Field(example = 4.716011695371454)
    verification_status:str = Field(example = 'Source Verified')
    purpose:str = Field(example = 'debt_consolidation')
    addr_state:str = Field(example = 'AZ')
    dti:float = Field(example = 29.74)
    pub_rec:float = Field(example = 0)
    total_acc:float = Field(example = 12)
    initial_list_status:str = Field(example = 'w')
    application_type:str = Field(example = 'Individual')
    mo_sin_old_il_acct:float = Field(example = 157)
    mo_sin_old_rev_tl_op:float = Field(example = 71)
    mort_acc:float = Field(example = 2)
    fico:float = Field(example = 672)

class Sub_grade(BaseModel):
    loan_amnt:float = Field(example = 4575)
    term:int = Field(example = 36)
    grade:str = Field(example = 'C')
    emp_length:float = Field(example = 10)
    home_ownership:str = Field(example = 'MORTGAGE')
    annual_inc:float = Field(example = 4.716011695371454)
    verification_status:str = Field(example = 'Source Verified')
    purpose:str = Field(example = 'debt_consolidation')
    addr_state:str = Field(example = 'AZ')
    dti:float = Field(example = 29.74)
    pub_rec:float = Field(example = 0)
    total_acc:float = Field(example = 12)
    initial_list_status:str = Field(example = 'w')
    application_type:str = Field(example = 'Individual')
    mo_sin_old_il_acct:float = Field(example = 157)
    mo_sin_old_rev_tl_op:float = Field(example = 71)
    mort_acc:float = Field(example = 2)
    fico:float = Field(example = 672)

class Int_rate(BaseModel):
    loan_amnt:float = Field(example = 4575)
    term:int = Field(example = 36)
    grade:str = Field(example = 'C')
    sub_grade:str = Field(example = 'C2')
    emp_length:float = Field(example = 10)
    home_ownership:str = Field(example = 'MORTGAGE')
    annual_inc:float = Field(example = 4.716011695371454)
    verification_status:str = Field(example = 'Source Verified')
    purpose:str = Field(example = 'debt_consolidation')
    addr_state:str = Field(example = 'AZ')
    dti:float = Field(example = 29.74)
    pub_rec:float = Field(example = 0)
    total_acc:float = Field(example = 12)
    initial_list_status:str = Field(example = 'w')
    application_type:str = Field(example = 'Individual')
    mo_sin_old_il_acct:float = Field(example = 157)
    mo_sin_old_rev_tl_op:float = Field(example = 71)
    mort_acc:float = Field(example = 2)
    fico:float = Field(example = 672)
    
app = FastAPI(debug=True, title = 'Loan Risks Prediction API', version = 1.0, description = 'Simple API to predict Loan risks.')

#creating the classifier
pickle_in = open('models/charged_off_xgb.pkl', 'rb')
chargeoff_model = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('models/grade_xgb.pkl', 'rb')
grade_model = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('models/sub_grade_lr.pkl', 'rb')
subgrade_model = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('models/int_rate_lreg.pkl', 'rb')
intrate_model = pickle.load(pickle_in)
pickle_in.close()



@app.get('/')
def get_root():
    return {'message': 'Welcome to the loan risk detection api v2'}

@app.post('/chargeoff_detection/')
async def predict_chargeoff(data:ChargeOff):
    columns = ['loan_amnt', 'purpose', 'fico', 'dti', 'addr_state', 'emp_length']
    data = data.dict()
    inp = [value for (key,value) in data.items()]
    df = pd.DataFrame([inp], columns=columns)
    prediction = chargeoff_model.predict(df).tolist()
    prob = chargeoff_model.predict_proba(df)[0][1].tolist()
    return {'prediction':prediction, 'probability':prob}

@app.post('/grade_prediction/')
async def predict_grade(data:Grade):
    columns = ['loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc','verification_status', 'purpose', 'addr_state', 'dti', 'pub_rec','total_acc', 'initial_list_status', 'application_type','mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mort_acc', 'fico']
    data = data.dict()
    inp = [value for (key,value) in data.items()]
    df = pd.DataFrame([inp], columns=columns)
    prediction = grade_model.predict(df).tolist()
    prob = grade_model.predict_proba(df)[0][1].tolist()
    return {'prediction':prediction, 'probability':prob}

@app.post('/sub_grade_prediction/')
async def predict_subgrade(data:Sub_grade):
    columns = ['loan_amnt', 'term', 'grade', 'emp_length', 'home_ownership', 'annual_inc','verification_status', 'purpose', 'addr_state', 'dti', 'pub_rec','total_acc', 'initial_list_status', 'application_type','mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mort_acc', 'fico']
    data = data.dict()
    inp = [value for (key,value) in data.items()]
    df = pd.DataFrame([inp], columns=columns)
    prediction = subgrade_model.predict(df).tolist()
    prob = subgrade_model.predict_proba(df)[0][1].tolist()
    return {'prediction':prediction, 'probability':prob}

@app.post('/intrate_prediction/')
async def predict_intrate(data:Int_rate):
    columns = ['loan_amnt', 'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc','verification_status', 'purpose', 'addr_state', 'dti', 'pub_rec','total_acc', 'initial_list_status', 'application_type','mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mort_acc', 'fico']
    data = data.dict()
    inp = [value for (key,value) in data.items()]
    df = pd.DataFrame([inp], columns=columns)
    prediction = intrate_model.predict(df).tolist()
    return {'prediction':prediction}


#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)
