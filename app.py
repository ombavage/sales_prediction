
import json
import numpy as np
import pickle
from flask import Flask, render_template,request

app =Flask(__name__)

column = json.load(open('feature.json','r'))
list_col =column['col']
model = pickle.load(open('rf_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    arr =np.zeros(len(column['col']))

    Item_Wt = request.form['Item_Weight']
    Item_Fat   = request.form['Item_Fat_Content']
    Item_Visi  = request.form['Item_Visibility']
    Item_MRP = request.form['Item_MRP']
    Outlet_Size  = request.form['Outlet_Size']
    Outlet_Location_Type  = request.form['Outlet_Location_Type']
    Outlet_Type  = request.form['Outlet_Type']
    Outlet_Age = request.form['Outlet_Age']
    Item_Type  = request.form['Item_Type']
    

    arr[0] = Item_Wt
    arr[1] = Item_Fat
    arr[2] = Item_Visi
    arr[3] = Item_MRP
    arr[4] = Outlet_Size
    arr[5] = Outlet_Location_Type
    arr[6] = Outlet_Type
    arr[list_col.index(Item_Type)] = 1
    arr[-1] = Outlet_Age
    

    pred = model.predict([arr])

    return render_template('index.html',result=pred)


if __name__ =='__main__':
    app.run()

