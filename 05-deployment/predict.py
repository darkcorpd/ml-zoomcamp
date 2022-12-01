import pickle
import numpy as np
import requests
from flask import Flask, request, jsonify

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in_model:
    model = pickle.load(f_in_model)

with open(dv_file, 'rb') as f_in_dv:
    dv = pickle.load(f_in_dv)

app = Flask('card')
@app.route('/predict', methods=['POST'])

def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[:, 1]

    return str(y_pred[0])
    
#prediction = predict()

#print('prediction: %.3f' % prediction)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)