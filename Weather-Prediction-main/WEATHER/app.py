
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/weather', methods=['POST', 'GET'])
def rweather():
    return render_template('resultw.html')

@app.route('/resultw.html', methods=['POST', 'GET'])
def weather():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    
    if(prediction == 'drizzle'):
        pred = "Drizzle"
    elif(prediction =='fog'):
        pred = "Fog"
    elif(prediction =='rain'):
        pred = "Rain"
    elif(prediction=='snow'):
        pred = "Snow"
    elif(prediction == 'sun'):
        pred = 'Sun'
    
    output = pred
    return render_template('resultw.html', prediction_text = 'The Climate is  {}'.format(output))
        
if __name__ == "__main__":
    app.run(debug=True)
   