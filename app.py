#from flask import Flask, render_template, request, redirect
import os  # new
from flask import Flask, send_file, jsonify, redirect
from flask import request, render_template
from flask_cors import CORS
import pandas as pd
import sklearn
import json, pickle
from sklearn.tree import DecisionTreeClassifier 
from joblib import load

app = Flask(__name__)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

my_dir = os.path.dirname( __file__)
model_file_path = os.path.join(my_dir, 'model.pickle')
scaler_file_path = os.path.join(my_dir, 'scaler.pickle')

#final selected data
#feature_names = ['precip l_0', 'temp_prom l_0','precip l_5', 'temp_prom l_5', 'provincia']
feature_names = ['latitud', 'longitud', 'precip l_0', 'temp_max l_0', 'temp_min l_0', 'temp_max l_2', 'temp_min l_2', 'precip l_2']
#numeric_feature_names = ['precip l_0', 'temp_prom l_0', 'precip l_5', 'temp_prom l_5']
numeric_feature_names = ['latitud', 'longitud', 'precip l_0', 'temp_max l_0', 'temp_min l_0', 'temp_max l_2', 'temp_min l_2', 'precip l_2']

#categor_feature_names = ['provincia']
categor_feature_names = []

outcome_name = ['TARGET']

#categorical_engineered_features=['provincia_SAN MARTIN',
# 'provincia_PICOTA',
# 'provincia_MOYOBAMBA',
# 'provincia_BELLAVISTA',
# 'provincia_RIOJA',
# 'provincia_TOCACHE']
categorical_engineered_features = []


@app.route('/api/predict', methods=['GET','POST'])
def api_predict():
    """API request
    """
    if request.method == 'POST':  #this block is only entered when the form is submitted
        
        #req_data = request.get_json () 
        #if not req_data:
        #    return jsonify(error="request body cannot be empty2"), 400
        #anho = req_data['anho']

        features = { 
            'anho': 2023
        }

        #Load the saved model
        print("Cargando modelos machine learning ...")
        model = cargarModeloSiEsNecesario()
        scaler = cargarScalerSiEsNecesario()
        

        

        '''new_data = pd.DataFrame([
        {'Name': 'M 3', 'precip l_0': 22.068888, 'temp_prom l_0': 26.891296, 'precip l_5': 0.056287, 'temp_prom l_5': 28.582603, 'provincia': 'SAN MARTIN'},
        {'Name': 'M 1', 'precip l_0': 2.736914, 'temp_prom l_0': 26.846750, 'precip l_5': 13.684924, 'temp_prom l_5': 23.525642, 'provincia': 'PICOTA'},
        {'Name': 'M 2', 'precip l_0': 2.736914, 'temp_prom l_0': 26.846750, 'precip l_5': 13.684924, 'temp_prom l_5': 23.525642, 'provincia': 'PICOTA'},
         
        ])'''
        new_data = pd.DataFrame([
    
        {'latitud': 354860, 'longitud': 9271177, 'precip l_0': 2.271762, 'temp_max l_0': 31.162016, 'temp_min l_0': 20.965277, 'temp_max l_2': 32.038571, 'temp_min l_2': 20.915900, 'precip l_2': 1.013100},
        {'latitud': 353419, 'longitud': 9271645, 'precip l_0': 0, 'temp_max l_0': 28.029667, 'temp_min l_0': 20.262200, 'temp_max l_2': 31.262200, 'temp_min l_2': 23.347366, 'precip l_2': 11.307649}

        ])
        print(new_data)

        ## data preparation
        prediction_features = new_data[feature_names]

        # scaling
        prediction_features[numeric_feature_names] = scaler.transform(prediction_features[numeric_feature_names])

        # engineering categorical variables
        prediction_features = pd.get_dummies(prediction_features, columns=categor_feature_names)

        # view feature set
        print(prediction_features)

        # add missing categorical feature columns
        current_categorical_engineered_features = set(prediction_features.columns) - set(numeric_feature_names)
        missing_features = set(categorical_engineered_features) - current_categorical_engineered_features
        for feature in missing_features:
            # add zeros since feature is absent in these data samples
            prediction_features[feature] = [0] * len(prediction_features) 


        # view final feature set
        print(prediction_features)

        ## predict using model
        predictions = model.predict(prediction_features)

        ## display results
        new_data['TARGET'] = predictions
        print(new_data)

        result = new_data.to_json(orient="records")
        parsed = json.loads(result)
        parsed2 = json.dumps(parsed, indent=4)

        print(parsed2)


        prediction = parsed 

        return jsonify( inputs=features,predictions=prediction)

    return '''User postman u otro cliente para ejecutar esta API REST'''

@app.route('/', methods=['GET','POST'])
def predict():
    """
    """
    if request.method == 'POST':  #this block is only entered when the form is submitted
        anho = request.form.get('anho')

        features = { 
            'anho': anho
        }

        #Load the saved model
        print("Cargando modelos machine learning ...")
        model = cargarModeloSiEsNecesario()
        scaler = cargarScalerSiEsNecesario()

        '''new_data = pd.DataFrame([
        {'Name': 'M 3', 'precip l_0': 22.068888, 'temp_prom l_0': 26.891296, 'precip l_5': 0.056287, 'temp_prom l_5': 28.582603, 'provincia': 'SAN MARTIN'},
        {'Name': 'M 1', 'precip l_0': 2.736914, 'temp_prom l_0': 26.846750, 'precip l_5': 13.684924, 'temp_prom l_5': 23.525642, 'provincia': 'PICOTA'},
        {'Name': 'M 2', 'precip l_0': 2.736914, 'temp_prom l_0': 26.846750, 'precip l_5': 13.684924, 'temp_prom l_5': 23.525642, 'provincia': 'PICOTA'},
         
        ])'''
        new_data = pd.DataFrame([
    
        {'latitud': 354860, 'longitud': 9271177, 'precip l_0': 2.271762, 'temp_max l_0': 31.162016, 'temp_min l_0': 20.965277, 'temp_max l_2': 32.038571, 'temp_min l_2': 20.915900, 'precip l_2': 1.013100},
        {'latitud': 353419, 'longitud': 9271645, 'precip l_0': 0, 'temp_max l_0': 28.029667, 'temp_min l_0': 20.262200, 'temp_max l_2': 31.262200, 'temp_min l_2': 23.347366, 'precip l_2': 11.307649}

        ])
        print(new_data)

        ## data preparation
        prediction_features = new_data[feature_names]

        # scaling
        prediction_features[numeric_feature_names] = scaler.transform(prediction_features[numeric_feature_names])

        # engineering categorical variables
        prediction_features = pd.get_dummies(prediction_features, columns=categor_feature_names)

        # view feature set
        print(prediction_features)

        # add missing categorical feature columns
        current_categorical_engineered_features = set(prediction_features.columns) - set(numeric_feature_names)
        missing_features = set(categorical_engineered_features) - current_categorical_engineered_features
        for feature in missing_features:
            # add zeros since feature is absent in these data samples
            prediction_features[feature] = [0] * len(prediction_features) 

        # view final feature set
        print(prediction_features)

        ## predict using model
        predictions = model.predict(prediction_features)

        ## display results
        new_data['TARGET'] = predictions
        print(new_data)

        result = new_data.to_json(orient="records")
        parsed = json.loads(result)
        parsed2 = json.dumps(parsed, indent=4)

        print(parsed2)


        prediction = parsed 

        return render_template("index.html", inputs=features, predictions=prediction)

    return render_template("index.html")


global_model = None


def cargarModeloSiEsNecesario():
    global global_model
    if global_model is not None:
        print('Modelo YA cargado')
        return global_model
    else:
        global_model = load(model_file_path) 
        print('Modelo cargado')
        return global_model


global_scaler = None
def cargarScalerSiEsNecesario():
    global global_scaler
    if global_scaler is not None:
        print('Scaler YA cargado')
        return global_scaler
    else:
        global_scaler = load(scaler_file_path) 
        print('Scaler cargado')
        return global_scaler

# puede eliminar desde esta línea en adelante
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify({'status': 'pong'})

@app.route('/<name>')
def hello_name(name):
    return "Hello {} {}!".format(name, sklearn.__version__)

if __name__ == "__main__":
    app.run()
    # app.run(debug=True)
