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
feature_names = ['temp_max l_4', 'precip l_5', 'temp_max l_2', 'temp_min l_1', 'temp_prom l_3', 
                 'temp_min l_3', 'temp_min l_4', 'temp_min l_2', 'temp_prom l_2',
                'provincia']
numeric_feature_names = ['temp_max l_4', 'precip l_5', 'temp_max l_2', 'temp_min l_1',
       'temp_prom l_3', 'temp_min l_3', 'temp_min l_4', 'temp_min l_2',
       'temp_prom l_2']
categor_feature_names = ['provincia']
outcome_name = ['TARGET']

categorical_engineered_features=['provincia_SAN MARTIN',
 'provincia_PICOTA',
 'provincia_MOYOBAMBA',
 'provincia_BELLAVISTA',
 'provincia_RIOJA',
 'provincia_TOCACHE']


@app.route('/api/predict', methods=['GET','POST'])
def api_predict():
    """API request
    """
    if request.method == 'POST':  #this block is only entered when the form is submitted
        
        req_data = request.get_json () 
        if not req_data:
            return jsonify(error="request body cannot be empty"), 400
        anho = req_data['anho']

        features = { 
            'anho': anho
        }

        #Load the saved model
        print("Cargando modelos machine learning ...")
        model = cargarModeloSiEsNecesario()
        scaler = cargarScalerSiEsNecesario()
        

        

        new_data = pd.DataFrame([
        {'temp_max l_4': 27.303525, 'precip l_5': 5.453901, 'temp_max l_2': 33.722817, 'temp_min l_1': 21.857262, 'temp_prom l_3': 25.483028, 'temp_min l_3': 21.190212, 'temp_min l_4': 21.686244, 'temp_min l_2': 21.037416, 'temp_prom l_2': 27.380116, 'provincia': 'BELLAVISTA' },
{'temp_max l_4': 30.221358, 'precip l_5': 22.944548, 'temp_max l_2': 31.381693, 'temp_min l_1': 22.960336, 'temp_prom l_3': 27.196949, 'temp_min l_3': 23.375591, 'temp_min l_4': 22.978642, 'temp_min l_2': 23.566438  , 'temp_prom l_2': 27.474066, 'provincia': 'SAN MARTIN' },
{'temp_max l_4': 30.876208, 'precip l_5': 3.793558, 'temp_max l_2': 31.280876, 'temp_min l_1': 20.371897, 'temp_prom l_3': 25.245815, 'temp_min l_3': 19.853952, 'temp_min l_4': 20, 'temp_min l_2': 20, 'temp_prom l_2': 20, 'provincia': 'RIOJA' },
{'temp_max l_4': 29, 'precip l_5': 0.6, 'temp_max l_2': 28, 'temp_min l_1': 20, 'temp_prom l_3': 24, 'temp_min l_3': 20, 'temp_min l_4': 20, 'temp_min l_2': 20, 'temp_prom l_2': 24, 'provincia': 'MOYOBAMBA' },

{'temp_max l_4': 27.303525, 'precip l_5': 5.453901, 'temp_max l_2': 33.722817, 'temp_min l_1': 21.857262, 'temp_prom l_3': 25.483028, 'temp_min l_3': 21.190212, 'temp_min l_4': 21.686244, 'temp_min l_2': 21.037416, 'temp_prom l_2': 27.380116, 'provincia': 'PICOTA' },
{'temp_max l_4': 27.303525, 'precip l_5': 5.453901, 'temp_max l_2': 33.722817, 'temp_min l_1': 21.857262, 'temp_prom l_3': 25.483028, 'temp_min l_3': 21.190212, 'temp_min l_4': 21.686244, 'temp_min l_2': 21.037416, 'temp_prom l_2': 27.380116, 'provincia': 'TOCACHE' },
  

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

        new_data = pd.DataFrame([
      {'temp_max l_4': 27.303525, 'precip l_5': 5.453901, 'temp_max l_2': 33.722817, 'temp_min l_1': 21.857262, 'temp_prom l_3': 25.483028, 'temp_min l_3': 21.190212, 'temp_min l_4': 21.686244, 'temp_min l_2': 21.037416, 'temp_prom l_2': 27.380116, 'provincia': 'BELLAVISTA' },
{'temp_max l_4': 30.221358, 'precip l_5': 22.944548, 'temp_max l_2': 31.381693, 'temp_min l_1': 22.960336, 'temp_prom l_3': 27.196949, 'temp_min l_3': 23.375591, 'temp_min l_4': 22.978642, 'temp_min l_2': 23.566438  , 'temp_prom l_2': 27.474066, 'provincia': 'SAN MARTIN' },
{'temp_max l_4': 30.876208, 'precip l_5': 3.793558, 'temp_max l_2': 31.280876, 'temp_min l_1': 20.371897, 'temp_prom l_3': 25.245815, 'temp_min l_3': 19.853952, 'temp_min l_4': 20, 'temp_min l_2': 20, 'temp_prom l_2': 20, 'provincia': 'RIOJA' },
{'temp_max l_4': 29, 'precip l_5': 0.6, 'temp_max l_2': 28, 'temp_min l_1': 20, 'temp_prom l_3': 24, 'temp_min l_3': 20, 'temp_min l_4': 20, 'temp_min l_2': 20, 'temp_prom l_2': 24, 'provincia': 'MOYOBAMBA' },

{'temp_max l_4': 27.303525, 'precip l_5': 5.453901, 'temp_max l_2': 33.722817, 'temp_min l_1': 21.857262, 'temp_prom l_3': 25.483028, 'temp_min l_3': 21.190212, 'temp_min l_4': 21.686244, 'temp_min l_2': 21.037416, 'temp_prom l_2': 27.380116, 'provincia': 'PICOTA' },
{'temp_max l_4': 27.303525, 'precip l_5': 5.453901, 'temp_max l_2': 33.722817, 'temp_min l_1': 21.857262, 'temp_prom l_3': 25.483028, 'temp_min l_3': 21.190212, 'temp_min l_4': 21.686244, 'temp_min l_2': 21.037416, 'temp_prom l_2': 27.380116, 'provincia': 'TOCACHE' },
    
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
