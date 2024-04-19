from flask import Flask, request, render_template
import numpy as np
import pickle
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Keras model

model = load_model("models/modelwqp.h5")
# Load the StandardScaler object
with open('models\scaler_params.pkl','rb') as f:
    d=pickle.load(f)
sc = StandardScaler()
sc.mean_ = d['mean']
sc.scale_ = d['scale']

# Define a function to preprocess input data
def preprocess_input(data):
    # Convert input dictionary to array and reshape it
    lis = ["ph","hardness","solids","chloramines","sulfates","conductivity","organic_carbon","trihalomethanes","turbidity"]
    features_arr = np.array([[float(data[key]) for key in lis]])
    # Perform any preprocessing steps here, like scaling
    # Example: features_arr_scaled = scaler.transform(features_arr)
    return features_arr

@app.route('/')
def index():
    return render_template('index.html', prediction="Enter Data to predict")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the request
    data = request.form.to_dict()
    # Preprocess input data
    processed_data = preprocess_input(data)
    # Fit the Preprocessed data
    processed_data = sc.transform(processed_data)
    # Predict water quality
    prediction = f"%.2f%% Safe Drinking Water"%(model.predict(processed_data)[0][0] * 100)
    # Render the result in result.html
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
