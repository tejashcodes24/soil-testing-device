# Importing essential libraries and modules
from flask import Flask, render_template, request
import numpy as np
import pickle

# Loading the crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# Initialize Flask app
app = Flask(__name__)

# render home page
@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('home1.html', title=title)

# Route to render the crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='Harvestify - Crop Recommendation')

# Route to handle crop prediction
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Remove city and state dependency
        # Predict the crop
        data = np.array([[N, P, K,temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        # Render the result template with prediction
        return render_template('crop_result.html', prediction=final_prediction, title='Harvestify - Crop Recommendation')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
