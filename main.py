from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the pkl file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        features = [float(x) for x in request.form.values()]
        # Convert to numpy array for the model
        features_array = np.array([features])
        # Make prediction
        prediction = model.predict(features_array)
        # Get the prediction result
        result = 'Cancer Risk: Positive' if prediction[0] == 1 else 'Cancer Risk: Negative'
    except Exception as e:
        result = str(e)

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
