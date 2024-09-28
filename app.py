from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model.pk1')

@app.route('/')
def home():
    return render_template('index.html')  # Render the form

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Convert data to DataFrame for prediction
    cust_df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(cust_df)

    # Format the output
    cost_pred = prediction[0]
    
    # Return the result as JSON
    return jsonify({'cost': f"The medical insurance cost of the new customer is: {cost_pred:.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
