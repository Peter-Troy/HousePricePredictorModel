from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (change the path to where you have stored it)
model = joblib.load('C:/JN/house_price_model.pkl')

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the user
        house_quality = int(request.form['house_quality'])
        living_area = float(request.form['living_area'])
        garage_cars = int(request.form['garage_cars'])
        basement_area = float(request.form['basement_area'])

        # Create a DataFrame with the user inputs
        specific_house = pd.DataFrame([[house_quality, living_area, garage_cars, basement_area]], 
                                      columns=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF'])

        # Predict the log price
        estimated_log_price = model.predict(specific_house)

        # Convert the log price back to the original scale
        estimated_price = np.exp(estimated_log_price).flatten()[0]

        return render_template('index.html', estimated_price=f"${estimated_price:,.2f}")
    except Exception as e:
        return render_template('index.html', error_message=f"Error: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
