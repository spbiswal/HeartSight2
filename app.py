from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model from the file
model_filename = 'best_random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Print feature names used during model training
print(f"Feature names during model training: {loaded_model.feature_names_in_}")

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the request
    input_data = request.form.to_dict()

    # Convert input data to appropriate types
    input_data = {k: float(v) if k in ['age', 'sysBP', 'totChol', 'BMI', 'heartRate', 'glucose'] else int(v) for k, v in input_data.items()}

    print(input_data)

    # Make prediction using the loaded model
    prediction = predict_user_input(loaded_model, input_data)

    # Return the prediction result
    return render_template('result.html', prediction=prediction)

# Function to make prediction using the loaded model
def predict_user_input(model, user_input_dict):
    user_input_df = pd.DataFrame([user_input_dict])

    # Check and print the columns of the user input
    print(f"User input columns: {user_input_df.columns}")

    # Ensure the input DataFrame has the same columns as the model's training data
    user_input_df = user_input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    prediction = model.predict(user_input_df)
    return prediction[0]

if __name__ == '__main__':
   app.run(debug=True)