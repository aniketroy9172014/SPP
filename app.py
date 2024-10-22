from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the ML model
model = joblib.load('Mytracker.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Convert Gender to integer (male = 0, female = 1)
    gender_value = data['Gender']
    Gender = 0 if gender_value == 'Male' else 1

    # Convert ParentalEducation to integer
    education_value = data['ParentalEducation']
    education_mapping = {
        'None': 0,
        'High School': 1,
        'Some College': 2,
        'Bachelor\'s': 3,
        'Higher': 4
    }
    ParentalEducation = education_mapping.get(education_value, 0)

    # Convert Tutoring to integer (No = 0, Yes = 1)
    Tutoring = 1 if data['Tutoring'] == 'Yes' else 0

    # Convert ParentalSupport to integer
    support_value = data['ParentalSupport']
    support_mapping = {
        'None': 0,
        'Low': 1,
        'Moderate': 2,
        'High': 3,
        'Very High': 4
    }
    ParentalSupport = support_mapping.get(support_value, 0)

    # Convert Extracurricular to integer (No = 0, Yes = 1)
    Extracurricular = 1 if data['Extracurricular'] == 'Yes' else 0

    # Convert Sports to integer (No = 0, Yes = 1)
    Sports = 1 if data['Sports'] == 'Yes' else 0

    # Convert other form data to floats
    StudyTimeWeekly = float(data['StudyTimeWeekly'])
    Absences = float(data['Absences'])
    
    # Prepare input for model
    features = np.array([[Gender, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports]])

    # Predict
    prediction = abs(model.predict(features)[0])

    # Scale the prediction to be out of 10
    # Assuming your model predicts GPA out of 4.0, scale it accordingly
    scaled_prediction = (prediction / 4.0) * 10.0

    # Format the prediction to two decimal places
    formatted_prediction = "{:.2f}".format(scaled_prediction)

    return render_template('result.html', prediction=formatted_prediction)

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=False, host='0.0.0.0')
