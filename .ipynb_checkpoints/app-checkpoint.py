print("Starting Flask App")
 
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import pickle
import numpy as np
import cv2
import os
import torch
from torchvision import models, transforms


print('Libraries Loaded')
# Initialize the Flask application
app = Flask(__name__)
 
# Load the saved Random Forest model and preprocessor
with open('random_forest.pkl', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)
print(type(random_forest_model))
 
# Load the saved SVC model and preprocessor
with open('SVC.pkl', 'rb') as model_file:
    SVC_model = pickle.load(model_file)
print(type(SVC_model))
 
# Load the saved Logistic model and preprocessor
with open('model_pipeline.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
print(type(loaded_model))
 
with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)
print('Pickle Loaded')
 
# Load the pre-trained ResNet model for image feature extraction
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
model.eval()
 
print('Model Loaded')
# Define the transformation for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
# Function to extract image features
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image file")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).squeeze().numpy()
    return features
 
 
@app.route('/')
def index():
    return render_template('index.html')
 
# @app.route('/predict_random_forest', methods=['POST'])
# def predict():
# 🔹 Route for Random Forest Prediction
@app.route('/predict_random_forest', methods=['POST'])
def predict_random_forest():
    return predict_model(random_forest_model)
 
# 🔹 Route for SVC Prediction
@app.route('/predict_svc', methods=['POST'])
def predict_svc():
    return predict_model(SVC_model)
        # Get the form inputs
        #sex = request.form['Sex']
       # 🔹 Generalized Function for Model Prediction
    
    
def predict_model(model):
    try:
        sex = 'male' if request.form['Sex'].strip().lower() == 'male' else 'female'
        us_performed = 'yes' if request.form['US_Performed'].strip() == '1' else 'no'
        ipsilateral_rebound_tenderness = 'yes' if request.form['Ipsilateral_Rebound_Tenderness'].strip() == '1' else 'no'
        lower_right_abd_pain = 'yes' if request.form['Lower_Right_Abd_Pain'].strip() == '1' else 'no'
        coughing_pain = 'yes' if request.form['Coughing_Pain'].strip() == '1' else 'no'
        nausea = 'yes' if request.form['Nausea'].strip() == '1' else 'no'
        migratory_pain = 'yes' if request.form['Migratory_Pain'].strip() == '1' else 'no'
 
        alvarado_score = float(request.form['Alvarado_Score'])
        body_temperature = float(request.form['Body_Temperature'])
        pediatric_appendicitis_score = float(request.form['Paedriatic_Appendicitis_Score'])
        wbc_count = float(request.form['WBC_Count'])
        neutrophil_percentage = float(request.form['Neutrophil_Percentage'])
        crp = float(request.form['CRP'])
        bmi = float(request.form['BMI'])
        height = float(request.form['Height'])
        weight = float(request.form['Weight'])
 
        # Process the uploaded images
        image_file_1 = request.files['Image1']
        image_file_2 = request.files['Image2']
 
        if image_file_1 and image_file_2:
            image_path_1 = os.path.join('tmp', image_file_1.filename)
            image_file_1.save(image_path_1)
            image_features_1 = extract_features(image_path_1)
            os.remove(image_path_1)
 
            image_path_2 = os.path.join('tmp', image_file_2.filename)
            image_file_2.save(image_path_2)
            image_features_2 = extract_features(image_path_2)
            os.remove(image_path_2)
        else:
            raise ValueError("Two images must be uploaded")
 
        # Combine tabular data and image features
        cat_tabular_data = np.array([
            sex, migratory_pain, us_performed, ipsilateral_rebound_tenderness, lower_right_abd_pain, coughing_pain, nausea
        ])
 
        tabular_data = np.array([
            alvarado_score, body_temperature, pediatric_appendicitis_score, wbc_count,
            neutrophil_percentage, crp, bmi, height, weight
        ])
 
        combined_features = np.concatenate([tabular_data, image_features_1, image_features_2, cat_tabular_data])
 
        column_order = [
            'Alvarado_Score', 'Body_Temperature', 'Paedriatic_Appendicitis_Score', 'WBC_Count',
            'Neutrophil_Percentage', 'CRP', 'BMI', 'Height', 'Weight'
        ]
 
        image_features_names = [f"Image_1_Feature_{i}" for i in range(1, 2049)] + \
                               [f"Image_2_Feature_{i}" for i in range(1, 2049)]
 
        categorical_feature_names = [
            'Sex', 'Migratory_Pain', 'US_Performed', 'Ipsilateral_Rebound_Tenderness',
            'Lower_Right_Abd_Pain', 'Coughing_Pain', 'Nausea'
        ]
 
        expected_columns = column_order + image_features_names + categorical_feature_names
        combined_features_df = pd.DataFrame([combined_features], columns=expected_columns)
        combined_features_df = combined_features_df.reindex(columns=expected_columns, fill_value=0)
        
 
        processed_features = preprocessor.transform(combined_features_df)
        combined_features_df_test = pd.DataFrame(processed_features, columns=expected_columns)
        combined_features_df_test.to_csv('D:/test_input.csv', index=False)
 
        prediction = model.predict(processed_features)[0]
        result_message = (
            "The patient is likely to have appendicitis."
            if prediction == 1
            else "The patient is unlikely to have appendicitis."
        )
        return render_template('index.html', prediction_message=result_message)
 
    except Exception as e:
        return render_template('index.html', error_message=f"Error: {str(e)}")
   
    # 🔹 Route for Logistic Regression Prediction
    
@app.route("/predict_logistic_regression", methods=["POST"])

def predict_logistic_regression():
    # Loading the encoder
    with open('encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

# Loading the scaler
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    # Collect form data
        form_data = {
        'Sex': [request.form['Sex'].strip().lower()],
        'US_Performed': ['yes' if request.form['US_Performed'].strip() == '1' else 'no'],
        'Ipsilateral_Rebound_Tenderness': ['yes' if request.form['Ipsilateral_Rebound_Tenderness'].strip() == '1' else 'no'],
        'Lower_Right_Abd_Pain': ['yes' if request.form['Lower_Right_Abd_Pain'].strip() == '1' else 'no'],
        'Coughing_Pain': ['yes' if request.form['Coughing_Pain'].strip() == '1' else 'no'],
        'Nausea': ['yes' if request.form['Nausea'].strip() == '1' else 'no'],
        'Migratory_Pain': ['yes' if request.form['Migratory_Pain'].strip() == '1' else 'no'],
        'Alvarado_Score': [float(request.form['Alvarado_Score'])],
        'Body_Temperature': [float(request.form['Body_Temperature'])],
        'WBC_Count': [float(request.form['WBC_Count'])],
        'CRP': [float(request.form['CRP'])],
        'BMI': [float(request.form['BMI'])]
    }



    df = pd.DataFrame(form_data)
    print('Right after input')
    print(df)
    
    def categorize_temperature(temp):
        if 35 <= temp < 36:
            return "Low(35-36)"
        elif 36 <= temp < 37:
            return "Normal(36-37)"
        elif 37 <= temp < 38:
            return "High(37-38)"
        else:
            return "Fever(38-40)"


    def categorize_wbc(wbc):
        if 0 <= wbc < 4.5:
            return "Low(0-4.5)"
        elif 4.5 <= wbc < 11:
            return "Normal(4.5-11)"
        elif 11 <= wbc < 20:
            return "High(11-20)"
        else:
            return "Out of Range"

    def categorize_crp(crp):
        if 0 <= crp < 10:
            return "Low"
        elif 10 <= crp < 20:
            return "Normal"
        elif 20 <= crp < 40:
            return "High"
        elif 40 <= crp <= 100:
            return "Very High"
        else:
            return "Out of Range"

    def categorize_alvarado(alvarado):
        if 0 <= alvarado < 4:
            return "Very Low(0-4)"
        elif 4 <= alvarado < 6:
            return "Low(4-6)"
        elif 6 <= alvarado < 8:
            return "Medium(6-8)"
        elif 8 <= alvarado <= 10:
            return "High(8-10)"
        else:
            return "Out of Range"

    def categorize_bmi(bmi):
        if 0 <= bmi < 18.5:
            return "Underweight(0-18.5)"
        elif 18.5 <= bmi < 24.9:
            return "Normal weight(18.5-24.9)"
        elif 24.9 <= bmi < 29.9:
            return "Overweight(24.9-29.9)"
        elif 29.9 <= bmi <= 40:
            return "Obesity(29.9-40)"
        else:
            return "Out of Range"

    # Convert form data to DataFrame


    # Categorize continuous variables
    df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
    df['CRP_Category'] = df['CRP'].apply(categorize_crp)
    df['WBC_Count_Category'] = df['WBC_Count'].apply(categorize_wbc)
    df['Temperature_Category'] = df['Body_Temperature'].apply(categorize_temperature)
    df['Alvarado_Score_Category'] = df['Alvarado_Score'].apply(categorize_alvarado)
    categorical_features = ['Sex', 'BMI_Category', 'CRP_Category', 'WBC_Count_Category', 'Temperature_Category', 'Alvarado_Score_Category', 'Migratory_Pain', 'US_Performed', 'Ipsilateral_Rebound_Tenderness', 'Lower_Right_Abd_Pain', 'Coughing_Pain', 'Nausea']
    # Select features
    df = df[['Sex', 'BMI_Category', 'CRP_Category', 'WBC_Count_Category', 'Temperature_Category', 'Alvarado_Score_Category', 'Migratory_Pain', 'US_Performed', 'Ipsilateral_Rebound_Tenderness', 'Lower_Right_Abd_Pain', 'Coughing_Pain', 'Nausea']]
    print(df)
    preprocessor_logistics =loaded_model.named_steps['preprocessor']
    prediction = loaded_model.predict(df)
    transformed_data = preprocessor_logistics.transform(df)
    
# Print the transformed data (encoded & scaled)
    print("🔄 Transformed Data (Encoded & Standardized):")
    print(transformed_data)

    # Generate result message
    result_message = "The patient is likely to have appendicitis." if prediction == 1 else "The patient is unlikely to have appendicitis."
    return render_template("index.html", prediction_message=result_message)

 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)