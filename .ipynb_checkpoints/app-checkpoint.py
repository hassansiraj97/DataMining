print("Starting Flask App")

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form inputs
        sex = request.form['Sex']
        alvarado_score = float(request.form['Alvarado_Score'])
        body_temperature = float(request.form['Body_Temperature'])
        pediatric_appendicitis_score = float(request.form['Paedriatic_Appendicitis_Score'])
        wbc_count = float(request.form['WBC_Count'])
        neutrophil_percentage = float(request.form['Neutrophil_Percentage'])
        crp = float(request.form['CRP'])
        bmi = float(request.form['BMI'])
        height = float(request.form['Height'])
        weight = float(request.form['Weight'])
        us_performed = int(request.form['US_Performed'])
        ipsilateral_rebound_tenderness = int(request.form['Ipsilateral_Rebound_Tenderness'])
        lower_right_abd_pain = int(request.form['Lower_Right_Abd_Pain'])
        coughing_pain = int(request.form['Coughing_Pain'])
        nausea = int(request.form['Nausea'])
        migratory_pain = int(request.form['Migratory_Pain'])

        # Process the uploaded images
        image_file_1 = request.files['Image1']
        image_file_2 = request.files['Image2']

        if image_file_1 and image_file_2:
            # Save and extract features from the first image
            image_path_1 = os.path.join('tmp', image_file_1.filename)
            image_file_1.save(image_path_1)
            image_features_1 = extract_features(image_path_1)
            os.remove(image_path_1)  # Clean up the uploaded image

            # Save and extract features from the second image
            image_path_2 = os.path.join('tmp', image_file_2.filename)
            image_file_2.save(image_path_2)
            image_features_2 = extract_features(image_path_2)
            os.remove(image_path_2)  # Clean up the uploaded image
        else:
            raise ValueError("Two images must be uploaded")



        # Combine tabular data and image features
        tabular_data = np.array([
            1 if sex.lower() == 'male' else 0,
            alvarado_score, body_temperature, pediatric_appendicitis_score, wbc_count,
            neutrophil_percentage, crp, bmi, height, weight, us_performed,
            ipsilateral_rebound_tenderness, lower_right_abd_pain, coughing_pain, nausea,
            migratory_pain
        ])
        
        print("Shape of tabular data:", tabular_data.shape)
        print("Shape of image features 1:", image_features_1.shape)
        print("Shape of image features 2:", image_features_2.shape)
       



        combined_features = np.concatenate([tabular_data, image_features_1, image_features_2])
        print("Shape of combined features before preprocessing:", combined_features.shape)
        print("Expected number of columns from preprocessor:", len(preprocessor.get_feature_names_out()))
        print("Columns in combined features:", combined_features.columns.tolist())
        print("Columns expected by preprocessor:", preprocessor.get_feature_names_out())

        # Preprocess the combined features
        combined_features = pd.DataFrame([combined_features], columns=preprocessor.get_feature_names_out())
        processed_features = preprocessor.transform(combined_features)
        
        
        # Make prediction
        prediction = random_forest_model.predict(processed_features)[0]
        result_message = (
            "The patient is likely to have appendicitis."
            if prediction == 1
            else "The patient is unlikely to have appendicitis."
        )
        return render_template('index.html', prediction_message=result_message)

    except Exception as e:
        return render_template('index.html', error_message=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
