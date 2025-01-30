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
print(type(random_forest_model))

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
        #sex = request.form['Sex']
        #us_performed = request.form['US_Performed']
        #print(us_performed)
        sex = 'male' if request.form['Sex'].strip().lower() == 'male' else 'female'
        us_performed = 'yes' if request.form['US_Performed'] == '1' else 'no'
        ipsilateral_rebound_tenderness = 'yes' if request.form['Ipsilateral_Rebound_Tenderness'] == '1' else 'no'
        lower_right_abd_pain = 'yes' if request.form['Lower_Right_Abd_Pain'] == '1' else 'no'
        coughing_pain = 'yes' if request.form['Coughing_Pain']== '1' else 'no'
        nausea = 'yes' if request.form['Nausea'] == '1' else 'no'
        migratory_pain = 'yes' if request.form['Migratory_Pain'] == '1' else 'no'

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


        print(us_performed)
        # Combine tabular data and image features
        cat_tabular_data = np.array([
            sex, migratory_pain, us_performed, ipsilateral_rebound_tenderness, lower_right_abd_pain, coughing_pain, nausea,    
        ])
        #categorical_features = ['Sex', 'Migratory_Pain', 'US_Performed', 'Ipsilateral_Rebound_Tenderness', 'Lower_Right_Abd_Pain', 'Coughing_Pain', 'Nausea']

        tabular_data = np.array([
            alvarado_score, body_temperature, pediatric_appendicitis_score, wbc_count,
            neutrophil_percentage, crp, bmi, height, weight
        ])
        #numerical_features = [ 'Alvarado_Score', 'Body_Temperature','Paedriatic_Appendicitis_Score', 'WBC_Count', 'Neutrophil_Percentage', 'CRP', 'BMI', 'Height', 'Weight'] + [col for col in all_columns if col.startswith('Image')]
        

       



        combined_features = np.concatenate([tabular_data, image_features_1, image_features_2, cat_tabular_data])

        print(preprocessor.get_feature_names_out().shape)
        print(combined_features.shape)
        # Ensure feature order is exactly the same as used in training
        column_order = [
            'Alvarado_Score', 'Body_Temperature', 'Paedriatic_Appendicitis_Score', 'WBC_Count',
            'Neutrophil_Percentage', 'CRP', 'BMI', 'Height', 'Weight'
        ] 

# Add dynamically generated image feature names (Image_1_Feature_1 to Image_2_Feature_2048)
        image_features_names = [f"Image_1_Feature_{i}" for i in range(1, 2049)] + \
                                           [f"Image_2_Feature_{i}" for i in range(1, 2049)]

        categorical_feature_names = [
                'Sex', 'Migratory_Pain', 'US_Performed', 'Ipsilateral_Rebound_Tenderness', 
                'Lower_Right_Abd_Pain', 'Coughing_Pain', 'Nausea'
        ]

# Combine all feature names in the same order as used in training
        expected_columns = column_order + image_features_names + categorical_feature_names

# Create DataFrame with correct column names
        combined_features_df = pd.DataFrame([combined_features], columns=expected_columns)

# Reindexing to ensure the correct order
        combined_features_df = combined_features_df.reindex(columns=expected_columns, fill_value=0)

# Preprocess the data
        processed_features = preprocessor.transform(combined_features_df)
        print(processed_features)
        combined_features_df_test = pd.DataFrame(processed_features, columns=expected_columns)
        combined_features_df_test.to_csv('D:/test_input.csv', index=False)


# Ensure the correct order

        print(combined_features_df)
        processed_features = preprocessor.transform(combined_features_df)
        print('Preprocessing completed')
        
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
