from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Logging setup
logging.basicConfig(level=logging.INFO)

# Set up detailed logging to file for debugging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = logging.FileHandler('logs/app_debug.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

# Load model components
model_data = joblib.load('models/bayesian_model.pkl')
CPT = model_data['CPT']
parents = model_data['parents']
encoders = model_data['label_encoders']
features = model_data['features']
simplified_predictors = model_data.get('simplified_predictors', {})

# Load GPA quantiles instead of mean_gpa
gpa_quantiles = joblib.load('models/gpa_quantiles.pkl')

# Load sample of training data for fallback mechanisms
try:
    train_df = pd.read_csv('data/Cleaned_Student_performance_data.csv')
    train_df = train_df.sample(min(1000, len(train_df)))  # Use a subset for efficiency
    logging.info(f"Loaded training data sample: {train_df.shape}")
    
    # Apply same preprocessing as in training
    gpa_bins   = [0, 1.0, 2.0, 3.0, 4.0]
    gpa_labels = ['VeryLow', 'Low', 'Medium', 'High']
    train_df['GPA_bin'] = pd.cut(train_df['GPA'], bins=gpa_bins, labels=gpa_labels, include_lowest=True)
    
    study_bins   = [0, 5, 10, 15, 20, float('inf')]
    study_labels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
    train_df['Study_bin'] = pd.cut(train_df['StudyTimeWeekly'], bins=study_bins, labels=study_labels, include_lowest=True)
    
    abs_bins   = [0, 5, 10, 15, 20, float('inf')]
    abs_labels = ['None', 'Few', 'Moderate', 'High', 'VeryHigh']
    train_df['Absences_bin'] = pd.cut(train_df['Absences'], bins=abs_bins, labels=abs_labels, include_lowest=True)
    
    # Encode categorical variables using the same encoders from the model
    for col in encoders:
        if col in train_df.columns and col in encoders:
            try:
                train_df[col] = encoders[col].transform(train_df[col])
            except:
                logging.warning(f"Could not transform column {col}")
    
except Exception as e:
    logging.warning(f"Could not load training data: {e}")
    # Create a minimal dataframe with default values if loading fails
    train_df = pd.DataFrame({
        'StudyTimeWeekly': [10.0],
        'Absences': [5.0],
        'ParentalSupport': [2],
        'GPA': [2.5]
    })

# Binning setup for continuous features
gpa_bins = [0, 1.0, 2.0, 3.0, 4.0]
gpa_labels = ['VeryLow', 'Low', 'Medium', 'High']
study_bins = [0, 5, 10, 15, 20, float('inf')]
study_labels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
abs_bins = [0, 5, 10, 15, 20, float('inf')]
abs_labels = ['None', 'Few', 'Moderate', 'High', 'VeryHigh']

# Helper: bin continuous value
def bin_value(x, bins, labels):
    for i in range(len(bins) - 1):
        if bins[i] <= x < bins[i + 1]:
            return labels[i]
    return labels[-1]

# Helper: Process input data
def process_input(data):
    evidence = {}
    
    # Get StudyTimeWeekly as direct value
    if 'StudyTimeWeekly' in data:
        evidence['StudyTimeWeekly'] = float(data.get('StudyTimeWeekly', 0))
        
        # Also create Study_bin for nodes that might use it
        study_hours = evidence['StudyTimeWeekly']
        study_bin = bin_value(study_hours, study_bins, study_labels)
        if 'Study_bin' in encoders:
            evidence['Study_bin'] = int(encoders['Study_bin'].transform([study_bin])[0])
    
    # Get Absences as direct value
    if 'Absences' in data:
        evidence['Absences'] = float(data.get('Absences', 0))
        
        # Also create Absences_bin for nodes that might use it
        absences = evidence['Absences']
        abs_bin = bin_value(absences, abs_bins, abs_labels)
        if 'Absences_bin' in encoders:
            evidence['Absences_bin'] = int(encoders['Absences_bin'].transform([abs_bin])[0])
    
    # Direct categorical features
    for feature in ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 
                    'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']:
        if feature in data:
            evidence[feature] = int(data[feature])
    
    return evidence

# Helper: Predict GPA using the improved approach
def predict_gpa(evidence):
    # Get the relevant parent values for GPA based on the reduced parent set from training
    important_features = ['StudyTimeWeekly', 'Absences', 'ParentalSupport']
    parent_values = []
    
    logging.info(f"Evidence for GPA prediction: {evidence}")
    
    # Get values for each parent, with proper type handling
    for p in important_features:
        if p in evidence:
            # Make sure we're using the correct type
            if isinstance(evidence[p], (int, float)):
                parent_values.append(evidence[p])
            else:
                parent_values.append(float(evidence[p]))
        else:
            # Default to a common value if missing
            logging.warning(f"Missing parent {p} in evidence, using default")
            if p in ['StudyTimeWeekly', 'Absences']:
                parent_values.append(float(train_df[p].median()))
            else:
                parent_values.append(int(train_df[p].mode()[0]))
    
    key = tuple(parent_values)
    logging.info(f"Looking up key in CPT: {key}")
    
    # Try direct CPT lookup
    if 'GPA' in CPT and key in CPT['GPA']:
        probs = CPT['GPA'][key]
        logging.info(f"Found direct match in CPT with probabilities: {probs}")
        predicted = max(probs, key=probs.get)
        return float(predicted)
    
    logging.info("No direct match in CPT, trying nearest neighbor lookup...")
    
    # If not found, try to find closest match (nearest neighbor approach)
    if 'GPA' in CPT and len(CPT['GPA']) > 0:
        min_distance = float('inf')
        best_key = None
        
        for existing_key in CPT['GPA'].keys():
            if len(existing_key) != len(key):
                continue
                
            # Calculate Euclidean distance between existing key and our key
            distance = sum((float(a)-float(b))**2 for a, b in zip(existing_key, key))
            
            if distance < min_distance:
                min_distance = distance
                best_key = existing_key
        
        if best_key is not None and min_distance < 100:  # Threshold to avoid very distant matches
            logging.info(f"Using nearest neighbor key: {best_key} with distance: {min_distance}")
            probs = CPT['GPA'][best_key]
            predicted = max(probs, key=probs.get)
            return float(predicted)
    
    # If still no match, try one feature at a time
    logging.info("Trying single-feature predictors...")
    
    for i, p in enumerate(important_features):
        predictor_key = ('GPA', p)
        if predictor_key in simplified_predictors:
            p_value = parent_values[i]
            # Find closest value in simplified predictors
            closest_value = None
            min_diff = float('inf')
            
            for existing_value in simplified_predictors[predictor_key].keys():
                try:
                    diff = abs(float(existing_value) - float(p_value))
                    if diff < min_diff:
                        min_diff = diff
                        closest_value = existing_value
                except (ValueError, TypeError):
                    continue
            
            if closest_value is not None and closest_value in simplified_predictors[predictor_key]:
                simple_probs = simplified_predictors[predictor_key][closest_value]
                if simple_probs:
                    logging.info(f"Using simplified predictor for {p} with value {closest_value}")
                    predicted = max(simple_probs, key=simple_probs.get)
                    return float(predicted)
    
    # Try looking at the training data distribution
    logging.info("Using training data distribution for prediction")
    
    # Predict based on StudyTimeWeekly (strongest predictor)
    if 'StudyTimeWeekly' in evidence:
        study_time = evidence['StudyTimeWeekly']
        if study_time >= 15:  # High study time
            return 3.5  # Good GPA
        elif study_time >= 10:  # Medium study time
            return 3.0  # Average to good GPA
        elif study_time >= 5:   # Low study time
            return 2.5  # Average GPA
        else:  # Very low study time
            return 2.0  # Below average GPA
    
    # Final fallback: use the mean GPA
    logging.warning("All prediction methods failed, using mean GPA")
    return float(train_df['GPA'].mean())

# Helper: Predict GradeClass using the improved approach
def predict_gradeclass(evidence):
    # For GradeClass, parent is GPA
    if 'GPA' not in evidence:
        # If GPA is not available, predict it first
        gpa_value = predict_gpa(evidence)
        evidence['GPA'] = gpa_value
    else:
        gpa_value = float(evidence['GPA'])
    
    logging.info(f"Predicting GradeClass with GPA: {gpa_value}")
    
    # Try to find closest GPA value in CPT
    if 'GradeClass' in CPT:
        # Find closest GPA key
        closest_gpa = None
        min_distance = float('inf')
        
        for key_tuple in CPT['GradeClass'].keys():
            if not isinstance(key_tuple, tuple) or len(key_tuple) != 1:
                continue
            
            try:
                key_gpa = float(key_tuple[0])
                distance = abs(key_gpa - gpa_value)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_gpa = key_tuple
            except (ValueError, TypeError):
                continue
        
        if closest_gpa is not None and min_distance < 0.5:  # Threshold for GPA distance
            logging.info(f"Using closest GPA key: {closest_gpa} with distance: {min_distance}")
            probs = CPT['GradeClass'][closest_gpa]
            return int(max(probs, key=probs.get))
    
    # If no direct mapping, use a simple rule-based approach as fallback
    logging.info("Using rule-based GradeClass prediction")
    if gpa_value >= 3.5:
        return 0  # A/Excellent
    elif gpa_value >= 3.0:
        return 1  # B/Good
    elif gpa_value >= 2.0:
        return 2  # C/Average
    elif gpa_value >= 1.0:
        return 3  # D/Below Average
    else:
        return 4  # F/Poor

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from JSON request
        data = request.json
        logging.info(f"Received prediction request: {data}")
        
        # Validate input
        required_fields = [
            'StudyTimeWeekly', 'Absences', 'ParentalEducation',
            'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400
        
        # Process input data
        evidence = process_input(data)
        logging.info(f"Processed evidence: {evidence}")
        
        # Predict GPA directly
        gpa_value = predict_gpa(evidence)
        evidence['GPA'] = gpa_value
        logging.info(f"Predicted GPA: {gpa_value}")
        
        # Predict GradeClass based on GPA
        grade_class = predict_gradeclass(evidence)
        logging.info(f"Predicted GradeClass: {grade_class}")
        
        # Map GradeClass to label
        grade_label = "Unknown"
        try:
            grade_label = encoders['GradeClass'].inverse_transform([grade_class])[0]
        except (ValueError, KeyError, IndexError) as e:
            logging.error(f"Error mapping GradeClass: {str(e)}")
            # Use a simple mapping as fallback
            grade_labels = ["Excellent", "Good", "Average", "Below Average", "Poor"]
            if 0 <= grade_class < len(grade_labels):
                grade_label = grade_labels[grade_class]

        # Discretize GPA for display purposes
        gpa_bin = None
        gpa_bin_label = "Unknown"
        if 0 <= gpa_value < 1.0:
            gpa_bin = 0
            gpa_bin_label = "VeryLow"
        elif 1.0 <= gpa_value < 2.0:
            gpa_bin = 1
            gpa_bin_label = "Low"
        elif 2.0 <= gpa_value < 3.0:
            gpa_bin = 2
            gpa_bin_label = "Medium"
        elif 3.0 <= gpa_value <= 4.0:
            gpa_bin = 3
            gpa_bin_label = "High"

        # Prepare result
        result = {
            'gpa': round(float(gpa_value), 2),
            'gpa_bin': int(gpa_bin) if gpa_bin is not None else None,
            'gpa_bin_label': gpa_bin_label,
            'gradeClass': int(grade_class),
            'gradeClass_label': str(grade_label)
        }
        logging.info(f"Prediction result: {result}")
        
        return jsonify(result)
        
    except KeyError as e:
        logging.error(f"KeyError: {str(e)}")
        return jsonify({"error": f"Missing key in model: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "API is running"})

# Add a model info endpoint
@app.route('/model-info', methods=['GET'])
def model_info():
    try:
        # Get basic model information to share
        info = {
            "features": features,
            "target": "GPA and GradeClass",
            "model_type": "Bayesian Network",
            "parents_structure": {k: v for k, v in parents.items()},
            "evaluation_metrics": {
                "GPA_MAE": float(joblib.load('models/evaluation_metrics_new_structure.pkl').get('MAE_GPA', "N/A")),
                "GradeClass_Accuracy": float(joblib.load('models/evaluation_metrics_new_structure.pkl').get('Accuracy_GradeClass', "N/A")),
            }
        }
        return jsonify(info)
    except Exception as e:
        logging.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)