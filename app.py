from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from json import JSONEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Logging
logging.basicConfig(level=logging.INFO)

# Numpy JSON Encoder
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = NumpyEncoder

# Load model components
model_data = joblib.load('models/bayesian_model.pkl')
CPT = model_data['CPT']
parents = model_data['parents']
encoders = model_data['label_encoders']
features = model_data['features']
mean_gpa = joblib.load('models/mean_gpa.pkl')

# Binning setup
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

# Bayesian node prediction
def predict_node(node, evidence):
    node_parents = parents.get(node, [])
    if not node_parents:
        return int(max(CPT[node], key=CPT[node].get))
    
    key = tuple(evidence[p] for p in node_parents)
    probs = CPT[node].get(key)
    if not probs:
        probs = next(iter(CPT[node].values()))
    
    return int(max(probs, key=probs.get))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    logging.info(f"Incoming data: {data}")

    # Validate fields
    required = ['StudyTimeWeekly', 'Absences',
                'Gender', 'Ethnicity', 'ParentalEducation',
                'Tutoring', 'ParentalSupport', 'Extracurricular']
    missing = [field for field in required if field not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # Encode evidence
    evidence = {}

    # Discretize StudyTimeWeekly
    try:
        study_hours = float(data['StudyTimeWeekly'])
        study_bin = bin_value(study_hours, study_bins, study_labels)
        evidence['Study_bin'] = int(encoders['Study_bin'].transform([study_bin])[0])
    except Exception:
        return jsonify({'error': 'Invalid StudyTimeWeekly value'}), 400

    # Discretize Absences
    try:
        absences = float(data['Absences'])
        abs_bin = bin_value(absences, abs_bins, abs_labels)
        evidence['Absences_bin'] = int(encoders['Absences_bin'].transform([abs_bin])[0])
    except Exception:
        return jsonify({'error': 'Invalid Absences value'}), 400

    # Encode categorical fields
    for f in ['Gender', 'Ethnicity', 'ParentalEducation',
              'Tutoring', 'ParentalSupport', 'Extracurricular']:
        try:
            evidence[f] = int(encoders[f].transform([data[f]])[0])
        except Exception:
            return jsonify({'error': f'Invalid value for {f}: {data[f]}'}), 400

    # Predict GPA_bin
    gpa_bin = predict_node('GPA_bin', evidence)
    evidence['GPA_bin'] = gpa_bin

    # Predict GradeClass
    grade_class = predict_node('GradeClass', evidence)

    # Decode outputs
    gpa_label = encoders['GPA_bin'].inverse_transform([gpa_bin])[0]
    grade_label = encoders['GradeClass'].inverse_transform([grade_class])[0]
    gpa_value = float(mean_gpa.get(gpa_bin, np.mean(list(mean_gpa.values()))))

    return jsonify({
        'gpa': round(gpa_value, 2),
        'gpa_bin': int(gpa_bin),
        'gpa_bin_label': str(gpa_label),
        'gradeClass': int(grade_class),
        'gradeClass_label': str(grade_label)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)