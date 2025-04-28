# Student Performance Prediction System

This project implements a Bayesian Network model to predict student academic performance metrics such as GPA and grade classification. It analyzes various factors including study habits, attendance, and support systems to provide actionable insights for improving academic outcomes.

## About the Program

### Model Overview

The system uses a Bayesian Network to model the probabilistic relationships between different factors that influence student performance. The model:

- Predicts GPA (0.0-4.0 scale) and grade classifications
- Identifies key factors influencing academic performance
- Provides personalized recommendations for improvement
- Handles missing data through probabilistic inference

### Key Features

- **Bayesian Network Model**: Captures complex dependencies between educational factors and performance metrics
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and normalizes numerical features
- **Performance Prediction**: Achieves a Mean Absolute Error (MAE) of approximately 0.27 for GPA predictions
- **Recommendation Engine**: Generates personalized improvement suggestions based on model insights
- **Intervention Analysis**: Quantifies the potential impact of specific actions (e.g., increasing study time)
- **Web Interface**: Provides an intuitive UI for inputting student data and viewing predictions
- **Cross-Validation**: Ensures model reliability with k-fold cross-validation (k=5)
- **Model Comparison**: Benchmarks against Random Forest and Linear Regression approaches

### Model Structure

The Bayesian Network models relationships between:

- Independent variables (ParentalEducation, Sports, Music, Volunteering)
- Intermediate variables (Extracurricular, Tutoring)
- Direct influences on GPA (ParentalSupport, Absences, StudyTimeWeekly, Tutoring, Extracurricular)
- Academic outcome (GPA → GradeClass)

## Installation Guide

### Prerequisites

- Python 3.7+
- Flask
- Modern web browser

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Verify Data Files

Ensure the following files exist in your project structure:

- `data/Cleaned_Student_performance_data.csv`
- `models/bayesian_model.pkl`
- `models/evaluation_metrics.pkl`
- `models/gpa_quantiles.pkl`

### Step 4: Launch the Application

Start the Flask server:

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000` by default.

## Usage Guide

### Making Predictions

1. Open the web interface by navigating to `http://127.0.0.1:5000` in your browser
2. Fill in the student information form with the following details:
   - Weekly study time (hours)
   - Number of absences
   - Gender
   - Ethnicity
   - Parental education level
   - Tutoring participation (yes/no)
   - Parental support (yes/no)
   - Extracurricular activities (yes/no)
3. Click "Predict" to generate performance predictions
4. View the results showing predicted GPA and grade classification

### Understanding Results

The system provides:

- **GPA Prediction**: Numerical value on a 0.0-4.0 scale
- **GPA Category**: Classification as VeryLow, Low, Medium, High, or VeryHigh
- **Grade Class**: Letter grade prediction (A, B, C, D, or F)
- **Recommendations**: Personalized suggestions to improve academic performance
- **Potential Impact**: Estimated GPA improvement for each recommendation

### API Endpoints

For programmatic access, the system provides the following REST API endpoints:

#### 1. Prediction Endpoint

```
POST /predict
```

**Input Format:**

```json
{
  "StudyTimeWeekly": 18,
  "Absences": 1,
  "Gender": 1,
  "Ethnicity": 1,
  "ParentalEducation": 1,
  "Tutoring": 1,
  "ParentalSupport": 1,
  "Extracurricular": 1,
  "Sports": 0,
  "Music": 1,
  "Volunteering": 0
}
```

**Output Format:**

```json
{
  "gpa": 3.35,
  "gpa_bin": 3,
  "gpa_bin_label": "High",
  "gradeClass": 1,
  "gradeClass_label": "B"
}
```

#### 2. Recommendation Endpoint

```
POST /recommend
```

Uses the same input format as `/predict` but returns improvement recommendations:

**Output Format:**

```json
{
  "gpa": 3.35,
  "gpa_category": "High",
  "recommendations": [
    {
      "factor": "Study Time",
      "issue": "Weekly study time below optimal level",
      "recommendation": "Increase study time to 15 hours per week",
      "impact": 0.45,
      "priority": "Medium"
    }
  ],
  "total_potential_improvement": 0.45,
  "potential_gpa": 3.8
}
```

## Advanced Usage

### Model Interpretation

- **Study Time Impact**: The most significant predictor of GPA, with optimal returns around 15 hours/week
- **Attendance**: Keeping absences below 5 days per term shows significant positive impact
- **Support Systems**: Parental support and tutoring provide additional positive effects

### Performance Metrics

- **MAE (GPA)**: 0.2668
- **RMSE (GPA)**: 0.3323
- **Accuracy (GradeClass)**: 91.65%
- **Precision (GradeClass)**: 91.79%
- **ROC AUC**: 0.9009
- **Calibration Error**: 0.2890

## Dependencies

The project uses the following Python libraries:

- `pandas` and `numpy`: Data manipulation
- `scikit-learn`: Machine learning components
- `networkx`: Bayesian Network visualization
- `matplotlib` and `seaborn`: Data visualization
- `flask`: Web API backend
- `joblib`: Model serialization

Install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Dataset: Cleaned_Student_performance_data.csv
- Bayesian Network implementation inspired by academic research on probabilistic graphical models.
