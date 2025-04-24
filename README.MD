# Student Performance Prediction System

This project predicts student performance metrics such as GPA and GradeClass using a Bayesian Network. It also includes data preprocessing, visualization, and evaluation of the model's performance.

---

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, and normalizes numerical features.
- **Bayesian Network**: Predicts GPA and GradeClass based on input features.
- **Visualization**: Displays data distributions, correlation matrices, and Bayesian Network structure.
- **Evaluation**: Provides metrics such as MAE, RMSE, and accuracy for model evaluation.

---

## Project Structure

```
app.py                              # Flask application for predictions
cleaning_data.ipynb                 # Notebook for data cleaning
student_performance_predict.ipynb   # Notebook for model training and evaluation
model.ipynb                         # Notebook for Bayesian Network model
README.MD                           # Project documentation
requirements.txt                    # Python dependencies
data/
    Cleaned_Student_performance_data.csv  # Dataset
models/
    bayesian_model.pkl              # Trained Bayesian Network model
    evaluation_metrics.pkl          # Evaluation metrics
    mean_gpa.pkl                    # Mean GPA per GPA_bin
UI/
    index.html                      # Frontend HTML
    script.js                       # Frontend JavaScript
    style.css                       # Frontend CSS
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

---

## Usage

1. Open the web interface by navigating to `http://127.0.0.1:5000` in your browser.
2. Input student data to predict GPA and GradeClass.
3. View visualizations of the Bayesian Network and model evaluation metrics.

---

## Input Format

The system expects input in the following JSON format:

```json
{
  "StudyTimeWeekly": 12,
  "Absences": 1,
  "Gender": 1,
  "Ethnicity": 3,
  "ParentalEducation": 3,
  "Tutoring": 0,
  "ParentalSupport": 1,
  "Extracurricular": 1
}
```

---

## Output

The system provides predictions in the following format:

```json
{
  "Predicted_GPA_bin": "Medium",
  "Predicted_GradeClass": "B"
}
```

---

## Dependencies

The project uses the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `networkx`
- `matplotlib`
- `seaborn`
- `joblib`

Install all dependencies using the `requirements.txt` file.
pip install -r requirements.txt

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- Dataset: [Cleaned_Student_performance_data.csv](data/Cleaned_Student_performance_data.csv)
- Bayesian Network implementation inspired by academic research on probabilistic graphical models.
