// Load Bayesian Network structure
const nodes = new vis.DataSet([
  { id: 1, label: "StudyTimeWeekly" },
  { id: 2, label: "ParentalEducation" },
  { id: 3, label: "Absences" },
  { id: 4, label: "Tutoring" },
  { id: 5, label: "ParentalSupport" },
  { id: 6, label: "Extracurricular" },
  { id: 7, label: "GPA_bin" },
  { id: 8, label: "GradeClass" },
]);

const edges = new vis.DataSet([
  { from: 1, to: 7 },
  { from: 2, to: 7 },
  { from: 3, to: 7 },
  { from: 4, to: 7 },
  { from: 7, to: 8 },
  { from: 5, to: 8 },
  { from: 6, to: 8 },
]);

const container = document.getElementById("network");
const data = { nodes, edges };
const options = { physics: false };
const network = new vis.Network(container, data, options);

// Handle inference form submission
document
  .getElementById("inference-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    // Collect form data
    const formData = {
      StudyTimeWeekly: parseFloat(document.getElementById("studyTime").value),
      Absences: parseFloat(document.getElementById("absences").value),
      Gender: parseInt(document.getElementById("gender").value),
      Ethnicity: parseInt(document.getElementById("ethnicity").value),
      ParentalEducation: parseInt(
        document.getElementById("parentalEducation").value
      ),
      Tutoring: parseInt(document.getElementById("tutoring").value),
      ParentalSupport: parseInt(
        document.getElementById("parentalSupport").value
      ),
      Extracurricular: parseInt(
        document.getElementById("extracurricular").value
      ),
    };

    try {
      // Send data to backend
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const error = await response.json();
        document.getElementById(
          "inference-result"
        ).innerHTML = `<p style="color: red;">Error: ${error.error}</p>`;
        return;
      }

      // Display prediction results
      const result = await response.json();
      document.getElementById("inference-result").innerHTML = `
                <h3>Prediction Result</h3>
                <p><strong>GPA:</strong> ${result.gpa}</p>
                <p><strong>GPA Bin:</strong> ${result.gpa_bin_label}</p>
                <p><strong>Grade Class:</strong> ${result.gradeClass_label}</p>
            `;
    } catch (error) {
      document.getElementById(
        "inference-result"
      ).innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
  });
