// Debug toggle functionality
document.getElementById('toggleDebug').addEventListener('click', function() {
  const debugPanel = document.getElementById('debugPanel');
  if (debugPanel.style.display === 'block') {
      debugPanel.style.display = 'none';
      this.textContent = 'Show Debug Info';
  } else {
      debugPanel.style.display = 'block';
      this.textContent = 'Hide Debug Info';
  }
});

// Debug log function
function debugLog(message, data = null) {
  const debugInfo = document.getElementById('debugInfo');
  const timestamp = new Date().toLocaleTimeString();
  let logEntry = document.createElement('div');
  
  if (data) {
      logEntry.innerHTML = `<strong>${timestamp}:</strong> ${message}<br><pre>${JSON.stringify(data, null, 2)}</pre>`;
  } else {
      logEntry.innerHTML = `<strong>${timestamp}:</strong> ${message}`;
  }
  
  debugInfo.appendChild(logEntry);
  debugInfo.appendChild(document.createElement('hr'));
  
  // Auto-scroll to bottom
  debugInfo.scrollTop = debugInfo.scrollHeight;
}

// Form validation function
function validateForm() {
  let isValid = true;
  const fields = [
      { id: "studentName", errorId: "studentNameError", message: "Please enter student name" },
      { id: "studentAge", errorId: "studentAgeError", message: "Please enter age (10-100)" },
      { id: "gender", errorId: "genderError", message: "Please select gender" },
      { id: "ethnicity", errorId: "ethnicityError", message: "Please select ethnicity" },
      { id: "studyTime", errorId: "studyTimeError", message: "Please enter study hours (0-20)" },
      { id: "absences", errorId: "absencesError", message: "Please enter number of absences (0-30)" },
      { id: "parentEducation", errorId: "parentEducationError", message: "Please select parental education" },
      { id: "tutoring", errorId: "tutoringError", message: "Please select tutoring option" },
      { id: "parentalSupport", errorId: "parentalSupportError", message: "Please select parental support option" },
      { id: "extracurricular", errorId: "extracurricularError", message: "Please select extracurricular option" },
      { id: "sports", errorId: "sportsError", message: "Please select sports participation" },
      { id: "music", errorId: "musicError", message: "Please select music participation" },
      { id: "volunteering", errorId: "volunteeringError", message: "Please select volunteering activities" }
  ];

  // Reset previous errors
  fields.forEach(field => {
      const errorElement = document.getElementById(field.errorId);
      if (errorElement) {
          errorElement.style.display = 'none';
          errorElement.textContent = "";
      }
      const inputElement = document.getElementById(field.id);
      if (inputElement) {
          inputElement.classList.remove("input-error");
      }
  });

  // Check each field
  fields.forEach(field => {
      const element = document.getElementById(field.id);
      if (!element) return;
      
      const value = element.value.trim();
      
      if (value === "" || (["studyTime", "absences", "studentAge"].includes(field.id) && isNaN(parseFloat(value)))) {
          const errorElement = document.getElementById(field.errorId);
          if (errorElement) {
              errorElement.textContent = field.message;
              errorElement.style.display = 'block';
          }
          element.classList.add("input-error");
          isValid = false;
      }
      
      // Additional validation for numeric fields
      if (["studyTime", "absences", "studentAge"].includes(field.id)) {
          const numValue = parseFloat(value);
          let min, max;
          
          if (field.id === "studyTime") {
              min = 0; max = 20;
          } else if (field.id === "absences") {
              min = 0; max = 30;
          } else if (field.id === "studentAge") {
              min = 10; max = 100;
          }
          
          if (numValue < min || numValue > max) {
              const errorElement = document.getElementById(field.errorId);
              if (errorElement) {
                  errorElement.textContent = `Value must be between ${min} and ${max}`;
                  errorElement.style.display = 'block';
              }
              element.classList.add("input-error");
              isValid = false;
          }
      }
  });
  
  return isValid;
}

// Function to collect form data
function collectFormData() {
  return {
      studentName: document.getElementById('studentName').value.trim(),
      studentAge: parseInt(document.getElementById('studentAge').value),
      gender: parseInt(document.getElementById('gender').value),
      ethnicity: parseInt(document.getElementById('ethnicity').value),
      studyTime: parseInt(document.getElementById('studyTime').value),
      absences: parseInt(document.getElementById('absences').value),
      tutoring: parseInt(document.getElementById('tutoring').value),
      extracurricular: parseInt(document.getElementById('extracurricular').value),
      parentEducation: parseInt(document.getElementById('parentEducation').value),
      parentalSupport: parseInt(document.getElementById('parentalSupport').value),
      sports: parseInt(document.getElementById('sports').value),
      music: parseInt(document.getElementById('music').value),
      volunteering: parseInt(document.getElementById('volunteering').value)
  };
}

// Function to convert form data to API format
function prepareDataForAPI(formData) {
  return {
      'StudyTimeWeekly': formData.studyTime,
      'Absences': formData.absences,
      'Gender': formData.gender,
      'Ethnicity': formData.ethnicity,
      'ParentalEducation': formData.parentEducation,
      'Tutoring': formData.tutoring,
      'ParentalSupport': formData.parentalSupport,
      'Extracurricular': formData.extracurricular,
      'Sports': formData.sports,
      'Music': formData.music,
      'Volunteering': formData.volunteering
  };
}

// Function to call prediction API
async function predictPerformance(formData) {
  debugLog("Starting prediction preparation with form data:", formData);
  
  try {
      // Convert form data to API format
      const apiData = prepareDataForAPI(formData);
      debugLog("Prepared API data:", apiData);
      
      // Call prediction API
      const response = await fetch('http://127.0.0.1:5000/predict', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify(apiData)
      });
      
      if (!response.ok) {
          const errorText = await response.text();
          debugLog("API error response:", errorText);
          throw new Error(`API request failed with status ${response.status}: ${errorText}`);
      }
      
      const predictionResult = await response.json();
      debugLog("Prediction API response:", predictionResult);
      
      // Create recommendation text
      const recommendation = createRecommendationText(predictionResult.gpa);
      
      // Prepare strengths and improvements based on form data
      const strengths = [];
      const improvements = [];
      
      // Add basic strengths based on form data
      if (formData.studyTime >= 10) strengths.push('Dedicated study habits');
      if (formData.absences <= 5) strengths.push('Consistent attendance');
      if (formData.tutoring === 1) strengths.push('Taking advantage of tutoring');
      if (formData.extracurricular === 1) strengths.push('Balanced academic and extracurricular involvement');
      if (formData.parentalSupport >= 3) strengths.push('Strong parental support');
      
      // Add improvements based on form data
      if (formData.studyTime < 10) improvements.push('Increase weekly study time');
      if (formData.absences > 5) improvements.push('Improve attendance record');
      if (formData.tutoring === 0) improvements.push('Consider seeking tutoring support');
      
      // Map GPA to performance level
      const performance = mapGpaToPerformance(predictionResult.gpa);
      const grade = mapGpaToGrade(predictionResult.gpa);
      
      return {
          score: Math.round(predictionResult.gpa * 25), // Convert GPA to score out of 100
          grade: grade,
          performance: performance,
          recommendation: recommendation,
          strengths: strengths,
          improvements: improvements,
          gpa: predictionResult.gpa,
          gpa_bin_label: predictionResult.gpa_bin_label || "Medium",
          gradeClass_label: predictionResult.gradeClass_label
      };
  } catch (error) {
      debugLog("Error calling API:", error);
      throw error;
  }
}

// Function to call recommendation API
async function getRecommendations(formData) {
  debugLog("Requesting recommendations from backend API", formData);
  
  try {
      // Convert form data to API format
      const apiData = prepareDataForAPI(formData);
      debugLog("Prepared API data for recommendations:", apiData);
      
      // Call recommendation API
      const response = await fetch('http://127.0.0.1:5000/recommend', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify(apiData)
      });
      
      if (!response.ok) {
          const errorText = await response.text();
          debugLog("API error response:", errorText);
          throw new Error(`API request failed with status ${response.status}: ${errorText}`);
      }
      
      const result = await response.json();
      debugLog("Recommendation API response:", result);
      
      return result;
  } catch (error) {
      debugLog("Error calling recommendation API:", error);
      throw error;
  }
}

// Helper function to map GPA to grade letter
function mapGpaToGrade(gpa) {
  if (gpa >= 3.7) return 'A';
  if (gpa >= 3.3) return 'A-';
  if (gpa >= 3.0) return 'B+';
  if (gpa >= 2.7) return 'B';
  if (gpa >= 2.3) return 'B-';
  if (gpa >= 2.0) return 'C+';
  if (gpa >= 1.7) return 'C';
  if (gpa >= 1.3) return 'C-';
  if (gpa >= 1.0) return 'D';
  return 'F';
}

// Helper function to map GPA to performance level
function mapGpaToPerformance(gpa) {
  if (gpa >= 3.5) return 'Excellent';
  if (gpa >= 3.0) return 'Good';
  if (gpa >= 2.0) return 'Average';
  if (gpa >= 1.0) return 'Below Average';
  return 'Needs Improvement';
}

// Helper function to create recommendation text
function createRecommendationText(gpa) {
  const performance = mapGpaToPerformance(gpa);
  
  const baseRecommendations = {
      'Excellent': 'Keep up the excellent work! Consider mentoring other students or exploring advanced topics.',
      'Good': 'Strong performance! To reach excellence, consider increasing study time and seeking additional resources.',
      'Average': 'You\'re on the right track. Consider tutoring assistance and more consistent study habits.',
      'Below Average': 'More focused effort is needed. Consider reducing absences and seeking academic support.',
      'Needs Improvement': 'Immediate intervention recommended. Increase study time, reduce absences, and seek tutoring support.'
  };
  
  return baseRecommendations[performance] || baseRecommendations['Average'];
}

// Function to display prediction results
function displayResults(data, prediction) {
  const resultCard = document.getElementById('output-result');
  const resultContent = document.getElementById('resultContent');
  const studentInfoBox = document.getElementById('studentInfoBox');
  const studentInfoContent = document.getElementById('studentInfoContent');
  
  // Display student info
  studentInfoBox.style.display = 'block';
  studentInfoContent.innerHTML = `
      <p><strong>Name:</strong> ${data.studentName}</p>
      <p><strong>Age:</strong> ${data.studentAge}</p>
      <p><strong>Gender:</strong> ${data.gender === 0 ? 'Male' : 'Female'}</p>
      <p><strong>Weekly Study Hours:</strong> ${data.studyTime}</p>
  `;
  
  // Create performance meter
  const performanceMeter = `
      <div style="margin: 25px 0;">
          <div style="margin-bottom: 10px;">Performance Score: <strong>${prediction.score}/100</strong></div>
          <div style="height: 10px; background-color: #e0e0e0; border-radius: 5px; overflow: hidden;">
              <div style="height: 100%; width: ${prediction.score}%; background: linear-gradient(90deg, #4a90e2 0%, #63b3ed 100%); border-radius: 5px;"></div>
          </div>
      </div>
  `;
  
  // Create strengths and improvements lists
  const strengthsList = prediction.strengths.length > 0 
      ? `<ul style="padding-left: 20px; margin-bottom: 15px;">${prediction.strengths.map(s => `<li>${s}</li>`).join('')}</ul>`
      : '<p>No specific strengths identified.</p>';
      
  const improvementsList = prediction.improvements.length > 0
      ? `<ul style="padding-left: 20px; margin-bottom: 15px;">${prediction.improvements.map(i => `<li>${i}</li>`).join('')}</ul>`
      : '<p>No specific improvements needed.</p>';
  
  // Add GPA and Grade Class info
  const gpaInfo = `
      <div style="margin: 15px 0;">
          <p><strong>GPA:</strong> ${prediction.gpa.toFixed(2)} (${prediction.gpa_bin_label})</p>
          <p><strong>Grade Class:</strong> ${prediction.gradeClass_label}</p>
      </div>
  `;
  
  // Build the complete result content
  resultContent.innerHTML = `
      <p>Based on our analysis, your predicted performance is: <strong>${prediction.performance}</strong> (Grade: ${prediction.grade})</p>
      ${performanceMeter}
      ${gpaInfo}
      <div style="margin-top: 20px;">
          <h4 style="font-size: 16px; margin-bottom: 10px; color: #4a90e2;">Strengths</h4>
          ${strengthsList}
          
          <h4 style="font-size: 16px; margin-bottom: 10px; color: #4a90e2;">Areas for Improvement</h4>
          ${improvementsList}
          
          <h4 style="font-size: 16px; margin-bottom: 10px; color: #4a90e2;">Recommendation</h4>
          <p>${prediction.recommendation}</p>
      </div>
  `;
  
  // Show the result card with animation
  resultCard.style.display = 'block';
  setTimeout(() => {
      resultCard.classList.add('animate');
  }, 100);
  
  // Scroll to results
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Function to display recommendation results
function displayRecommendations(data, recommendations) {
  const resultCard = document.getElementById('output-result');
  const resultContent = document.getElementById('resultContent');
  const studentInfoBox = document.getElementById('studentInfoBox');
  const studentInfoContent = document.getElementById('studentInfoContent');
  
  // Display student info
  studentInfoBox.style.display = 'block';
  studentInfoContent.innerHTML = `
      <p><strong>Name:</strong> ${data.studentName}</p>
      <p><strong>Age:</strong> ${data.studentAge}</p>
      <p><strong>Gender:</strong> ${data.gender === 0 ? 'Male' : 'Female'}</p>
      <p><strong>Weekly Study Hours:</strong> ${data.studyTime}</p>
  `;
  
  // Get the GPA values
  const currentGPA = recommendations.gpa;
  const potentialGPA = recommendations.potential_gpa;
  const improvement = recommendations.total_potential_improvement;
  
  // Create GPA display with progress bars
  const gpaDisplay = `
    <div style="margin: 25px 0;">
      <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
        <span>Current GPA:</span>
        <strong>${currentGPA.toFixed(2)}</strong>
      </div>
      <div style="height: 10px; background-color: #e0e0e0; border-radius: 5px; overflow: hidden; margin-bottom: 15px;">
        <div style="height: 100%; width: ${(currentGPA / 4) * 100}%; background: linear-gradient(90deg, #4a90e2 0%, #63b3ed 100%); border-radius: 5px;"></div>
      </div>
      
      <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
        <span>Potential GPA:</span>
        <strong>${potentialGPA.toFixed(2)}</strong>
      </div>
      <div style="height: 10px; background-color: #e0e0e0; border-radius: 5px; overflow: hidden;">
        <div style="height: 100%; width: ${(potentialGPA / 4) * 100}%; background: linear-gradient(90deg, #4caf50 0%, #81c784 100%); border-radius: 5px;"></div>
      </div>
      
      <div style="margin-top: 10px; font-weight: bold; color: #4caf50;">
        Potential Improvement: +${improvement.toFixed(2)} GPA points
      </div>
    </div>
  `;
  
  // Create recommendations list
  let recommendationsList = '<div style="margin-top: 20px;"><h4 style="font-size: 16px; margin-bottom: 15px; color: #4a90e2;">Recommendations for Improvement</h4>';
  
  if (recommendations.recommendations && recommendations.recommendations.length > 0) {
    recommendationsList += '<ul style="padding-left: 0; list-style-type: none;">';
    recommendations.recommendations.forEach(rec => {
      // Set different styles based on priority
      let priorityColor = '#4caf50'; // Default: low priority (green)
      if (rec.priority === 'Medium') priorityColor = '#ff9800'; // Medium: orange
      if (rec.priority === 'High') priorityColor = '#f44336'; // High: red
      
      recommendationsList += `
        <li style="margin-bottom: 15px; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 4px solid ${priorityColor};">
          <div style="font-weight: 600; margin-bottom: 8px; display: flex; justify-content: space-between;">
            <span>${rec.factor}</span>
            <span style="color: ${priorityColor}; font-size: 14px;">${rec.priority} Priority</span>
          </div>
          <div style="margin-bottom: 5px; color: #555; font-size: 14px;">${rec.issue}</div>
          <div style="font-weight: 500;">${rec.recommendation}</div>
          <div style="margin-top: 8px; font-size: 13px; color: #4a90e2;">Potential improvement: +${rec.impact.toFixed(2)} GPA points</div>
        </li>
      `;
    });
    recommendationsList += '</ul>';
  } else {
    recommendationsList += '<p>No specific recommendations are available at this time. You are already performing well!</p>';
  }
  
  recommendationsList += '</div>';
  
  // Build the complete result content
  resultContent.innerHTML = `
    <p>Based on our analysis, your current GPA is <strong>${currentGPA.toFixed(2)}</strong> (${recommendations.gpa_category})</p>
    ${gpaDisplay}
    ${recommendationsList}
  `;
  
  // Show the result card with animation
  resultCard.style.display = 'block';
  setTimeout(() => {
    resultCard.classList.add('animate');
  }, 100);
  
  // Scroll to results
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Add recommendation button on page load
document.addEventListener('DOMContentLoaded', function() {
  const formCard = document.querySelector('.form-card:last-child');
  const generateBtn = document.getElementById('generateBtn');
  
  if (formCard && generateBtn) {
    // Create new recommendation button
    const recommendBtn = document.createElement('button');
    recommendBtn.type = 'button';
    recommendBtn.id = 'recommendBtn';
    recommendBtn.className = 'submit-btn';
    recommendBtn.style.marginTop = '10px';
    recommendBtn.style.backgroundColor = '#4a90e2';
    recommendBtn.style.color = '#fff';
    recommendBtn.textContent = 'Get Recommendations';
    
    // Insert it after the generate button
    generateBtn.insertAdjacentElement('afterend', recommendBtn);
    
    // Add event listener for recommendation button
    recommendBtn.addEventListener('click', async function() {
      if (!validateForm()) {
        debugLog("Form validation failed");
        return;
      }
      
      // Hide previous results
      const resultCard = document.getElementById('output-result');
      resultCard.style.display = 'none';
      resultCard.classList.remove('animate');
      
      // Show loading indicator
      const loadingIndicator = document.getElementById('loadingIndicator');
      loadingIndicator.style.display = 'block';
      
      try {
        // Collect form data
        const formData = collectFormData();
        debugLog("Form data collected for recommendations", formData);
        
        // Get recommendations
        const recommendations = await getRecommendations(formData);
        
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        
        // Display recommendation results
        displayRecommendations(formData, recommendations);
      } catch (error) {
        // Handle errors
        debugLog("Error during recommendations", error);
        loadingIndicator.style.display = 'none';
        alert("Error processing your request. Please try again.");
      }
    });
  }
});

// Handle generate button click
document.getElementById('generateBtn').addEventListener('click', async function() {
  if (!validateForm()) {
      debugLog("Form validation failed");
      return;
  }
  
  // Hide previous results
  const resultCard = document.getElementById('output-result');
  resultCard.style.display = 'none';
  resultCard.classList.remove('animate');
  
  // Show loading indicator
  const loadingIndicator = document.getElementById('loadingIndicator');
  loadingIndicator.style.display = 'block';
  
  try {
      // Collect form data
      const formData = collectFormData();
      debugLog("Form data collected", formData);
      
      // Predict performance
      const prediction = await predictPerformance(formData);
      
      // Hide loading indicator
      loadingIndicator.style.display = 'none';
      
      // Display results
      displayResults(formData, prediction);
      
  } catch (error) {
      // Handle errors
      debugLog("Error during prediction", error);
      loadingIndicator.style.display = 'none';
      alert("Error processing your request. Please try again.");
  }
});

// Scroll animation for elements
function handleScrollAnimations() {
  const elements = document.querySelectorAll('.fade-in-up, .form-card, .frame-wrapper, .title-left, .title-right');
  
  elements.forEach(element => {
      const elementTop = element.getBoundingClientRect().top;
      const windowHeight = window.innerHeight;
      
      if (elementTop < windowHeight * 0.85) {
          element.classList.add('animate');
      }
  });
}

// Initialize animations on page load
window.addEventListener('load', () => {
  setTimeout(handleScrollAnimations, 100);
});

// Handle scroll animations
window.addEventListener('scroll', handleScrollAnimations);