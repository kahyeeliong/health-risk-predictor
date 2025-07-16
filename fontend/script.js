document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('riskForm');
  const resultDiv = document.getElementById('result');

  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(form);
    const data = {};

    formData.forEach((value, key) => {
      data[key] = isNaN(value) ? value : parseFloat(value);
    });

    try {
      const response = await fetch('https://health-risk-predictor-kyom.onrender.com/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("Result:", result);

      // Update risk level text
      riskText.innerText = `Risk Level: ${result.risk.toUpperCase()}`;
      
      // Update confidence text
      if (result.confidence !== undefined) {
        confidenceText.innerText = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
      } else {
        confidenceText.innerText = '';
      }

      // Reset circle class
      riskCircle.className = 'circle';

      // Apply color class based on risk
      if (result.risk === 'low') {
        riskCircle.classList.add('low');
      } else if (result.risk === 'moderate') {
        riskCircle.classList.add('moderate');
      } else if (result.risk === 'high') {
        riskCircle.classList.add('high');
      }

    } catch (error) {
      console.error('Error:', error);
      riskText.innerText = 'Something went wrong.';
      riskCircle.className = 'circle';
      confidenceText.innerText = '';
    }
  });
});