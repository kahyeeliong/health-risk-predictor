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
      resultDiv.innerText = `Risk Level: ${result.risk}`;
      resultDiv.style.color = result.risk === 'high' ? 'red' : 'green';

    } catch (error) {
      console.error('Error:', error);
      resultDiv.innerText = 'Something went wrong. Please try again.';
      resultDiv.style.color = 'orange';
    }
  });
});