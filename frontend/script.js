// Same server when the page is served by FastAPI; local API when opened as a file.
const API_BASE = window.location.protocol.startsWith('http') ? '' : 'http://127.0.0.1:8000';

const EXAMPLES = {
  low:  { Pregnancies: 1, Glucose: 95,  BloodPressure: 70, BMI: 24.0, DiabetesPedigreeFunction: 0.3, Age: 28 },
  high: { Pregnancies: 3, Glucose: 175, BloodPressure: 85, BMI: 38.5, DiabetesPedigreeFunction: 0.9, Age: 50 },
};

const RISK_LABEL = { low: 'Low risk', moderate: 'Moderate risk', high: 'High risk' };
const TOP_N = 4;

const $ = (id) => document.getElementById(id);
const form = $('riskForm');

function fillForm(values) {
  for (const input of form.querySelectorAll('input')) {
    input.value = values[input.name] ?? '';
  }
}

function readForm() {
  const data = {};
  for (const input of form.querySelectorAll('input')) {
    if (input.value !== '') data[input.name] = Number(input.value);
  }
  return data;
}

function showError(message) {
  $('output').hidden = true;
  $('placeholder').hidden = true;
  $('error').hidden = false;
  $('error').textContent = message;
}

function render(result) {
  $('placeholder').hidden = true;
  $('error').hidden = true;
  $('output').hidden = false;

  $('riskCircle').className = `circle ${result.risk}`;
  $('riskText').textContent = RISK_LABEL[result.risk];

  const pct = result.probability * 100;
  $('meterMarker').style.left = `${pct}%`;
  $('probabilityText').textContent =
    `Estimated probability of diabetes: ${pct.toFixed(0)}%`;

  const top = result.factors.slice(0, TOP_N);
  const maxWeight = Math.max(...top.map((f) => Math.abs(f.weight)), 0.01);
  $('factors').innerHTML = '';
  for (const f of top) {
    const li = document.createElement('li');
    li.className = f.effect;
    const value = f.value === null ? 'not given' : f.value;
    li.innerHTML = `
      <div class="factor-text">
        <span class="factor-name">${f.label} <small>${value}</small></span>
        <span class="factor-effect">${f.effect} risk</span>
      </div>
      <div class="factor-bar"><span style="width:${(Math.abs(f.weight) / maxWeight) * 100}%"></span></div>`;
    $('factors').appendChild(li);
  }
}

async function predict() {
  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }
  const button = $('submitBtn');
  button.disabled = true;
  button.textContent = 'Predicting...';
  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(readForm()),
    });
    const body = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(body.detail)
        ? body.detail.map((d) => `${d.loc.at(-1)}: ${d.msg}`).join('; ')
        : `status ${response.status}`;
      throw new Error(detail);
    }
    render(body);
  } catch (err) {
    showError(`Could not get a prediction (${err.message}). Is the API running?`);
  } finally {
    button.disabled = false;
    button.textContent = 'Predict risk';
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  predict();
});

for (const chip of document.querySelectorAll('[data-example]')) {
  chip.addEventListener('click', () => {
    fillForm(EXAMPLES[chip.dataset.example]);
    predict();
  });
}
