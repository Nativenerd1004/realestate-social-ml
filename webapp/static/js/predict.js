document.getElementById('predict-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = document.getElementById('btn-text');
  btn.textContent = 'Predicting...';

  const payload = {
    price:                parseFloat(document.getElementById('price').value),
    availability_365:     parseFloat(document.getElementById('availability_365').value),
    minimum_nights:       parseFloat(document.getElementById('minimum_nights').value),
    number_of_reviews:    parseFloat(document.getElementById('number_of_reviews').value),
    review_scores_rating: parseFloat(document.getElementById('review_scores_rating').value),
    host_response_rate:   parseFloat(document.getElementById('host_response_rate').value) / 100,
    accommodates:         parseFloat(document.getElementById('accommodates').value),
    beds:                 parseFloat(document.getElementById('beds').value),
    bedrooms:             parseFloat(document.getElementById('bedrooms').value),
    bathrooms:            parseFloat(document.getElementById('bathrooms').value),
    room_type:            parseFloat(document.getElementById('room_type').value),
    host_is_superhost:    parseFloat(document.getElementById('host_is_superhost').value),
    host_response_time:   parseFloat(document.getElementById('host_response_time').value),
  };

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) { alert('Error: ' + data.error); btn.textContent = 'Run Prediction →'; return; }

    showResult(data);
  } catch(err) {
    alert('Request failed: ' + err);
    btn.textContent = 'Run Prediction →';
  }
});

function showResult(data) {
  const panel = document.getElementById('result-panel');
  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth' });

  // Gauge
  const pct   = data.probability;
  const color = data.color;
  const ring  = document.getElementById('gauge-ring');
  ring.style.background = `conic-gradient(${color} 0% ${pct}%, #21262d ${pct}% 100%)`;
  ring.style.boxShadow  = `0 0 30px ${color}55`;
  document.getElementById('gauge-pct').textContent  = pct + '%';
  document.getElementById('gauge-pct').style.color  = color;

  // Label
  const lbl = document.getElementById('result-label');
  lbl.textContent  = data.label;
  lbl.style.color  = color;

  // Progress bar
  const bar = document.getElementById('result-bar');
  bar.style.width      = pct + '%';
  bar.style.background = color;

  // Top features
  const maxImp = Math.max(...data.top_features.map(f => f.importance));
  const list   = document.getElementById('feature-list');
  list.innerHTML = data.top_features.map(f => `
    <div class="feat-row">
      <span class="feat-name">${f.feature}</span>
      <div class="feat-bar-wrap">
        <div class="feat-bar" style="width:${Math.round(f.importance/maxImp*100)}%"></div>
      </div>
      <span class="feat-val">${(f.importance*100).toFixed(2)}%</span>
    </div>
  `).join('');
}

function resetForm() {
  document.getElementById('result-panel').style.display = 'none';
  document.getElementById('btn-text').textContent = 'Run Prediction →';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
