// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.predict-tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.ptab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}

// ── Shared helpers ────────────────────────────────────────────────────────────
function setGauge(prefix, pct, color) {
  const ring  = document.getElementById(prefix + '-gauge-ring');
  const pctEl = document.getElementById(prefix + '-gauge-pct');
  ring.style.background = `conic-gradient(${color} 0% ${pct}%, var(--surface2) ${pct}% 100%)`;
  ring.style.boxShadow  = `0 0 30px ${color}55`;
  pctEl.textContent = pct + '%';
  pctEl.style.color = color;
}

function setBar(prefix, pct, color) {
  const bar = document.getElementById(prefix + '-bar');
  bar.style.width      = pct + '%';
  bar.style.background = color;
}

function setFeats(prefix, feats) {
  const el = document.getElementById(prefix + '-feats');
  if (!feats || feats.length === 0) {
    el.innerHTML = '<p style="color:var(--muted);font-size:0.85rem">Not available for this model type.</p>';
    return;
  }
  const max = feats[0].importance;
  el.innerHTML = feats.map(f => `
    <div class="feat-row">
      <span class="feat-name">${f.feature}</span>
      <div class="feat-bar-wrap">
        <div class="feat-bar" style="width:${Math.round((f.importance / max) * 100)}%"></div>
      </div>
      <span class="feat-val">${(f.importance * 100).toFixed(1)}%</span>
    </div>`).join('');
}

function showResult(prefix) {
  const el = document.getElementById(prefix + '-result');
  el.style.display = 'block';
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetResult(prefix) {
  document.getElementById(prefix + '-result').style.display = 'none';
}

async function runPredict(endpoint, prefix, payload, btnTextId, originalText) {
  const btn = document.getElementById(btnTextId);
  btn.textContent = 'Running Model...';
  try {
    const res  = await fetch(endpoint, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    setGauge(prefix, data.probability, data.color);
    setBar(prefix, data.probability, data.color);

    const lbl = document.getElementById(prefix + '-label');
    lbl.textContent = data.label;
    lbl.style.color = data.color;

    setFeats(prefix, data.top_features);

    const note = document.getElementById(prefix + '-model-note');
    if (note) note.innerHTML = `<strong>Result:</strong> ${data.label} &nbsp;·&nbsp; <strong>Confidence:</strong> ${data.probability}%`;

    showResult(prefix);
  } catch (err) {
    alert('Prediction error: ' + err.message);
  } finally {
    btn.textContent = originalText;
  }
}

// ── TAB 1: Host Churn ────────────────────────────────────────────────────────
document.getElementById('churn-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const g = id => parseFloat(document.getElementById('c-' + id)?.value ?? 0);
  await runPredict('/predict/churn', 'churn', {
    propensity_score: g('propensity_score'),
    segment_enc:      g('segment_enc'),
    acquisition_enc:  g('acquisition_enc'),
    income_enc:       g('income_enc'),
    age:              g('age'),
    household_size:   g('household_size'),
    avg_calls:        g('avg_calls'),
    avg_views:        g('avg_views'),
    avg_visits:       g('avg_visits'),
    avg_sessions:     g('avg_sessions'),
    avg_session_min:  g('avg_session_min'),
    avg_engagement:   g('avg_engagement'),
    avg_churn_risk:   g('avg_churn_risk'),
    avg_revenue:      g('avg_revenue'),
    total_deals:      g('total_deals'),
    months_active:    g('months_active'),
  }, 'churn-btn-text', 'Predict Churn Risk →');
});

// ── TAB 2: Social Media ──────────────────────────────────────────────────────
document.getElementById('social-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const g = id => parseFloat(document.getElementById('s-' + id)?.value ?? 0);
  await runPredict('/predict/social', 'social', {
    account_type_enc:     g('account_type_enc'),
    media_type_enc:       g('media_type_enc'),
    content_category_enc: g('content_category_enc'),
    traffic_source_enc:   g('traffic_source_enc'),
    follower_count:       g('follower_count'),
    has_call_to_action:   g('has_call_to_action'),
    post_hour:            g('post_hour'),
    day_enc:              g('day_enc'),
    caption_length:       g('caption_length'),
    hashtags_count:       g('hashtags_count'),
  }, 'social-btn-text', 'Predict Performance →');
});

// ── TAB 3: Listing Sale ──────────────────────────────────────────────────────
document.getElementById('listing-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const g = id => parseFloat(document.getElementById('l-' + id)?.value ?? 0);
  await runPredict('/predict/listing', 'listing', {
    prop_type_enc:    g('prop_type_enc'),
    size_sqm:         g('size_sqm'),
    bedrooms:         g('bedrooms'),
    bathrooms:        g('bathrooms'),
    year_built:       g('year_built'),
    location_score:   g('location_score'),
    amenities_count:  g('amenities_count'),
    list_price:       g('list_price'),
    has_parking:      g('has_parking'),
    near_transit:     g('near_transit'),
    near_school:      g('near_school'),
    list_channel_enc: g('list_channel_enc'),
    city_enc:         g('city_enc'),
    listed_price:     g('listed_price'),
  }, 'listing-btn-text', 'Predict Sale Likelihood →');
});
