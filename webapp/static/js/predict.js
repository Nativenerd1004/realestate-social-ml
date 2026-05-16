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

// ── TAB 2: Market Position ───────────────────────────────────────────────────
document.getElementById('social-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const g = id => parseFloat(document.getElementById('s-' + id)?.value ?? 0);
  await runPredict('/predict/social', 'social', {
    state_enc:  g('state_enc'),
    area_enc:   g('area_enc'),
    bed:        g('bed'),
    bath:       g('bath'),
    house_size: g('house_size'),
    acre_lot:   g('acre_lot'),
  }, 'social-btn-text', 'Predict Market Position →');
});

// ── TAB 4: Social Media Performance (YouTube Trending) ───────────────────────
document.getElementById('yt-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const g   = id => parseFloat(document.getElementById('yt-' + id)?.value ?? 0);
  const rawViews    = g('views_raw');
  const commentsPerK = g('comments_per_k');
  await runPredict('/predict/yt', 'yt', {
    cat_enc:      g('cat_enc'),
    country_enc:  g('country_enc'),
    title_length: g('title_length'),
    tag_count:    g('tag_count'),
    publish_hour: g('publish_hour'),
    publish_dow:  g('publish_dow'),
    log_views:    Math.log1p(rawViews),
    comment_rate: commentsPerK / 1000,
  }, 'yt-btn-text', 'Predict Engagement →');
});

// ── TAB 3: Listing Sale (Illinois 2026) ──────────────────────────────────────
document.getElementById('listing-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const g = id => parseFloat(document.getElementById('l-' + id)?.value ?? 0);
  await runPredict('/predict/listing', 'listing', {
    type_enc:             g('type_enc'),
    sub_type_enc:         g('sub_type_enc'),
    listPrice:            g('listPrice'),
    sqft:                 g('sqft'),
    lot_sqft:             g('lot_sqft'),
    stories:              g('stories'),
    beds:                 g('beds'),
    baths:                g('baths'),
    baths_full:           g('baths_full'),
    garage:               g('garage'),
    year_built:           g('year_built'),
    price_per_sqft_sold:  g('price_per_sqft_sold'),
    property_age_at_sale: g('property_age_at_sale'),
  }, 'listing-btn-text', 'Predict Sale Outcome →');
});
