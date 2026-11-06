const API = "";

// ---------- Tabs ----------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("panel-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "dashboard") loadDashboard();
  });
});

// ---------- Status pills ----------
async function loadStatus() {
  try {
    const res = await fetch(`${API}/api/status`);
    const data = await res.json();
    const pills = document.getElementById("statusPills");
    pills.innerHTML = `
      <span class="pill ${data.llm_available ? "pill-on" : "pill-off"}">LLM fallback ${data.llm_available ? "ready" : "offline"}</span>
      <span class="pill ${data.factcheck_available ? "pill-on" : "pill-off"}">Fact-check ${data.factcheck_available ? "ready" : "offline"}</span>
      <span class="pill pill-on">${data.primary_model} · ${(data.metrics.accuracy * 100).toFixed(1)}% accuracy</span>
    `;
    window._statusData = data;
  } catch (e) {
    document.getElementById("statusPills").innerHTML = `<span class="pill pill-off">Backend not reachable</span>`;
  }
}
loadStatus();

// ---------- Analyze ----------
document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const text = document.getElementById("newsInput").value.trim();
  if (!text) { alert("Please paste a headline or article first."); return; }

  const btn = document.getElementById("analyzeBtn");
  btn.disabled = true; btn.textContent = "Analyzing…";

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        use_llm: document.getElementById("useLlm").checked,
        use_factcheck: document.getElementById("useFactcheck").checked,
      }),
    });
    const result = await res.json();
    renderResult(result);
  } catch (e) {
    alert("Could not reach the backend. Is `python backend/api.py` running?");
  } finally {
    btn.disabled = false; btn.textContent = "Analyze News";
  }
});

function renderResult(result) {
  const card = document.getElementById("resultCard");
  card.style.display = "block";

  const badge = document.getElementById("verdictBadge");
  badge.textContent = result.final_label === "fake" ? "Flagged: Fake" : "Verified: Real";
  badge.className = "verdict-badge " + (result.final_label === "fake" ? "fake" : "real");

  document.getElementById("confidenceValue").textContent = `${(result.final_confidence * 100).toFixed(1)}%`;
  document.getElementById("confidenceBar").style.width = `${(result.final_confidence * 100).toFixed(1)}%`;

  document.getElementById("mlModelName").textContent = `Model: ${result.ml_prediction.model_used}`;
  document.getElementById("mlModelResult").textContent =
    `Predicted ${result.ml_prediction.label.toUpperCase()} at ${(result.ml_prediction.confidence * 100).toFixed(1)}% confidence`;

  const sig = result.style_signals;
  document.getElementById("styleSignals").innerHTML = `
    <li>Exclamation marks: ${sig.exclamations}</li>
    <li>ALL-CAPS words: ${sig.all_caps_words}</li>
    <li>Clickbait phrases: ${sig.clickbait_phrases}</li>
  `;

  const llmCard = document.getElementById("llmCard");
  if (result.llm_review) {
    llmCard.style.display = "block";
    if (result.llm_review.error) {
      document.getElementById("llmContent").textContent = `LLM call failed: ${result.llm_review.error}`;
    } else {
      document.getElementById("llmContent").textContent =
        `${result.llm_review.label.toUpperCase()} (${(result.llm_review.confidence * 100).toFixed(1)}%) — ${result.llm_review.explanation}`;
    }
  } else {
    llmCard.style.display = "none";
  }

  const fcCard = document.getElementById("factcheckCard");
  if (result.fact_check) {
    fcCard.style.display = "block";
    const fc = result.fact_check;
    if (fc.claims && fc.claims.length) {
      document.getElementById("factcheckContent").innerHTML = fc.claims.map(c => `
        <div class="factcheck-item">
          <strong>${c.claim_text}</strong><br>
          Rating: ${c.rating} · Publisher: ${c.publisher}
          ${c.url ? `<br><a href="${c.url}" target="_blank" rel="noopener">Read more</a>` : ""}
        </div>
      `).join("");
    } else {
      document.getElementById("factcheckContent").textContent = fc.message || "No related fact-checks found.";
    }
  } else {
    fcCard.style.display = "none";
  }

  const notesBox = document.getElementById("notesBox");
  if (result.verification_notes && result.verification_notes.length) {
    notesBox.style.display = "block";
    document.getElementById("notesList").innerHTML = result.verification_notes.map(n => `<li>${n}</li>`).join("");
  } else {
    notesBox.style.display = "none";
  }
}

// ---------- Batch ----------
document.getElementById("downloadSampleBtn").addEventListener("click", () => {
  window.location.href = `${API}/api/sample-batch-csv`;
});

document.getElementById("batchAnalyzeBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("batchFile");
  if (!fileInput.files.length) { alert("Please choose a CSV file first."); return; }

  const btn = document.getElementById("batchAnalyzeBtn");
  btn.disabled = true; btn.textContent = "Running batch…";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch(`${API}/api/batch`, { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    renderBatch(data);
  } catch (e) {
    alert("Could not reach the backend. Is `python backend/api.py` running?");
  } finally {
    btn.disabled = false; btn.textContent = "Run Batch Analysis";
  }
});

function renderBatch(data) {
  document.getElementById("batchResultCard").style.display = "block";
  document.getElementById("batchSummary").innerHTML = `
    <div class="batch-stat"><b>${data.total_rows}</b>Rows analyzed</div>
    <div class="batch-stat"><b>${data.fake_count}</b>Flagged fake</div>
    <div class="batch-stat"><b>${data.real_count}</b>Verified real</div>
    <div class="batch-stat"><b>${data.column_used}</b>Column used</div>
  `;
  const tbody = document.querySelector("#batchTable tbody");
  tbody.innerHTML = data.results.map(r => `
    <tr>
      <td>${r.row}</td>
      <td>${r.text}</td>
      <td><span class="verdict-tag ${r.label}">${r.label.toUpperCase()}</span></td>
      <td>${(r.confidence * 100).toFixed(1)}%</td>
    </tr>
  `).join("");
}

document.getElementById("downloadResultsBtn").addEventListener("click", () => {
  window.location.href = `${API}/api/batch/csv`;
});

// ---------- Dashboard ----------
let _dashboardLoaded = false;
async function loadDashboard() {
  if (_dashboardLoaded) return;
  const data = window._statusData || await (await fetch(`${API}/api/status`)).json();
  _dashboardLoaded = true;

  const m = data.metrics;
  document.getElementById("statGrid").innerHTML = `
    <div class="stat-box"><div class="stat-value">${(m.accuracy * 100).toFixed(1)}%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat-box"><div class="stat-value">${(m.precision * 100).toFixed(1)}%</div><div class="stat-label">Precision</div></div>
    <div class="stat-box"><div class="stat-value">${(m.recall * 100).toFixed(1)}%</div><div class="stat-label">Recall</div></div>
    <div class="stat-box"><div class="stat-value">${(m.f1 * 100).toFixed(1)}%</div><div class="stat-label">F1-score</div></div>
  `;

  const models = data.all_models;
  new Chart(document.getElementById("modelChart"), {
    type: "bar",
    data: {
      labels: Object.keys(models),
      datasets: [{
        label: "F1-score",
        data: Object.values(models).map(v => v.metrics.f1),
        backgroundColor: ["#F0B429", "#3ECF8E"],
        borderRadius: 4,
      }],
    },
    options: {
      scales: { y: { beginAtZero: true, max: 1, ticks: { color: "#93A0B4" } }, x: { ticks: { color: "#93A0B4" } } },
      plugins: { legend: { display: false } },
    },
  });

  new Chart(document.getElementById("balanceChart"), {
    type: "doughnut",
    data: {
      labels: ["Real", "Fake"],
      datasets: [{
        data: [data.class_balance.real, data.class_balance.fake],
        backgroundColor: ["#3ECF8E", "#E4572E"],
        borderColor: "#121B2E",
        borderWidth: 3,
      }],
    },
    options: {
      plugins: { legend: { labels: { color: "#E7ECF3" } } },
    },
  });
}
