"use strict";
const COLORS = {"Low":"#2e9d57","Moderate":"#e5c100","High":"#f08a24","Very High":"#d1342f"};
const CATS = Object.keys(COLORS);
const FEATURES = [["rainfall_1h","Rainfall 1h (mm)"],["rainfall_3h","Rainfall 3h (mm)"],["rainfall_24h","Rainfall 24h (mm)"],
  ["elevation","Elevation (m)"],["slope","Slope (deg)"],["drainage_density","Drainage density"],
  ["impervious_pct","Impervious (%)"],["historical_flood_count","Past flood count"]];
let map, layer, legendAdded = false, scenario = null, distChart, impChart;
const roadLayers = {};

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pill = (c) => `<span class="pill" style="background:${COLORS[c]}">${esc(c)}</span>`;

function showError(msg){ const e=$("#error"); e.textContent=msg; e.hidden=!msg; }

async function api(path, opts){
  let r;
  try { r = await fetch(path, opts); } catch { throw new Error("Cannot reach the server. Is it running?"); }
  const body = await r.json().catch(() => ({}));
  if(!r.ok){
    let d = body.detail;
    if(Array.isArray(d)) d = d.map(x => (x.loc||[]).slice(-1)[0] + ": " + x.msg).join("; ");
    throw new Error(d || `Request failed (${r.status})`);
  }
  return body;
}

// ---- navigation ----
document.querySelectorAll("nav button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("nav button").forEach(x => x.classList.toggle("active", x === b));
  document.querySelectorAll(".view").forEach(v => v.hidden = v.id !== "view-" + b.dataset.view);
  showError("");
  if(b.dataset.view === "map") initMap();
  if(b.dataset.view === "model") loadModel();
}));

// ---- dashboard ----
async function loadDashboard(){
  try{
    const d = await api("/api/dashboard");
    const ban = $("#banner");
    if(d.mode === "SYNTHETIC_DEMO"){ ban.textContent = "SYNTHETIC DEMO DATA: fictional street grid and generated labels. Not real flood risk."; ban.hidden = false; }
    else if(d.mode === "NO_MODEL"){ ban.textContent = "No trained model loaded. Run: python -m ml.train"; ban.hidden = false; }
    const k = [["Road segments", d.total_roads], ["Observed flood rate", d.observed_flood_rate==null?"n/a":(100*d.observed_flood_rate).toFixed(1)+"%"],
      ["Dataset updated", d.dataset_updated||"n/a"], ["Model trained", d.model_trained_at||"n/a"]];
    (d.counts||[]).forEach(c => k.push([c.category + " risk", `${c.count} (${c.percent}%)`]));
    $("#kpis").innerHTML = k.map(([l,v]) => `<div class="kpi"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join("");
    if(d.counts && window.Chart){
      distChart && distChart.destroy();
      distChart = new Chart($("#distChart"), {type:"bar", data:{labels:d.counts.map(c=>c.category),
        datasets:[{label:"Roads", data:d.counts.map(c=>c.count), backgroundColor:d.counts.map(c=>c.color)}]},
        options:{maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true,ticks:{precision:0}}}}});
    }
    $("#hrTable tbody").innerHTML = d.high_risk.length ? d.high_risk.map(r =>
      `<tr><td class="link" data-id="${esc(r.road_id)}">${esc(r.road_id)}</td><td>${r.probability.toFixed(2)}</td><td>${pill(r.category)}</td></tr>`).join("")
      : `<tr><td colspan="3" class="muted">No high-risk roads.</td></tr>`;
    document.querySelectorAll("#hrTable td.link").forEach(td => td.addEventListener("click", () => {
      document.querySelector('nav button[data-view="map"]').click();
      $("#search").value = td.dataset.id; setTimeout(() => findRoad(td.dataset.id), 50);
    }));
  }catch(e){ showError(e.message); $("#kpis").innerHTML = ""; }
}

// ---- map ----
function selectedCats(){ return [...document.querySelectorAll("#cats input:checked")].map(i => i.value); }

function initMap(){
  if(!window.L){ showError("Leaflet failed to load (needs internet once, or vendor it into app/static; see README)."); return; }
  if(!map){
    $("#cats").innerHTML = CATS.map(c => `<label><input type="checkbox" value="${c}" checked> ${c}</label>`).join("");
    map = L.map("map", {preferCanvas:true, zoomControl:true});
    layer = L.featureGroup().addTo(map);
    const legend = L.control({position:"bottomright"});
    legend.onAdd = () => { const d = L.DomUtil.create("div","legend");
      d.innerHTML = "<b>Risk</b><br>" + CATS.map(c=>`<i style="background:${COLORS[c]}"></i>${c}`).join("<br>"); return d; };
    legend.addTo(map);
    let tileLayer = null;
    $("#tiles").addEventListener("change", e => {
      if(e.target.checked){ tileLayer = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        {maxZoom:19, attribution:"© OpenStreetMap contributors"}).addTo(map); }
      else if(tileLayer){ map.removeLayer(tileLayer); }
    });
    document.querySelectorAll("#cats input").forEach(i => i.addEventListener("change", () => loadRoads(false)));
    $("#minp").addEventListener("input", e => { $("#minpOut").textContent = (+e.target.value).toFixed(2); });
    $("#minp").addEventListener("change", () => loadRoads(false));
    $("#colorSrc").addEventListener("change", () => loadRoads(false));
    $("#searchBtn").addEventListener("click", () => findRoad($("#search").value.trim()));
    $("#search").addEventListener("keydown", e => { if(e.key==="Enter") findRoad($("#search").value.trim()); });
    loadRoads(true);
  } else { setTimeout(() => map.invalidateSize(), 50); }
}

async function loadRoads(fit){
  $("#mapNote").textContent = "Loading roads…";
  try{
    const params = new URLSearchParams({min_probability: $("#minp").value});
    const cs = selectedCats(); if(cs.length && cs.length < 4) params.set("category", cs.join(","));
    if(!cs.length){ layer.clearLayers(); $("#mapNote").textContent = "Select at least one category."; return; }
    const fc = await api("/api/roads?" + params);
    const useScn = scenario && $("#colorSrc").value === "scenario";
    layer.clearLayers(); Object.keys(roadLayers).forEach(k => delete roadLayers[k]);
    fc.features.forEach(f => {
      const p = f.properties; let prob = p.probability, cat = p.category;
      if(useScn && scenario.roads[p.road_id]){ prob = scenario.roads[p.road_id].probability; cat = scenario.roads[p.road_id].category; }
      if(useScn && !cs.includes(cat)) return;
      const line = L.geoJSON(f.geometry, {style:{color: cat ? COLORS[cat] : "#6b7b83", weight:5, opacity:.9}});
      line.bindPopup(popupHtml(p, prob, cat, useScn));
      line.addTo(layer); roadLayers[p.road_id] = line;
    });
    $("#mapNote").textContent = `${fc.features.length} road segments shown. Mode: ${fc.mode}.` + (useScn ? " Colors show a hypothetical scenario." : "");
    if(fit && layer.getLayers().length) map.fitBounds(layer.getBounds().pad(0.05));
  }catch(e){ $("#mapNote").textContent = ""; showError(e.message); }
}

function popupHtml(p, prob, cat, isScn){
  const feats = Object.entries(p.features||{}).map(([k,v]) => `<tr><td>${esc(k)}</td><td>${v==null?"missing":esc(v)}</td></tr>`).join("");
  return `<b>${esc(p.road_id)}</b><br>${isScn?"<i>Scenario (hypothetical)</i><br>":""}` +
    (prob==null ? "No model" : `Score: ${prob.toFixed(3)} ${pill(cat)}`) +
    `<br>Observed label: ${p.observed_label==null?"n/a":(p.observed_label?"flooded":"not flooded")}` +
    `<br>Source: ${esc(p.data_source)}<table>${feats}</table>`;
}

function findRoad(id){
  if(!id) return;
  const key = Object.keys(roadLayers).find(k => k.toLowerCase() === id.toLowerCase())
    || Object.keys(roadLayers).find(k => k.toLowerCase().includes(id.toLowerCase()));
  if(!key){ $("#mapNote").textContent = `No visible road matches "${id}" (check filters).`; return; }
  const l = roadLayers[key]; map.fitBounds(l.getBounds().pad(2)); l.openPopup(l.getBounds().getCenter());
}

// ---- predict & scenario ----
function buildForm(){
  $("#predForm").innerHTML = FEATURES.map(([k,l]) => `<label>${l}<input name="${k}" type="number" step="any"></label>`).join("");
}
$("#predBtn").addEventListener("click", async () => {
  const btn = $("#predBtn"), out = $("#predOut"); showError("");
  const body = {}; new FormData($("#predForm")).forEach((v,k) => { if(v !== "") body[k] = Number(v); });
  btn.disabled = true; out.textContent = "Predicting…";
  try{
    const r = await api("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    out.innerHTML = `<b>Score ${r.probability.toFixed(3)}</b> ${pill(r.category)}<br>` +
      `<span class="muted">${esc(r.note)}${r.imputed_features.length ? " Imputed: " + esc(r.imputed_features.join(", ")) + "." : ""}</span>`;
  }catch(e){ out.textContent = ""; showError(e.message); } finally { btn.disabled = false; }
});
$("#scBtn").addEventListener("click", async () => {
  const btn = $("#scBtn"), out = $("#scOut"); showError("");
  const body = {}; new FormData($("#scForm")).forEach((v,k) => body[k] = Number(v));
  btn.disabled = true; out.textContent = "Running scenario…";
  try{
    scenario = await api("/api/scenario", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    $("#srcWrap").hidden = false; $("#colorSrc").value = "scenario";
    out.innerHTML = `<b>${esc(scenario.kind)}</b><table><thead><tr><th>Category</th><th>Baseline</th><th>Scenario</th></tr></thead><tbody>` +
      scenario.counts.map(c => `<tr><td>${pill(c.category)}</td><td>${c.baseline_count}</td><td>${c.count}</td></tr>`).join("") +
      `</tbody></table><p class="muted">${esc(scenario.warning)} View it on the Risk map tab.</p>`;
    if(map) loadRoads(false);
  }catch(e){ out.textContent = ""; showError(e.message); } finally { btn.disabled = false; }
});

// ---- model ----
async function loadModel(){
  try{
    const m = await api("/api/model-info"), md = m.metadata;
    const rows = [["Primary model", md.primary_model], ["Trained", md.trained_at], ["Data mode", md.data_mode],
      ["Target", md.target_meaning], ["Validation", md.validation], ["Train / test rows", `${md.n_train} / ${md.n_test}`],
      ["Imbalance handling", md.class_imbalance_handling], ["Features", md.features.join(", ")]];
    $("#modelMeta").innerHTML = rows.map(([k,v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
    if(m.metrics){
      $("#valText").textContent = "Metrics are on the held-out split (" + m.metrics.validation + "). " + md.notes.join(" ");
      $("#metTable tbody").innerHTML = Object.entries(m.metrics.models).map(([n,v]) =>
        `<tr><td>${esc(n)}</td>` + ["pr_auc","precision","recall","f1","brier"].map(k => `<td>${v[k].toFixed(3)}</td>`).join("") + "</tr>").join("");
    }
    if(m.feature_importance && window.Chart){
      impChart && impChart.destroy();
      impChart = new Chart($("#impChart"), {type:"bar", data:{labels:m.feature_importance.map(f=>f.feature),
        datasets:[{label:"Permutation importance (PR-AUC drop)", data:m.feature_importance.map(f=>f.importance), backgroundColor:"#1c7c8c"}]},
        options:{indexAxis:"y", maintainAspectRatio:false, plugins:{legend:{display:false}}}});
    }
  }catch(e){ showError(e.message); }
}

buildForm();
loadDashboard();
