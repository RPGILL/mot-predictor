const form=document.getElementById("registration-form");
const registrationInput=document.getElementById("registration");
const submitButton=document.getElementById("submit-button");
const statusBox=document.getElementById("status");
const results=document.getElementById("results");

function normaliseRegistration(v){return v.toUpperCase().replace(/[^A-Z0-9]/g,"");}
function setStatus(m,t){statusBox.textContent=m;statusBox.className=`status ${t}`;}
function clearStatus(){statusBox.textContent="";statusBox.className="status hidden";}
function formatPercent(v){if(v===null||v===undefined||Number.isNaN(Number(v)))return "—";return `${Number(v).toFixed(1)}%`;}
function formatDate(v){if(!v)return "—";const d=new Date(`${v}T00:00:00`);if(Number.isNaN(d.getTime()))return v;return d.toLocaleDateString("en-GB",{day:"2-digit",month:"short",year:"numeric"});}
function riskClass(r){const v=String(r||"").toLowerCase();if(v==="low")return"risk-low";if(v==="moderate")return"risk-moderate";if(v==="elevated")return"risk-elevated";return"risk-high";}
function escapeHtml(v){const d=document.createElement("div");d.textContent=String(v??"");return d.innerHTML;}

function renderIssueCard(item,styleClass){
  const a=document.createElement("article");
  a.className=`issue-card ${styleClass}`;
  const reasons=item?.evidence?.reasons||[];
  const advisoryTexts=item?.evidence?.latest_advisory_texts||[];
  const reasonsHtml=reasons.length?`<ul class="evidence-list">${reasons.map(r=>`<li>${escapeHtml(r)}</li>`).join("")}</ul>`:`<p class="muted">No additional evidence text available.</p>`;
  const advisoriesHtml=advisoryTexts.length?`<div class="latest-advisory"><strong>Latest MOT:</strong><br>${advisoryTexts.map(t=>escapeHtml(t)).join("<br>")}</div>`:"";
  a.innerHTML=`<div class="issue-topline"><h3>${escapeHtml(item.label||item.category||"Unknown")}</h3><span class="score-chip">model score ${Number(item.model_score||0).toFixed(2)}</span></div>${reasonsHtml}${advisoriesHtml}<p class="recommended-check">${escapeHtml(item.recommended_check||"")}</p>`;
  return a;
}

function renderCollection(containerId,emptyId,items,styleClass){
  const c=document.getElementById(containerId),e=document.getElementById(emptyId);
  c.innerHTML="";
  if(!Array.isArray(items)||items.length===0){e.classList.remove("hidden");return;}
  e.classList.add("hidden");
  items.forEach(i=>c.appendChild(renderIssueCard(i,styleClass)));
}

function renderDevelopmentSignals(data){
  const c=document.getElementById("development-signals");c.innerHTML="";
  const signals=data?.top_model_only_signals||[];
  if(!signals.length){c.textContent="No model-only signals.";return;}
  signals.forEach(item=>{const row=document.createElement("div");row.className="dev-signal";row.textContent=`${item.label}: score ${Number(item.model_score).toFixed(3)}, threshold ${Number(item.warning_threshold).toFixed(2)}`;c.appendChild(row);});
}

function renderReport(report){
  const v=report.vehicle||{},r=report.readiness||{},ctx=report.prediction_context||{},h=report.history_snapshot||{};
  document.getElementById("vehicle-title").textContent=`${v.registration||"—"} · ${v.make||""} ${v.model||""}`.trim();
  document.getElementById("vehicle-meta").textContent=[v.fuel_type,v.first_used_date?`First used ${v.first_used_date}`:null].filter(Boolean).join(" · ");
  const pill=document.getElementById("risk-pill");pill.textContent=`${r.risk_band||"—"} RISK`;pill.className=`risk-pill ${riskClass(r.risk_band)}`;
  const fail=Number(r.fail_probability_percent||0),pass=Number(r.pass_probability_percent||0);
  document.getElementById("fail-percent").textContent=formatPercent(fail);
  document.getElementById("pass-percent").textContent=formatPercent(pass);
  document.getElementById("fail-percent-text").textContent=formatPercent(fail);
  document.getElementById("warning-threshold").textContent=formatPercent(Number(r.warning_threshold||0)*100);
  document.getElementById("risk-gauge").style.setProperty("--risk",`${Math.max(0,Math.min(360,fail*3.6))}deg`);
  document.getElementById("summary").textContent=report.summary||"";
  document.getElementById("latest-mot-date").textContent=formatDate(ctx.latest_mot_date);
  document.getElementById("mot-expiry-date").textContent=formatDate(ctx.mot_expiry_date);
  document.getElementById("prediction-horizon").textContent=ctx.mot_interval_days_used!=null?`${ctx.mot_interval_days_used} days`:"—";
  document.getElementById("historical-failure-rate").textContent=h.historical_failure_rate!=null?formatPercent(Number(h.historical_failure_rate)*100):"—";
  document.getElementById("recent-failures").textContent=h.failures_last_3_events!=null?`${h.failures_last_3_events} of last 3`:"—";
  document.getElementById("latest-advisories").textContent=h.latest_advisory_count!=null?String(h.latest_advisory_count):"—";
  renderCollection("likely-failure-areas","likely-empty",report.likely_failure_areas,"priority");
  renderCollection("current-advisories","advisory-empty",report.current_advisory_checks,"advisory");
  renderCollection("watchlist","watchlist-empty",report.watchlist,"watch");
  renderDevelopmentSignals(report.development_signals);
  document.getElementById("important-note").textContent=report.important_note||"";
  results.classList.remove("hidden");
}

async function fetchReport(registration){
  const response=await fetch(`/vehicle/${encodeURIComponent(registration)}/report`,{headers:{Accept:"application/json"}});
  if(!response.ok){let detail="Unable to generate MOT report.";try{const payload=await response.json();if(payload?.detail)detail=typeof payload.detail==="string"?payload.detail:JSON.stringify(payload.detail);}catch(_){}throw new Error(detail);}
  return response.json();
}

form.addEventListener("submit",async event=>{
  event.preventDefault();
  const registration=normaliseRegistration(registrationInput.value);
  if(!registration){setStatus("Enter a valid vehicle registration.","error");return;}
  registrationInput.value=registration;results.classList.add("hidden");submitButton.disabled=true;submitButton.textContent="Checking…";setStatus(`Retrieving MOT history and generating report for ${registration}…`,"loading");
  try{renderReport(await fetchReport(registration));clearStatus();}
  catch(error){setStatus(error?.message||"Unable to generate MOT report.","error");}
  finally{submitButton.disabled=false;submitButton.textContent="Check MOT readiness";}
});
