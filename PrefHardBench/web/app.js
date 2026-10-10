const REASONS_U = {
  a1: "Almost no preference-relevant visible difference",
  a2: "Visible differences, but no clear winner",
  a3: "Each image is better in a different aspect",
  a4: "The prompt has multiple reasonable interpretations",
  a5: "The comparison mainly depends on personal taste",
  a6: "Both images have comparable defects",
  a7: "Other difficulty that cannot be specified"
};
const REASONS_PREF = {
  b1: "Prompt alignment clearly determines the choice",
  b2: "Visual quality clearly determines the choice",
  b3: "Composition, aesthetics, or style clearly determines the choice",
  b4: "One side wins after a cross-dimension trade-off",
  b5: "A visible error clearly disqualifies the other side",
  b6: "The images are close, but one has a slight overall advantage",
  b7: "The choice comes mainly from an overall first impression"
};
const state = { question: null, preference: null, difficulty: null, confidence: null, annotatorId: null, started: null };
const reviewState = { index: 0, total: 0 };

const $ = (id) => document.getElementById(id);
function randomId() { return `anon-${crypto.randomUUID().slice(0, 10)}`; }
function setError(message) { $("error").textContent = message || ""; $("error").classList.toggle("hidden", !message); }
function makeRatings(containerId, type) {
  const container = $(containerId);
  for (let i = 1; i <= 5; i++) {
    const button = document.createElement("button");
    button.type = "button"; button.className = "rating"; button.textContent = i; button.dataset.value = i;
    button.addEventListener("click", () => {
      state[type] = i;
      [...container.children].forEach(el => el.classList.toggle("selected", Number(el.dataset.value) === i));
      updateSubmitState();
    });
    container.appendChild(button);
  }
}
function renderReasons() {
  const preference = state.preference;
  const options = preference === "U" ? REASONS_U : REASONS_PREF;
  const container = $("reason-options"); container.replaceChildren();
  $("reason-title").textContent = preference === "U" ? "What prevented a stable preference?" : "What evidence made you choose this side?";
  for (const [code, label] of Object.entries(options)) {
    const wrapper = document.createElement("label"); wrapper.className = "reason-option";
    const input = document.createElement("input"); input.type = preference === "U" ? "checkbox" : "radio"; input.name = "reason"; input.value = code;
    wrapper.append(input, document.createTextNode(label)); container.appendChild(wrapper);
  }
  updateSubmitState();
}
function selectedReasons() { return [...document.querySelectorAll('input[name="reason"]:checked')].map(el => el.value); }
function updateSubmitState() { $("submit-button").disabled = !(state.preference && state.difficulty && state.confidence && selectedReasons().length); }
function showQuestion(payload) {
  state.question = payload.question; state.preference = null; state.difficulty = null; state.confidence = null; state.started = performance.now();
  $("question-id").textContent = payload.question.question_id;
  $("task-count").textContent = `${payload.completed} completed / ${payload.total} total`;
  $("progress").textContent = `${payload.completed} / ${payload.total}`;
  $("prompt").textContent = payload.question.prompt_en;
  $("image-left").src = payload.question.image_left_url; $("image-right").src = payload.question.image_right_url;
  document.querySelectorAll("[data-preference]").forEach(button => { button.classList.remove("selected"); button.onclick = () => { state.preference = button.dataset.preference; document.querySelectorAll("[data-preference]").forEach(b => b.classList.toggle("selected", b === button)); renderReasons(); }; });
  $("difficulty-row").replaceChildren(); $("confidence-row").replaceChildren(); makeRatings("difficulty-row", "difficulty"); makeRatings("confidence-row", "confidence");
  $("note").value = ""; $("error").classList.add("hidden"); renderReasons();
  $("task").classList.remove("hidden"); $("done").classList.add("hidden");
}
async function loadNext() {
  const response = await fetch(`/api/next?annotator_id=${encodeURIComponent(state.annotatorId)}`);
  const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "Unable to load a question");
  if (payload.done) { $("task").classList.add("hidden"); $("done").classList.remove("hidden"); $("progress").textContent = `${payload.completed} / ${payload.total}`; return; }
  showQuestion(payload);
}
async function submit() {
  const body = { annotator_id: state.annotatorId, question_id: state.question.question_id, shown_order: state.question.shown_order, preference: state.preference, difficulty: state.difficulty, confidence: state.confidence, reason: selectedReasons(), note: $("note").value, elapsed_seconds: (performance.now() - state.started) / 1000 };
  $("submit-button").disabled = true; setError("");
  try { const response = await fetch("/api/submit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "Unable to submit"); await loadNext(); }
  catch (error) { setError(error.message); updateSubmitState(); }
}

function setReviewText(id, value) { $(id).textContent = value || "-"; }
function renderReview(payload) {
  const row = payload.annotation;
  reviewState.index = payload.index; reviewState.total = payload.total;
  setReviewText("review-record", `Record ${payload.index + 1} / ${payload.total}`);
  setReviewText("review-question", `Question ${row.question_id || "-"}`);
  setReviewText("review-prompt-en", row.prompt_en);
  setReviewText("review-prompt-zh", row.prompt_zh);
  $("review-image-left").src = row.image_left_url;
  $("review-image-right").src = row.image_right_url;
  $("review-result").innerHTML = `
    <span class="result-chip result-${String(row.judge || "").toLowerCase()}">Judge: ${row.judge || "-"}</span>
    <span class="result-chip">Confidence: ${row.confidence || "-"}/5</span>
    <span class="result-chip">Reason: ${row.reason || "-"}</span>`;
  setReviewText("review-session", row.session_id);
  setReviewText("review-slot", row.slot);
  setReviewText("review-chosen", row.chosen);
  setReviewText("review-rejected", row.rejected);
  setReviewText("review-reason-other", row.reason_other);
  $("review-index").value = payload.index + 1;
  $("review-prev").disabled = payload.index === 0;
  $("review-next").disabled = payload.index === payload.total - 1;
}
async function loadReview(index) {
  const response = await fetch(`/api/annotations?index=${encodeURIComponent(index)}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Unable to load annotation");
  renderReview(payload);
}
async function startReview() {
  $("identity-panel").classList.add("hidden");
  $("instructions").classList.add("hidden");
  $("task").classList.add("hidden");
  $("done").classList.add("hidden");
  $("review").classList.remove("hidden");
  $("progress").textContent = "Review mode";
  try { await loadReview(0); }
  catch (error) { $("review-error").textContent = error.message; $("review-error").classList.remove("hidden"); }
}

$("review-prev").addEventListener("click", () => loadReview(reviewState.index - 1));
$("review-next").addEventListener("click", () => loadReview(reviewState.index + 1));
$("review-go").addEventListener("click", () => {
  const requested = Number($("review-index").value) - 1;
  if (Number.isInteger(requested) && requested >= 0 && requested < reviewState.total) loadReview(requested);
});
$("review-index").addEventListener("keydown", (event) => { if (event.key === "Enter") $("review-go").click(); });

$("annotator-id").value = localStorage.getItem("prefhard_annotator_id") || randomId();
if (new URLSearchParams(window.location.search).get("mode") === "review") startReview();
$("start-button").addEventListener("click", async () => { const value = $("annotator-id").value.trim(); if (!value) return setError("Enter an anonymous annotator ID."); state.annotatorId = value; localStorage.setItem("prefhard_annotator_id", value); $("identity-panel").classList.add("hidden"); $("instructions").classList.add("hidden"); try { await loadNext(); } catch (error) { $("identity-panel").classList.remove("hidden"); setError(error.message); } });
$("submit-button").addEventListener("click", submit);
