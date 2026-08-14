/* VRINGON Vision QA — Air Max 90 lasted-upper inspection report */
"use strict";

const $ = (id) => document.getElementById(id);
const S = { manifest: null, sel: null, rec: null, layer: "measure", itemSel: null,
            track: "photo", tracks: {} };

/* Two acquisition tracks, because they answer different questions:
   photo — real product photographs, lateral only, no scale
   gso   — orthographic renders of metric CC BY 4.0 meshes: every camera view
           the spec needs, and millimetres that can be checked against truth */
const TRACKS = {
  photo: { base: "assets/inspect", label: "제품 사진 (측면)" },
  gso: { base: "assets/gso", label: "GSO 렌더 (4뷰 · mm 실측)" },
};
const base = () => TRACKS[S.track].base;

const STATUS_LABEL = {
  MEASURED: ["계측", "ok"],
  ADVISORY: ["참고", "warn"],
  NOT_SENSED: ["미계측", "off"],
  NO_GOLDEN: ["기준없음", "off"],
};
const VERDICT_LABEL = {
  PASS: ["PASS", "허용 범위 내"],
  REVIEW: ["REVIEW", "판정 유보 — 불확도 구간"],
  FAIL: ["FAIL", "허용 범위 밖"],
  NO_VERDICT: ["판정 없음", "계측된 항목이 없습니다"],
};
const FEAS_NOTE = {
  measured: "현재 뷰에서 계측",
  needs_view: "다른 카메라 뷰 필요",
  needs_3d: "단일 2D로는 원리상 불가",
  advisory: "개체 계측이 아닌 소재 능력 판정",
};

async function jget(u) {
  const r = await fetch(u, { cache: "no-store" });
  if (!r.ok) throw new Error(u + " " + r.status);
  return r.json();
}

(async function boot() {
  for (const [k, t] of Object.entries(TRACKS)) {
    try { S.tracks[k] = await jget(t.base + "/manifest.json"); } catch (e) {}
  }
  if (!Object.keys(S.tracks).length) {
    $("gallery").innerHTML = `<div class="empty">검사 결과를 불러오지 못했습니다.</div>`;
    return;
  }
  S.track = S.tracks.photo ? "photo" : Object.keys(S.tracks)[0];
  S.manifest = S.tracks[S.track];
  renderTrackBar();
  renderSpec();
  renderGallery();
  if (S.manifest.samples.length) select(S.manifest.samples[0].id);
  $("layerChips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-layer]");
    if (!b) return;
    S.layer = b.dataset.layer;
    [...$("layerChips").querySelectorAll(".chip")].forEach((c) =>
      c.classList.toggle("is-on", c.dataset.layer === S.layer));
    $("shot").src = `${base()}/${S.sel}/${S.layer === "measure" ? "measure" : "image"}.jpg`;
  });
})();

/* ---------- acquisition track switch ---------- */
function renderTrackBar() {
  const el = $("trackBar");
  el.innerHTML = Object.entries(TRACKS).filter(([k]) => S.tracks[k]).map(
    ([k, t]) => `<button class="chip ${S.track === k ? "is-on" : ""}" data-track="${k}">${t.label}</button>`
  ).join("") + `<span class="trackinfo" id="trackInfo"></span>`;
  el.querySelectorAll("[data-track]").forEach((b) =>
    b.addEventListener("click", async () => {
      S.track = b.dataset.track;
      S.manifest = S.tracks[S.track];
      S.itemSel = null;
      renderTrackBar();
      renderSpec();
      renderGallery();
      if (S.manifest.samples.length) await select(S.manifest.samples[0].id);
    }));
  const m = S.manifest;
  const gt = m.ground_truth, ds = m.dataset;
  $("trackInfo").innerHTML = gt
    ? `실측 대조 — 기준길이 추정 <b>평균 절대오차 ${gt.mean_abs_err_pct}%</b> ` +
      `(p95 ${gt.p95_abs_err_pct}%, n=${gt.n}) · ${ds.license}<br>` +
      `<i>${ds.caveat}</i>`
    : `제품 사진은 스케일 기준이 없어 mm 환산이 불가능하며, 측면 뷰만 존재합니다.`;
}

/* ---------- spec header: the eleven items and the rig they came from ---------- */
function renderSpec() {
  const sp = S.manifest.spec;
  $("specSource").textContent = sp.source;
  const rig = sp.rig;
  $("rigNote").innerHTML =
    `2D 카메라 <b>${rig.cameras_2d}대</b> · 3D 카메라 <b>${rig.cameras_3d}대</b> · ` +
    `1사이클 <b>${rig.cycle_sec}초</b><br>` + rig.notes.map((n) => `· ${n}`).join("<br>");
  $("camList").innerHTML = Object.entries(sp.cameras).map(
    ([k, v]) => `<span class="cam"><b>${k}</b>${v.name}</span>`).join("");
}

/* ---------- gallery ---------- */
function renderGallery() {
  const g = {};
  for (const s of S.manifest.samples) (g[s.sku] = g[s.sku] || []).push(s);
  const label = (s) => (S.track === "gso"
    ? (s.view || "") + (s.length_mm ? ` · ${s.length_mm}mm` : "")
    : s.id.split("_").slice(1).join("_"));
  $("gallery").innerHTML = Object.entries(g).map(([sku, list]) => `
    <div class="grp">${sku.replace(/_/g, " ").slice(0, 34)}</div>
    ${list.map((s) => `
      <button class="thumb ${S.sel === s.id ? "is-sel" : ""}" data-id="${s.id}">
        <img src="${base()}/${s.id}/image.jpg" alt="">
        <span class="thumb__m"><span>${label(s)}</span>
        <span class="v ${s.verdict}">${s.verdict}</span></span>
      </button>`).join("")}`).join("");
  $("gallery").querySelectorAll(".thumb").forEach(
    (b) => b.addEventListener("click", () => select(b.dataset.id)));
}

async function select(id) {
  S.sel = id;
  S.itemSel = null;
  S.rec = await jget(`${base()}/${id}/result.json`);
  $("shot").src = `${base()}/${id}/${S.layer === "measure" ? "measure" : "image"}.jpg`;
  renderVerdict();
  renderTable();
  renderGallery();
}

function renderVerdict() {
  const [t, sub] = VERDICT_LABEL[S.rec.verdict] || ["—", ""];
  const c = $("verdict");
  c.className = "verdict " + S.rec.verdict;
  $("verdictLabel").textContent = t;
  const s = S.rec.summary;
  $("verdictSub").innerHTML =
    `${sub} · 계측 <b>${s.n_measured}</b> · 참고 <b>${s.n_advisory}</b> · 미계측 <b>${s.n_not_sensed}</b>`;
  // the photo track groups golden statistics by style, the GSO track by camera
  // view, so the count has to be derived rather than read from one fixed field
  const g = S.manifest.golden[S.rec.sku] || S.manifest.golden[S.rec.view] || {};
  const perItem = g.stats || g;
  const n = g.n_golden != null ? g.n_golden
    : Math.max(0, ...Object.values(perItem).map((x) => (x && x.n) || 0));
  $("goldenNote").innerHTML =
    `골든 샘플 <b>${n}장</b>에서 항목별 중앙값과 로버스트 편차(1.4826×MAD)를 구하고, ` +
    `허용 범위를 <b>중앙값 ± ${S.manifest.k_sigma}σ</b>로 잡았습니다. ` +
    `측정 불확도만큼의 가드밴드 안쪽은 <b>REVIEW</b>로 두어, 측정이 구분하지 못하는 구간을 ` +
    `합격이나 불합격으로 단정하지 않습니다.`;
}

/* ---------- the eleven items ---------- */
function bar(r) {
  if (r.tol_lower == null || r.measured == null) return "";
  const lo = r.tol_lower, hi = r.tol_upper, c = r.nominal;
  const span = Math.max(hi - lo, 1e-6);
  const pad = span * 0.55;
  const x = (v) => (100 * (v - (lo - pad))) / (span + 2 * pad);
  const p = Math.max(0, Math.min(100, x(r.measured)));
  const u = r.uncertainty ? (100 * r.uncertainty) / (span + 2 * pad) : 0;
  return `<span class="tolbar">
    <span class="tolbar__band" style="left:${x(lo)}%;width:${x(hi) - x(lo)}%"></span>
    <span class="tolbar__c" style="left:${x(c)}%"></span>
    ${u > 0.3 ? `<span class="tolbar__u" style="left:${p - u}%;width:${2 * u}%"></span>` : ""}
    <span class="tolbar__v ${r.verdict || ""}" style="left:${p}%"></span>
  </span>`;
}

function renderTable() {
  $("itemTable").innerHTML = S.rec.items.map((r) => {
    const [sl, sc] = STATUS_LABEL[r.status] || [r.status, "off"];
    const val = r.measured == null ? "—"
      : `${r.measured.toFixed(r.units === "%" ? 1 : 2)}<em>${r.units === "%" ? "%" : "‰"}</em>` +
        (r.mm != null ? `<b class="mm">${r.mm} mm</b>` : "");
    const tol = r.tol_lower == null ? "—"
      : `${r.tol_lower.toFixed(1)} ~ ${r.tol_upper.toFixed(1)}`;
    return `<tr data-item="${r.item_id}" class="${S.itemSel === r.item_id ? "is-open" : ""}">
      <td class="n">${r.no}</td>
      <td class="nm"><b>${r.name_en}</b><em>${r.name_ko}</em></td>
      <td class="cam">${(r.cameras || []).join("<br>") || "—"}</td>
      <td class="st"><span class="pill ${sc}">${sl}</span></td>
      <td class="val">${val}${r.uncertainty ? `<i>±${r.uncertainty}</i>` : ""}</td>
      <td class="tol">${tol}${bar(r)}</td>
      <td class="vd"><span class="v ${r.verdict || "none"}">${r.verdict || "—"}</span></td>
    </tr>
    <tr class="detail ${S.itemSel === r.item_id ? "" : "hidden"}"><td colspan="7">
      <div class="dgrid">
        <div><span>원 사양 로직</span><p>${r.vendor_logic}</p></div>
        <div><span>본 데모 구현</span><p>${r.our_method}</p></div>
        <div><span>가능성</span><p>${FEAS_NOTE[r.feasibility] || r.feasibility}</p></div>
        ${r.note ? `<div class="wide"><span>비고</span><p>${r.note}</p></div>` : ""}
        ${r.geometry && !r.geometry.type
          ? `<div class="wide"><span>측정 지표</span><p>${
              Object.entries(r.geometry).map(([k, v]) => `${k} = <b>${v}</b>`).join(" · ")}</p></div>`
          : ""}
      </div></td></tr>`;
  }).join("");
  $("itemTable").querySelectorAll("tr[data-item]").forEach((tr) =>
    tr.addEventListener("click", () => {
      S.itemSel = S.itemSel === tr.dataset.item ? null : tr.dataset.item;
      renderTable();
    }));
}
