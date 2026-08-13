/* VRINGON Vision QA — demo app */
"use strict";

const DEFECT_LABELS = {
  upper_contamination: "어퍼 오염",
  bottom_contamination: "밑창 오염",
  excess_cement: "접착제 과다",
  loose_thread: "루즈 스레드 (실밥)",
  toe_scuff: "토 스커프",
  surface_anomaly: "미분류 표면 이상",
};
const PART_LABELS = {
  upper: "어퍼", toe: "토", heel: "힐", collar: "칼라",
  midsole: "미드솔", outsole: "아웃솔", cement_boundary: "접착 경계",
};
const PART_HEX = {
  upper: "#87B1FA", toe: "#8793FF", heel: "#232BC1", collar: "#444AE8",
  midsole: "#24DD9C", outsole: "#008454", cement_boundary: "#FFAB5C",
};
const SEV_COLOR = { major: "#FF7474", minor: "#FFAB5C", review: "#5D6CFA" };
const VERDICT_TXT = {
  pass: ["PASS", "허용 기준 내 — 결함 미검출"],
  review: ["REVIEW", "확인 필요 — 임계 근접 이상 감지"],
  fail: ["FAIL", "출하 불가 — Major 결함 검출"],
  unknown: ["판정 불가", "동급 레퍼런스 없음 — 아래 표시는 참고용"],
};

const $ = (id) => document.getElementById(id);
const state = { samples: [], sel: null, layers: { parts: false, heat: true, boxes: true, gt: false }, live: false, uploads: [] };

async function jget(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(url);
  return r.json();
}

/* ---------- boot ---------- */
(async function boot() {
  let manifest = { samples: [] }, metrics = null, bench = null;
  try { manifest = await jget("assets/samples/manifest.json"); } catch (e) {}
  try { metrics = await jget("assets/samples/metrics.json"); } catch (e) {}
  try { bench = await jget("assets/visa/benchmark.json"); } catch (e) {}
  state.samples = manifest.samples || [];
  renderKpis(metrics, bench);
  renderGallery();
  renderBench(bench, metrics);
  if (state.samples.length) select(state.samples[0].id);
  initLayers();
  initUpload();
  checkLive();
})();

/* ---------- KPI strip ---------- */
/* The KPI strip is a two-track comparison, because the demo's real finding is
   that the same engine is only decisive when acquisition is controlled. */
const KPI_ROWS = [
  ["이미지 판별 AUROC", (t) => t.image_auroc?.toFixed(2), "정상/불량 구분", 0.85],
  ["결함 우세율", (t) => pctOf(t.defect_dominates), "결함이 오검출보다 강함", 0.85],
  ["1순위 적중", (t) => pctOf(t.top1_region_is_gt), "최상위 표시가 진짜 결함", 0.85],
  ["정상 PASS", (t) => pctOf(t.normal_pass_rate), "정상을 정상이라 판정", 0.85],
  ["Pixel AUROC", (t) => t.mean_pixel_auroc?.toFixed(3), "히트맵 위치 정확도", 0.9],
];
const pctOf = (v) => (v == null ? null : (v * 100).toFixed(0) + "%");
const numOf = (s) => (s == null ? NaN : parseFloat(s));

function renderKpis(m, b) {
  const el = $("kpiStrip");
  const T = m?.tracks || {};
  const cols = [["rig", "고정 리그"], ["field", "자유 촬영"]].filter(([k]) => T[k]);
  if (!cols.length) { el.innerHTML = ""; return; }
  const head = cols.map(([, l]) => `<th>${l}</th>`).join("");
  const rows = KPI_ROWS.map(([label, get, sub, good]) => {
    const cells = cols.map(([k]) => {
      const v = get(T[k]);
      if (v == null) return "<td>—</td>";
      const n = numOf(v) > 1.5 ? numOf(v) / 100 : numOf(v);
      return `<td class="${n >= good ? "ok" : "bad"}">${v}</td>`;
    }).join("");
    return `<tr><th scope="row">${label}<em>${sub}</em></th>${cells}</tr>`;
  }).join("");
  const visa = b ? `<tr><th scope="row">VisA 실측 AUROC<em>실제 공장 불량 · 통제 촬영</em></th>
      <td class="ok" colspan="${cols.length}">${b.image_auroc.toFixed(3)} / ${b.pixel_auroc.toFixed(3)} (image / pixel)</td></tr>` : "";
  el.innerHTML = `<table class="cmp"><thead><tr><th>지표</th>${head}</tr></thead>
    <tbody>${rows}${visa}</tbody></table>`;
  const note = $("limitNote");
  if (note && m?.limitation) {
    note.innerHTML = `<b>같은 엔진, 다른 촬영 조건</b> ${m.limitation}`;
    note.hidden = false;
  }
}

/* ---------- gallery ---------- */
function sampleTitle(s) {
  if (s.kind === "normal") return "정상 (홀드아웃)";
  if (s.kind === "upload") return s.name || "업로드 이미지";
  const d = s.gt?.type || s.id.split("_").slice(3).join("_");
  return DEFECT_LABELS[d] || d;
}
const TRACK_LABEL = {
  rig: "고정 리그 (통제 촬영)",
  field: "자유 촬영 (공개 사진)",
};
function renderGallery() {
  const el = $("galleryList");
  const groups = {};
  for (const s of state.uploads) (groups["업로드"] = groups["업로드"] || []).push(s);
  const ordered = [...state.samples].sort(
    (a, b) => (a.track === "rig" ? 0 : 1) - (b.track === "rig" ? 0 : 1));
  for (const s of ordered) {
    const g = `${TRACK_LABEL[s.track] || "샘플"} · ${s.sku}`;
    (groups[g] = groups[g] || []).push(s);
  }
  el.innerHTML = Object.entries(groups).map(([sku, list]) => `
    <div class="grp__t">${sku}</div>
    ${list.map((s) => `
      <button class="thumb ${state.sel === s.id ? "is-sel" : ""} ${s.isNew ? "is-new" : ""}" data-id="${s.id}">
        <img src="${s.baseUrl || `assets/samples/${s.id}/image.jpg`}" alt="">
        <span class="thumb__meta">
          <span class="thumb__name">${sampleTitle(s)}</span>
          <span class="thumb__sub">${s.kind === "defect" ? "주입 결함 · GT 보유" : s.kind === "upload" ? "실시간 추론" : "정상 이미지"}</span>
        </span>
        <span class="dot ${s.verdict}"></span>
      </button>`).join("")}`).join("");
  el.querySelectorAll(".thumb").forEach((b) => b.addEventListener("click", () => select(b.dataset.id)));
}

/* ---------- viewer + inspector ---------- */
function findSample(id) {
  return state.uploads.find((s) => s.id === id) || state.samples.find((s) => s.id === id);
}
function select(id) {
  state.sel = id;
  const s = findSample(id);
  if (!s) return;
  const base = s.baseUrl || `assets/samples/${s.id}/image.jpg`;
  $("imgBase").src = base;
  $("imgParts").src = s.partsUrl || `assets/samples/${s.id}/parts.png`;
  $("imgHeat").src = s.heatUrl || `assets/samples/${s.id}/heat.png`;
  const hasGt = s.gt != null && s.kind === "defect";
  // src="" would make the browser re-request the page itself as an image
  if (hasGt) $("imgGt").src = `assets/samples/${s.id}/gt.png`;
  else $("imgGt").removeAttribute("src");
  document.querySelector('[data-layer="gt"]').style.display = hasGt ? "" : "none";
  renderBoxes(s);
  applyLayers();
  renderInspector(s);
  renderGallery();
}
function renderBoxes(s) {
  const svg = $("boxLayer");
  svg.setAttribute("viewBox", `0 0 ${s.w} ${s.h}`);
  svg.innerHTML = (s.detections || []).map((d, i) => {
    const [x, y, w, h] = d.bbox;
    const c = SEV_COLOR[d.severity] || "#5D6CFA";
    const label = (DEFECT_LABELS[d.type] || d.type).split(" ")[0];
    const ty = y > 16 ? y - 5 : y + h + 13;
    return `<rect x="${x}" y="${y}" width="${w}" height="${h}" stroke="${c}" rx="2"/>
      <text x="${x}" y="${ty}" fill="${c}">${label} ${(d.confidence * 100).toFixed(0)}%</text>`;
  }).join("");
}
function renderInspector(s) {
  const [vt, vs] = VERDICT_TXT[s.verdict] || ["—", ""];
  const card = $("verdictCard");
  card.className = "verdict " + s.verdict;
  $("verdictLabel").textContent = vt;
  $("verdictSub").textContent = vs + (s.kind === "defect" ? ` · 주입 결함: ${DEFECT_LABELS[s.gt?.type] || ""}` : "");
  $("detCount").textContent = s.detections?.length ? `${s.detections.length}건` : "";
  $("detList").innerHTML = s.detections?.length ? s.detections.map((d) => `
    <div class="det ${d.severity}">
      <div class="det__row">
        <span class="det__type">${DEFECT_LABELS[d.type] || d.type}</span>
        <span class="det__sev">${d.severity.toUpperCase()}</span>
      </div>
      <div class="det__meta">부위 <b>${PART_LABELS[d.part] || d.part}</b> · 신뢰도 <b>${(d.confidence * 100).toFixed(0)}%</b> · 면적 <b>${d.area_pct}%</b> · z<sub>max</sub> <b>${d.z_max}</b></div>
    </div>`).join("") : `<div class="empty">검출된 결함이 없습니다.</div>`;

  $("gtPanel").innerHTML = s.gt ? `
    <div>주입 결함 <b>${DEFECT_LABELS[s.gt.type] || s.gt.type}</b></div>
    <div>위치 검출 ${s.gt.localized ? '<span class="ok">성공</span>' : '<span class="no">실패</span>'} ·
         유형 일치 ${s.gt.type_match ? '<span class="ok">일치</span>' : '<span class="no">불일치</span>'}</div>
    <div>Pixel AUROC <b>${s.gt.pixel_auroc ?? "—"}</b> · IoU <b>${s.gt.iou}</b></div>`
    : `<div class="empty">${s.kind === "normal" ? "정상 홀드아웃 샘플 — 결함이 없어야 정상입니다." : "실시간 업로드 — 정답 마스크가 없습니다."}</div>`;

  const order = ["upper", "toe", "collar", "heel", "midsole", "outsole", "cement_boundary"];
  $("partBars").innerHTML = order.map((p) => {
    const v = s.parts_pct?.[p] ?? 0;
    return `<div class="bar"><span>${PART_LABELS[p]}</span>
      <span class="bar__track"><span class="bar__fill" style="--c:${PART_HEX[p]};width:${Math.min(100, v * 1.4)}%"></span></span>
      <span class="bar__v">${v}%</span></div>`;
  }).join("");

  const REF_MODE = {
    golden: ["골든 샘플 일치", "동일 개체 수준 레퍼런스 — 판정 신뢰 가능", "ok"],
    near: ["근접 레퍼런스", "다른 개체 — 판정은 참고용", "warn"],
    mismatch: ["레퍼런스 불일치", "판정 불가 — 위치 표시만 참고", "warn"],
  };
  const rm = REF_MODE[s.ref_mode];
  $("infInfo").innerHTML = `
    ${rm ? `<div><span>레퍼런스 적합도</span><b class="${rm[2]}">${rm[0]} (${s.ref_sim_top1})</b></div>
            <div class="kv__note">${rm[1]}</div>` : ""}
    <div><span>SKU / 레퍼런스</span><b>${s.sku || "—"} · ${s.colorway_refs ?? "—"}장</b></div>
    <div><span>이상도 z (평균/최대)</span><b>${s.z_mean} / ${s.z_max}</b></div>
    <div><span>메모리 뱅크 구축</span><b>${s.t_bank_s ?? "—"} s</b></div>
    <div><span>추론 시간 (CPU)</span><b>${s.t_infer_s} s</b></div>
    <div><span>엔진</span><b>PatchCore · WideResNet50</b></div>`;
}

/* ---------- layers ---------- */
function initLayers() {
  document.querySelectorAll("#layerChips .chip").forEach((c) =>
    c.addEventListener("click", () => {
      const k = c.dataset.layer;
      state.layers[k] = !state.layers[k];
      c.classList.toggle("is-on", state.layers[k]);
      applyLayers();
    }));
  $("opacity").addEventListener("input", applyLayers);
}
function applyLayers() {
  const op = $("opacity").value / 100;
  const map = { parts: "imgParts", heat: "imgHeat", gt: "imgGt", boxes: "boxLayer" };
  for (const [k, id] of Object.entries(map)) {
    const el = $(id);
    el.classList.toggle("is-on", !!state.layers[k]);
    if (id !== "boxLayer") el.style.opacity = op;
  }
}

/* ---------- benchmark section ---------- */
function renderBench(b, m) {
  const grid = $("benchGrid");
  const cards = [];
  if (b) {
    cards.push([`${b.image_auroc.toFixed(3)}`, "VisA cashew · Image AUROC", "good"]);
    cards.push([`${b.pixel_auroc.toFixed(3)}`, "VisA cashew · Pixel AUROC", "good"]);
    cards.push([`${b.n_test_normal + b.n_test_anomaly}`, `실측 테스트 (정상 ${b.n_test_normal} · 불량 ${b.n_test_anomaly})`, ""]);
    cards.push([`${b.n_bank}`, "메모리 뱅크 정상 이미지", ""]);
  }
  const rig = m?.tracks?.rig, fld = m?.tracks?.field;
  if (rig?.image_auroc != null) cards.push([`${rig.image_auroc}`, "고정 리그 · Image AUROC (판정)", "good"]);
  if (fld?.image_auroc != null) cards.push([`${fld.image_auroc}`, "자유 촬영 · Image AUROC (판정)", "bad"]);
  grid.innerHTML = cards.map(([v, t, c]) =>
    `<div class="bench__card ${c}"><b>${v}</b><span>${t}</span></div>`).join("");
  $("benchExamples").innerHTML = (b?.examples || []).map((k) => `
    <div class="bx"><div class="bx__pair">
      <img src="assets/visa/${k}_orig.jpg" alt=""><img src="assets/visa/${k}_heat.jpg" alt="">
    </div><span>실측 불량 #${k} — 원본 · 검출 히트맵</span></div>`).join("");
}

/* ---------- live upload ---------- */
async function checkLive() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    if (r.ok) {
      state.live = true;
      $("modeBadge").textContent = "LIVE";
      $("modeBadge").classList.add("is-live");
      $("uploadHint").textContent = "클릭 또는 드래그 — 실시간 AI 추론";
      return;
    }
  } catch (e) {}
  $("uploadHint").innerHTML =
    "정적 모드 — 아래 샘플은 모두 사전 추론 결과입니다.<br>실시간 업로드 추론은 저장소를 받아 " +
    "<code>python pipeline/server.py</code> 실행 시 동작합니다.";
  $("uploadZone").classList.add("is-off");
}
function initUpload() {
  const zone = $("uploadZone"), input = $("fileInput");
  zone.addEventListener("click", () => { if (state.live) input.click(); });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("is-drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault(); zone.classList.remove("is-drag");
    if (state.live && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files[0]) upload(input.files[0]); });
}
async function upload(file) {
  const hint = $("uploadHint");
  hint.innerHTML = '<span class="spin"></span> 추론 중… (CPU, 수십 초 소요)';
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const res = await r.json();
    const rec = res.record;
    rec.kind = "upload"; rec.isNew = true; rec.name = file.name;
    rec.baseUrl = res.images.image; rec.partsUrl = res.images.parts; rec.heatUrl = res.images.heat;
    state.uploads.unshift(rec);
    renderGallery();
    select(rec.id);
    hint.textContent = "완료 — 갤러리에서 확인하세요";
  } catch (e) {
    hint.textContent = "추론 실패: " + e.message;
  }
}
