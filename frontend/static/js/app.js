/* ═══════════════════════════════════════════════════════════
   MOOCs Auto-Generator — app.js (v3.1)
   Unified input flow: audio + teacher video + syllabus + materials
   ═══════════════════════════════════════════════════════════ */
'use strict';

const S = {
  sid:          crypto.randomUUID(),
  ws:           null,
  audioFile:    null,
  teacherFile:  null,
  imageFiles:   [],
  templateFile: null,
  // derived
  transcript:   '',
  keyPoints:    [],
  slidesData:   [],
  script:       '',
  faceVideoPath:'',
};

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initWS();
  checkStatus();
  setInterval(checkStatus, 30_000);

  // LLM radio → API key
  document.querySelectorAll('input[name="llm"]').forEach(r =>
    r.addEventListener('change', () => {
      const show = r.value === 'openai' && r.checked;
      const k = document.getElementById('openaiKey');
      if (k) k.style.display = show ? '' : 'none';
    })
  );

  // Drag-over highlight
  document.querySelectorAll('.dz,.uz').forEach(dz => {
    dz.addEventListener('dragenter', e => { e.preventDefault(); dz.classList.add('dragover'); });
    dz.addEventListener('dragleave', e => { if (!dz.contains(e.relatedTarget)) dz.classList.remove('dragover'); });
    dz.addEventListener('drop', () => dz.classList.remove('dragover'));
  });

  // Toast dismiss
  document.getElementById('toasts')?.addEventListener('click', e => {
    e.target.closest('.toast')?.remove();
  });

  updateChecklist();
});

// ── WebSocket ─────────────────────────────────────────────────
function initWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  S.ws = new WebSocket(`${proto}://${location.host}/ws/${S.sid}`);
  S.ws.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.type === 'progress') onWsProg(d.percent, d.message, d.step || '');
  };
  S.ws.onclose = () => setTimeout(initWS, 2500);
}

// ── Status ────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const d   = await fetch('/api/status').then(r => r.json());
    const ok  = d.ollama || d.openai;
    const led = document.getElementById('sysLed');
    const txt = document.getElementById('sysTxt');
    const sb  = document.getElementById('statusBadge');
    const parts = [];
    if (d.ollama) parts.push('🦙 Ollama');
    if (d.openai) parts.push('🤖 OpenAI');
    if (d.stable_diffusion) parts.push('🎨 SD');
    if (d.demo_video_ready) parts.push('🎬 Demo');
    if (!ok) parts.push('⚠ Start Ollama');
    if (led) led.className = 'led ' + (ok ? 'ok' : 'warn');
    if (txt) txt.textContent = parts.join(' · ') || 'No backends';
    if (sb)  sb.textContent  = parts.join(' · ') || '⚠ No backends';
    // Real mode
    const rb = document.getElementById('realBadge');
    if (rb) rb.style.display = d.real_mode_ready ? '' : 'none';
    const rt = document.getElementById('realModeTog');
    if (rt) rt.style.opacity = d.real_mode_ready ? '1' : '.4';
  } catch {
    const led = document.getElementById('sysLed');
    if (led) led.className = 'led err';
  }
}

// ── Progress ──────────────────────────────────────────────────
function onWsProg(pct, msg, step) {
  // Progress bars
  ['pbFill','pbFill2'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.width = `${pct}%`;
  });
  ['pbPct','pbPct2'].forEach(id => setEl(id, `${Math.round(pct)}%`));
  ['pbMsg','pbMsg2'].forEach(id => setEl(id, msg));

  // Log
  const log = document.getElementById('pbLog');
  if (log) {
    log.querySelectorAll('.cur').forEach(l => l.classList.remove('cur'));
    const l = document.createElement('div');
    l.className = `ll${pct >= 100 ? ' done' : ' cur'}`;
    l.textContent = `[${Math.round(pct)}%] ${msg}`;
    log.appendChild(l);
    log.scrollTop = log.scrollHeight;
  }

  // Flow mini step highlight
  if (step) {
    const stepMap= = {
      audio:'fm-audio', clean:'fm-clean', align:'fm-align',
      ppt:'fm-ppt', script:'fm-script', video:'fm-video',
      teacher_video:'fm-audio',
    };
    const fmId = stepMap[step];
    if (fmId) {
      document.querySelectorAll('.fm').forEach(f => f.classList.remove('running','done'));
      const el = document.getElementById(fmId);
      if (el) el.classList.add(pct >= 100 ? 'done' : 'running');
    }
    // Hero flow nodes (pro UI)
    const nodeMap= = {
      audio:'fn-audio', teacher_video:'fn-video', align:'fn-syllabus',
      ppt:'fn-slides', script:'fn-script', video:'fn-output',
    };
    const fnId = nodeMap[step];
    if (fnId) {
      const fn = document.getElementById(fnId);
      if (fn) {
        fn.className = 'fn';
        fn.classList.add(pct >= 100 ? 'done' : 'running');
      }
    }
  }
}

// ── Toast ──────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const tc = document.getElementById('toasts'); if (!tc) return;
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  const icons = {ok:'✅',err:'❌',info:'ℹ️'}; el.innerHTML = `<span>${icons[type]||'•'}</span><span>${msg}</span>`;
  tc.appendChild(el);
  setTimeout(() => el.remove(), 4800);
}

// ── DOM helpers ────────────────────────────────────────────────
function setEl(id, v) { const e = document.getElementById(id); if (e) e.textContent = v; }
function showEl(id) { const e = document.getElementById(id); if (e) e.style.display = ''; }
function hideEl(id) { const e = document.getElementById(id); if (e) e.style.display = 'none'; }
function val(id)    { const e = document.getElementById(id); return e ? e.value : ''; }
function ckd(id)    { const e = document.getElementById(id); return e ? e.checked : true; }
function getRadio(name) { return (document.querySelector(`input[name="${name}"]:checked`))?.value || ''; }

// ── Checklist & Generate button ───────────────────────────────
function updateChecklist() {
  const setCheck = (id: string, done: boolean, opt = false) => {
    const el = document.getElementById(id); if (!el) return;
    const icon = el.querySelector('.ci-icon');
    if (icon) icon.textContent = done ? '✓' : '○';
    el.classList.toggle('ci-done', done);
  };
  setCheck('ci-audio',    !!S.audioFile);
  setCheck('ci-teacher',  !!S.teacherFile);
  setCheck('ci-syllabus', val('syllabusTxt').length > 10, true);
  setCheck('ci-images',   S.imageFiles.length > 0, true);
  setCheck('ci-template', !!S.templateFile, true);

  const ready = S.audioFile && S.teacherFile;
  const btn = document.getElementById('btnGenerate');
  if (btn) btn.disabled = !ready;
  const hint = document.querySelector('.gen-hint') as HTMLElement;
  if (hint) hint.style.display = ready ? 'none' : '';
}

// ── Drop handler ──────────────────────────────────────────────
function handleDrop(event, type) {
  event.preventDefault();
  const files = [...(event.dataTransfer?.files || [])];
  if (!files.length) return;
  if      (type === 'audio')    handleAudioFile(files[0]);
  else if (type === 'teacher')  handleTeacherFile(files[0]);
  else if (type === 'img')      files.forEach(addImg);
  else if (type === 'template') handleTemplateFile(files[0]);
}

// ══════════════════════════════════════════════════════════════
// INPUT HANDLERS
// ══════════════════════════════════════════════════════════════

// ① Audio
function onAudioFile(inp) { if (inp.files?.[0]) handleAudioFile(inp.files[0]); }
function handleAudioFile(f) {
  const ok = ['.wav','.mp3','.m4a','.flac'];
  if (!ok.includes(f.name.slice(f.name.lastIndexOf('.')).toLowerCase())) {
    toast('Use WAV/MP3/M4A/FLAC', 'err'); return;
  }
  S.audioFile = f;
  const info = document.getElementById('audioInfo');
  if (info) { info.style.display = ''; info.textContent = `${f.name}  ·  ${(f.size/1e6).toFixed(1)} MB`; }
  const dz = document.getElementById('audioDz');
  if (dz) { dz.classList.add('filled'); const t = dz.querySelector('.uz-t,.dz span'); if (t) t.textContent = f.name; }
  toast(`Audio selected: ${f.name}`, 'info');
  updateChecklist();
}

// ② Teacher video
function onTeacherFile(inp) { if (inp.files?.[0]) handleTeacherFile(inp.files[0]); }
function handleTeacherFile(f) {
  S.teacherFile = f;
  const info = document.getElementById('teacherInfo');
  if (info) { info.style.display = ''; info.textContent = `${f.name}  ·  ${(f.size/1e9).toFixed(2)} GB`; }
  const dz = document.getElementById('teacherDz');
  if (dz) { dz.classList.add('filled'); const t = dz.querySelector('.uz-t,.dz span'); if (t) t.textContent = f.name; }
  toast(`Teacher video selected: ${f.name}`, 'info');
  updateChecklist();
}

// ③ Images
function addImg(f) {
  const ok = ['.jpg','.jpeg','.png','.webp'];
  if (!ok.includes(f.name.slice(f.name.lastIndexOf('.')).toLowerCase())) return;
  S.imageFiles.push(f); renderThumbs(); updateChecklist();
}
function onImgFiles(inp) { [...(inp.files||[])].forEach(addImg); }
function removeImg(i) { S.imageFiles.splice(i,1); renderThumbs(); updateChecklist(); }
function renderThumbs() {
  const g = document.getElementById('imgGrid'); if (!g) return;
  g.innerHTML = '';
  S.imageFiles.forEach((f, i) => {
    const url = URL.createObjectURL(f);
    const w = document.createElement('div'); w.className = 'tw';
    w.innerHTML = `<img src="${url}" alt="img${i}" title="${f.name}"/><button onclick="removeImg(${i})">✕</button>`;
    g.appendChild(w);
  });
  const dz = document.getElementById('imgDz');
  if (dz) {
    const t = dz.querySelector('.uz-t');
    if (t) t.textContent = S.imageFiles.length ? `${S.imageFiles.length} image(s) ready` : 'Drop teaching images here';
    if (S.imageFiles.length === 0) dz.classList.remove('filled');
    else dz.classList.add('filled');
  }
}

// ④ Template
function onTemplateModeChange() {
  const mode = getRadio('templateMode');
  const ts = document.getElementById('themeSection');
  const uts = document.getElementById('uploadTemplateSection');
  if (mode === 'upload') { ts && (ts.style.display = 'none'); uts && (uts.style.display = ''); }
  else { ts && (ts.style.display = ''); uts && (uts.style.display = 'none'); }
}
function onTemplateFile(inp) { if (inp.files?.[0]) handleTemplateFile(inp.files[0]); }
function handleTemplateFile(f) {
  if (!f.name.endsWith('.pptx')) { toast('Only .pptx files accepted', 'err'); return; }
  S.templateFile = f;
  const info = document.getElementById('templateInfo');
  if (info) { info.style.display = ''; info.textContent = `${f.name}  ·  ${(f.size/1e3).toFixed(0)} KB`; }
  const dz = document.getElementById('templateDz');
  if (dz) dz.classList.add('filled');
  toast(`Template: ${f.name}`, 'info');
  updateChecklist();
}

function selTheme(el) {
  document.querySelectorAll('.ts,.tp').forEach(s => s.classList.remove('act','active'));
  el.classList.add('act', 'active');
}

function loadTemplate() {
  const ta = document.getElementById('syllabusTxt'); if (!ta) return;
  ta.value = `Chapter Title\nChapter 1: Data Structures & Algorithms\n\nLearning Objectives\n1. Understand basic data structures and their operations\n2. Master sorting and searching algorithms\n3. Analyze time and space complexity\n\nKey Points\nArray and Linked List differences\nStack and Queue operations\nBinary Tree traversal\nSorting: Bubble Sort, Quick Sort, Merge Sort\nSearching: Linear Search, Binary Search\nComplexity: O(n), O(log n), O(n²)`;
  toast('Example syllabus loaded', 'info');
  updateChecklist();
}

// ══════════════════════════════════════════════════════════════
// ONE-CLICK FULL PIPELINE
// ══════════════════════════════════════════════════════════════
async function runFullPipeline() {
  if (!S.audioFile || !S.teacherFile) {
    toast('Upload lecture audio + teacher video to continue', 'err'); return;
  }

  // Show progress UI
  showEl('progressCard'); showEl('pnl-progress');
  hideEl('outputCard');   hideEl('pnl-output');
  document.getElementById('btnGenerate')?.setAttribute('disabled', '');
  document.getElementById('pnl-generate')?.querySelector('.bp-grad')?.classList.add('loading');

  // Upload template first (if provided)
  let templatePath = '';
  if (S.templateFile && getRadio('templateMode') === 'upload') {
    try {
      const tfd = new FormData();
      tfd.append('session_id', S.sid);
      tfd.append('template_file', S.templateFile);
      const tr = await fetch('/api/step/upload-template', { method: 'POST', body: tfd });
      const td = await tr.json();
      if (td.success) { templatePath = td.template_path; toast(`Template loaded: ${td.filename}`, 'info'); }
    } catch(e) { console.warn('Template upload:', e); }
  }

  // Build full pipeline form
  const fd = new FormData();
  fd.append('session_id',    S.sid);
  fd.append('audio_file',    S.audioFile);
  fd.append('teacher_video', S.teacherFile);
  fd.append('syllabus',      val('syllabusTxt') || '');
  fd.append('course_title',  val('courseTitle') || 'Lecture');
  fd.append('openai_key',    val('openaiKey') || '');
  fd.append('clean_method',  getRadio('llm') || 'ollama');
  fd.append('theme',         document.querySelector('.ts.act,.tp.active')?.getAttribute('data-t') || 'Modern Blue');
  fd.append('use_sd',        'false');
  fd.append('video_mode',    getRadio('vidMode') || 'demo');
  if (templatePath) fd.append('template_path', templatePath);
  S.imageFiles.forEach(f => fd.append('images', f));

  try {
    const r = await fetch('/api/pipeline/run', { method: 'POST', body: fd });
    const d = await r.json();

    if (!r.ok) throw new Error(d.detail || 'Pipeline failed');

    // Show output
    showEl('outputCard'); showEl('pnl-output');
    const vid = document.getElementById('outputVid');
    const stream_url = d.video_url || '/api/stream/完整moocs影片產出.mp4';
    if (vid) { vid.src = stream_url; vid.load(); }
    const dl = document.getElementById('videoDl');
    if (dl) { dl.href = stream_url; dl.download = 'moocs_video.mp4'; }

    // Extra downloads
    const extra = document.getElementById('extraDownloads');
    if (extra && d.output?.pptx_path) {
      const fname = d.output.pptx_path.split('/').pop() || 'slides.pptx';
      extra.innerHTML = `<div class="dl-row" style="margin-top:.5rem">
        <span>📊 Presentation</span>
        <a href="/api/download/${fname}" class="btn btn-dl cdl" download>⬇ Download PPTX</a>
      </div>`;
    }

    // Scroll to output
    setTimeout(() => {
      document.getElementById('outputCard')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      document.getElementById('pnl-output')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 400);

    if (d.errors?.length) toast(`Completed with notes: ${d.errors[0]}`, 'info');
    else toast('🎉 MOOCs video generated successfully!', 'ok');

  } catch(e) {
    toast(`Generation failed: ${e.message}`, 'err');
  } finally {
    document.getElementById('btnGenerate')?.removeAttribute('disabled');
    document.getElementById('pnl-generate')?.querySelector('.bp-grad')?.classList.remove('loading');
  }
}
