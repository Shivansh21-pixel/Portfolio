// Theme persistence
const htmlEl = document.documentElement;
const savedTheme = localStorage.getItem('sp-theme') || 'dark';
htmlEl.setAttribute('data-theme', savedTheme);

function toggleTheme(announce = false) {
  const next = htmlEl.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  htmlEl.setAttribute('data-theme', next);
  localStorage.setItem('sp-theme', next);
  drawOlemon();
  if (announce) showToast(`Theme: ${next === 'dark' ? 'Dark' : 'Light'}`);
}
document.getElementById('themeToggle').addEventListener('click', () => toggleTheme(false));

// Live IST Clock
function updateIST() {
  const now = new Date();
  const fmt = new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  }).format(now);
  const el = document.getElementById('istClock');
  if (el) el.textContent = `IST ${fmt}`;
}
updateIST();
setInterval(updateIST, 1000);

// 2px Scroll Reading Progress Bar + Active Nav Scroll Spy
const progressBar = document.getElementById('scrollProgress');
const trackedSections = ['lab', 'experience', 'projects', 'skills', 'certifications']
  .map(id => document.getElementById(id))
  .filter(Boolean);
const navItems = document.querySelectorAll('[data-section]');

function updateScrollUI() {
  const scrollTop = window.scrollY;
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  const pct = maxScroll > 0 ? Math.min(100, (scrollTop / maxScroll) * 100) : 0;
  progressBar.style.width = `${pct}%`;

  let currentId = '';
  const triggerLine = scrollTop + 180;
  trackedSections.forEach(sec => {
    if (sec.offsetTop <= triggerLine) currentId = sec.id;
  });
  navItems.forEach(link => {
    link.classList.toggle('active', link.dataset.section === currentId);
  });
}
window.addEventListener('scroll', updateScrollUI, { passive: true });
updateScrollUI();

// Toast helper
const toastEl = document.getElementById('toast');
let toastTimer;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 2200);
}

// Copy pills
document.querySelectorAll('.copy-pill').forEach(btn => {
  btn.addEventListener('click', () => {
    const val = btn.dataset.copy;
    navigator.clipboard.writeText(val).then(() => showToast(`Copied ${val}`));
  });
});

// Spotlight mouse tracking
document.querySelectorAll('.spotlight-card').forEach(card => {
  card.addEventListener('mousemove', e => {
    const rect = card.getBoundingClientRect();
    card.style.setProperty('--mx', `${e.clientX - rect.left}px`);
    card.style.setProperty('--my', `${e.clientY - rect.top}px`);
  });
});

// Scroll Reveal
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.classList.add('visible');
  });
}, { threshold: 0.08 });
document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// Lab Tabs
document.querySelectorAll('.lab-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.lab-tab').forEach(t => {
      t.classList.remove('active');
      t.setAttribute('aria-selected', 'false');
    });
    document.querySelectorAll('.lab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    tab.setAttribute('aria-selected', 'true');
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add('active');
    if (tab.dataset.tab === 'olemon') drawOlemon();
    if (tab.dataset.tab === 'nafnet') renderWaferCanvases();
  });
});

// HiDPI Canvas helper
function sizeHiDPI(canvas, cssWidth, cssHeight) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const w = Math.max(1, Math.floor(cssWidth));
  const h = Math.max(1, Math.floor(cssHeight));
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

// ==========================================
// LAB 1: OlemoN Vector Search (HiDPI 2D PCA)
// ==========================================
const olemonCanvas = document.getElementById('olemonCanvas');
let currentAlgo = 'HNSW';
let currentMetric = 'Cosine';
let currentTopK = 5;
let queryPt = { x: 0.52, y: 0.46 };

const docLabels = [
  'rag_01: HNSW graph layers',
  'rag_02: Llama 3.2 context',
  'rag_03: Cosine similarity',
  'rag_04: 768-d Ollama embed',
  'rag_05: KD-Tree split plane',
  'rag_06: REST /v1/query',
  'rag_07: PCA projection',
  'rag_08: Token chunking',
  'rag_09: Vector quantization',
  'rag_10: Recall@K benchmark',
  'rag_11: C++17 SIMD dot',
  'rag_12: Index persistence'
];

const vectors = Array.from({ length: 36 }, (_, i) => {
  const cluster = i % 3;
  const cx = [0.24, 0.74, 0.5][cluster];
  const cy = [0.3, 0.34, 0.76][cluster];
  const angle = i * 2.39996;
  const rad = 0.06 + ((i * 19) % 100) / 700;
  return {
    id: i,
    x: Math.min(0.94, Math.max(0.06, cx + Math.cos(angle) * rad)),
    y: Math.min(0.9, Math.max(0.08, cy + Math.sin(angle) * rad)),
    label: docLabels[i % docLabels.length]
  };
});

function calcScore(a, b, metric) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dist = Math.hypot(dx, dy);
  if (metric === 'Euclidean') return { sort: dist, display: `d=${dist.toFixed(3)}` };
  if (metric === 'Dot') {
    const dot = a.x * b.x + a.y * b.y;
    return { sort: -dot, display: `dot=${dot.toFixed(3)}` };
  }
  const sim = Math.max(0.01, 1 - dist * 1.15);
  return { sort: -sim, display: `cos=${sim.toFixed(3)}` };
}

function drawOlemon() {
  const rect = olemonCanvas.getBoundingClientRect();
  if (!rect.width) return;
  const { ctx: oCtx, w, h } = sizeHiDPI(olemonCanvas, rect.width, rect.height || 330);
  const isDark = htmlEl.getAttribute('data-theme') === 'dark';

  oCtx.clearRect(0, 0, w, h);

  // Subtle coordinate grid
  oCtx.strokeStyle = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)';
  oCtx.lineWidth = 1;
  for (let gx = 40; gx < w; gx += 40) {
    oCtx.beginPath(); oCtx.moveTo(gx, 0); oCtx.lineTo(gx, h); oCtx.stroke();
  }
  for (let gy = 40; gy < h; gy += 40) {
    oCtx.beginPath(); oCtx.moveTo(0, gy); oCtx.lineTo(w, gy); oCtx.stroke();
  }

  const ranked = vectors
    .map(v => ({ ...v, ...calcScore(queryPt, v, currentMetric) }))
    .sort((a, b) => a.sort - b.sort);
  const topK = ranked.slice(0, currentTopK);
  const topIds = new Set(topK.map(v => v.id));

  // Index structure overlay
  if (currentAlgo === 'HNSW') {
    oCtx.strokeStyle = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    for (let i = 0; i < vectors.length; i += 2) {
      const a = vectors[i];
      const b = vectors[(i + 3) % vectors.length];
      oCtx.beginPath();
      oCtx.moveTo(a.x * w, a.y * h);
      oCtx.lineTo(b.x * w, b.y * h);
      oCtx.stroke();
    }
  } else if (currentAlgo === 'KD-Tree') {
    oCtx.strokeStyle = isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)';
    oCtx.setLineDash([4, 4]);
    [0.5 * w].forEach(x => { oCtx.beginPath(); oCtx.moveTo(x, 0); oCtx.lineTo(x, h); oCtx.stroke(); });
    [0.52 * h].forEach(y => { oCtx.beginPath(); oCtx.moveTo(0, y); oCtx.lineTo(w, y); oCtx.stroke(); });
    oCtx.setLineDash([]);
  }

  // Lines to Top-K neighbors
  topK.forEach((pt, idx) => {
    const alpha = Math.max(0.22, 0.85 - idx * 0.08);
    oCtx.strokeStyle = isDark ? `rgba(255,255,255,${alpha})` : `rgba(10,10,10,${alpha})`;
    oCtx.lineWidth = idx === 0 ? 1.8 : 1.1;
    oCtx.beginPath();
    oCtx.moveTo(queryPt.x * w, queryPt.y * h);
    oCtx.lineTo(pt.x * w, pt.y * h);
    oCtx.stroke();
  });

  // Vector nodes
  vectors.forEach(v => {
    const isHit = topIds.has(v.id);
    oCtx.beginPath();
    oCtx.arc(v.x * w, v.y * h, isHit ? 5.2 : 3.4, 0, Math.PI * 2);
    oCtx.fillStyle = isHit
      ? (isDark ? '#ffffff' : '#09090b')
      : (isDark ? 'rgba(255,255,255,0.28)' : 'rgba(0,0,0,0.24)');
    oCtx.fill();
  });

  // Query vector glow + core
  oCtx.beginPath();
  oCtx.arc(queryPt.x * w, queryPt.y * h, 13, 0, Math.PI * 2);
  oCtx.fillStyle = 'rgba(74, 222, 128, 0.18)';
  oCtx.fill();

  oCtx.beginPath();
  oCtx.arc(queryPt.x * w, queryPt.y * h, 6.5, 0, Math.PI * 2);
  oCtx.fillStyle = '#4ade80';
  oCtx.fill();
  oCtx.strokeStyle = isDark ? '#050505' : '#ffffff';
  oCtx.lineWidth = 2;
  oCtx.stroke();

  // Stats update
  const baseLat = (currentAlgo === 'HNSW' ? 0.36 : currentAlgo === 'KD-Tree' ? 1.05 : 4.65) + currentTopK * 0.02;
  document.getElementById('olemonAlgoStat').textContent = `${currentAlgo} · ${currentMetric} · K=${currentTopK}`;
  document.getElementById('olemonLatency').textContent = `${baseLat.toFixed(2)} ms`;
  document.getElementById('olemonTopKHeader').textContent = `TOP-${currentTopK} NEAREST CHUNKS`;
  document.getElementById('olemonNeighbors').innerHTML = topK
    .map(n => `<li><span>${n.label}</span><span>${n.display}</span></li>`)
    .join('');
}

function setQueryFromEvent(e) {
  const rect = olemonCanvas.getBoundingClientRect();
  queryPt = {
    x: Math.min(0.95, Math.max(0.05, (e.clientX - rect.left) / rect.width)),
    y: Math.min(0.92, Math.max(0.08, (e.clientY - rect.top) / rect.height))
  };
  document.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
  drawOlemon();
}
olemonCanvas.addEventListener('pointerdown', setQueryFromEvent);
olemonCanvas.addEventListener('pointermove', e => { if (e.buttons === 1) setQueryFromEvent(e); });

document.querySelectorAll('#algoSwitch .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#algoSwitch .seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentAlgo = btn.dataset.algo;
    drawOlemon();
  });
});
document.querySelectorAll('#metricSwitch .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#metricSwitch .seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMetric = btn.dataset.metric;
    drawOlemon();
  });
});
document.querySelectorAll('#topKSwitch .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#topKSwitch .seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTopK = Number(btn.dataset.k);
    drawOlemon();
  });
});
document.getElementById('randomizeQuery').addEventListener('click', () => {
  queryPt = {
    x: 0.12 + Math.random() * 0.76,
    y: 0.14 + Math.random() * 0.72
  };
  document.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
  drawOlemon();
});
document.querySelectorAll('.chip-btn').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    queryPt = { x: parseFloat(chip.dataset.qx), y: parseFloat(chip.dataset.qy) };
    drawOlemon();
  });
});

// ==========================================
// LAB 2: NAFNet-Lite Wafer Restoration (HiDPI)
// ==========================================
const compareStage = document.getElementById('compareStage');
const compareSlider = document.getElementById('compareSlider');
const compareOverlay = document.getElementById('compareOverlay');
const compareHandle = document.getElementById('compareHandle');

function renderWaferCanvases() {
  const stageRect = compareStage.getBoundingClientRect();
  if (!stageRect.width) return;
  const cw = stageRect.width;
  const ch = stageRect.height || 330;

  const cRestored = document.getElementById('waferRestored');
  const cDegraded = document.getElementById('waferDegraded');
  cDegraded.style.width = `${cw}px`;

  const { ctx: ctxR, w, h } = sizeHiDPI(cRestored, cw, ch);
  const { ctx: ctxD } = sizeHiDPI(cDegraded, cw, ch);

  // Clean 2x semiconductor die pattern
  ctxR.fillStyle = '#0a0a0c';
  ctxR.fillRect(0, 0, w, h);

  ctxR.strokeStyle = 'rgba(255,255,255,0.14)';
  ctxR.lineWidth = 1;
  for (let x = 0; x < w; x += 48) {
    ctxR.beginPath(); ctxR.moveTo(x, 0); ctxR.lineTo(x, h); ctxR.stroke();
  }
  for (let y = 0; y < h; y += 48) {
    ctxR.beginPath(); ctxR.moveTo(0, y); ctxR.lineTo(w, y); ctxR.stroke();
  }

  ctxR.strokeStyle = 'rgba(245,245,245,0.75)';
  ctxR.lineWidth = 2;
  const rows = Math.floor((h - 40) / 22);
  for (let i = 0; i < rows; i++) {
    const y = 26 + i * 22;
    const kink = w * 0.24 + (i % 4) * 44;
    ctxR.beginPath();
    ctxR.moveTo(20, y);
    ctxR.lineTo(kink, y);
    ctxR.lineTo(kink + 36, y + 10);
    ctxR.lineTo(w - 28, y + 10);
    ctxR.stroke();

    ctxR.fillStyle = '#ffffff';
    ctxR.fillRect(kink - 3, y - 3, 6, 6);
    ctxR.fillRect(w - 34, y + 7, 6, 6);
  }

  const rx = w * 0.4;
  const ry = h * 0.28;
  ctxR.strokeStyle = '#4ade80';
  ctxR.lineWidth = 1.5;
  ctxR.strokeRect(rx, ry, 180, 134);
  ctxR.fillStyle = '#4ade80';
  ctxR.font = '12px JetBrains Mono, monospace';
  ctxR.fillText('DIE_04B · CLEAN 2x', rx + 10, ry + 20);

  // Degraded input: blur + sensor noise
  ctxD.fillStyle = '#0a0a0c';
  ctxD.fillRect(0, 0, w, h);
  ctxD.save();
  ctxD.filter = 'blur(2.5px) contrast(0.82)';
  ctxD.drawImage(cRestored, 0, 0, w, h);
  ctxD.restore();

  const rawW = cDegraded.width;
  const rawH = cDegraded.height;
  const imgData = ctxD.getImageData(0, 0, rawW, rawH);
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    const noise = (Math.random() - 0.5) * 90;
    d[i] = Math.min(255, Math.max(0, d[i] + noise));
    d[i + 1] = Math.min(255, Math.max(0, d[i + 1] + noise));
    d[i + 2] = Math.min(255, Math.max(0, d[i + 2] + noise));
  }
  ctxD.putImageData(imgData, 0, 0);
}

function setSplit(pct) {
  const clamped = Math.min(95, Math.max(5, pct));
  compareOverlay.style.width = `${clamped}%`;
  compareHandle.style.left = `${clamped}%`;
  compareSlider.value = clamped;
}
compareSlider.addEventListener('input', e => setSplit(e.target.value));
compareStage.addEventListener('pointermove', e => {
  if (e.buttons === 1) {
    const rect = compareStage.getBoundingClientRect();
    setSplit(((e.clientX - rect.left) / rect.width) * 100);
  }
});
compareStage.addEventListener('pointerdown', e => {
  const rect = compareStage.getBoundingClientRect();
  setSplit(((e.clientX - rect.left) / rect.width) * 100);
});

// Handle resize cleanly for HiDPI canvases
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    drawOlemon();
    renderWaferCanvases();
  }, 120);
});
drawOlemon();

// ==========================================
// LAB 3: WireSpy DPI Flow Benchmark (77 pkts)
// ==========================================
const sampleHosts = [
  { proto: 'TLS/SNI', host: 'api.github.com', action: 'FORWARD' },
  { proto: 'TLS/SNI', host: 'olemo-n-rag.vercel.app', action: 'FORWARD' },
  { proto: 'HTTP', host: 'telemetry.blocked.local', action: 'DROP' },
  { proto: 'TLS/SNI', host: 'leetcode.com', action: 'FORWARD' },
  { proto: 'UDP/QUIC', host: 'dns.google', action: 'FORWARD' },
  { proto: 'TCP', host: '10.0.4.22:5432', action: 'FORWARD' },
  { proto: 'HTTP', host: 'ad-tracker.invalid', action: 'DROP' },
  { proto: 'TLS/SNI', host: 'codeforces.com', action: 'FORWARD' }
];

const packets = Array.from({ length: 77 }, (_, idx) => {
  const isDrop = [4, 13, 22, 35, 44, 53, 64, 71].includes(idx);
  const pool = sampleHosts.filter(h => (isDrop ? h.action === 'DROP' : h.action === 'FORWARD'));
  const pick = pool[idx % pool.length];
  const hash = '0x' + ((0x9e3779b1 ^ (idx * 2654435761)) >>> 0).toString(16).padStart(8, '0');
  return {
    id: idx + 1,
    hash,
    proto: pick.proto,
    host: pick.host,
    bytes: 64 + ((idx * 137) % 1380),
    action: isDrop ? 'DROP' : 'FORWARD'
  };
});

let currentDpiFilter = 'ALL';
let currentDpiSearch = '';
let dpiInterval = null;

function getFilteredPackets() {
  const q = currentDpiSearch.toLowerCase().trim();
  return packets.filter(p => {
    const matchStatus = currentDpiFilter === 'ALL' || p.action === currentDpiFilter;
    const matchText = !q ||
      p.host.toLowerCase().includes(q) ||
      p.proto.toLowerCase().includes(q) ||
      p.hash.toLowerCase().includes(q) ||
      p.action.toLowerCase().includes(q);
    return matchStatus && matchText;
  });
}

function renderDpi(animated = false) {
  clearInterval(dpiInterval);
  const tbody = document.getElementById('dpiBody');
  const filtered = getFilteredPackets();
  document.getElementById('dpiMatchCount').textContent = `Showing ${filtered.length} / 77`;

  const rowHTML = p => `
    <tr>
      <td>#${String(p.id).padStart(2, '0')}</td>
      <td>${p.hash}</td>
      <td>${p.proto}</td>
      <td>${p.host}</td>
      <td>${p.bytes} B</td>
      <td class="${p.action === 'FORWARD' ? 'verdict-fwd' : 'verdict-drop'}">${p.action}</td>
    </tr>
  `;

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="muted" style="text-align:center;padding:24px;">No packets match "${currentDpiSearch}"</td></tr>`;
    return;
  }

  if (!animated) {
    tbody.innerHTML = filtered.map(rowHTML).join('');
    return;
  }

  tbody.innerHTML = '';
  let i = 0;
  dpiInterval = setInterval(() => {
    const slice = filtered.slice(i, i + 6);
    tbody.insertAdjacentHTML('beforeend', slice.map(rowHTML).join(''));
    i += 6;
    if (i >= filtered.length) clearInterval(dpiInterval);
  }, 32);
}
renderDpi(false);

document.querySelectorAll('#dpiFilter .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#dpiFilter .seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentDpiFilter = btn.dataset.filter;
    renderDpi(false);
  });
});
document.getElementById('dpiSearch').addEventListener('input', e => {
  currentDpiSearch = e.target.value;
  renderDpi(false);
});
document.getElementById('replayDpi').addEventListener('click', () => renderDpi(true));

// ==========================================
// Command Palette (Cmd+K / Ctrl+K / '/')
// ==========================================
const cmdModal = document.getElementById('cmdModal');
const cmdInput = document.getElementById('cmdInput');
const cmdList = document.getElementById('cmdList');
let selectedIdx = 0;

const commands = [
  { label: 'Go to Interactive Lab', group: 'NAV', run: () => location.hash = '#lab' },
  { label: 'Go to Experience (Landis+Gyr)', group: 'NAV', run: () => location.hash = '#experience' },
  { label: 'Go to Selected Projects', group: 'NAV', run: () => location.hash = '#projects' },
  { label: 'Go to Technical Stack', group: 'NAV', run: () => location.hash = '#skills' },
  { label: 'Go to Certifications', group: 'NAV', run: () => location.hash = '#certifications' },
  { label: 'Download Resume PDF (1-Click)', group: 'RESUME', run: () => { const a = document.createElement('a'); a.href = 'Shivansh_Pandey_Resume.pdf'; a.download = 'Shivansh_Pandey_Resume.pdf'; a.click(); showToast('Downloading Resume PDF'); } },
  { label: 'Open Quick Resume Preview', group: 'RESUME', run: () => openResume() },
  { label: 'Toggle Dark / Light Theme (T)', group: 'ACTION', run: () => toggleTheme(true) },
  { label: 'Copy Email (shivanshpandey210501@gmail.com)', group: 'COPY', run: () => navigator.clipboard.writeText('shivanshpandey210501@gmail.com').then(() => showToast('Copied Email')) },
  { label: 'Copy Phone (+91-7651859212)', group: 'COPY', run: () => navigator.clipboard.writeText('+91-7651859212').then(() => showToast('Copied Phone')) },
  { label: 'Open GitHub Profile', group: 'LINK', run: () => window.open('https://github.com/Shivansh21-pixel', '_blank') },
  { label: 'Open LinkedIn Profile', group: 'LINK', run: () => window.open('https://www.linkedin.com/in/shivansh-pandey-8330b5279/', '_blank') },
  { label: 'Open LeetCode Profile', group: 'LINK', run: () => window.open('https://leetcode.com/u/Shivansh2145/', '_blank') },
  { label: 'Open Codeforces Profile', group: 'LINK', run: () => window.open('https://codeforces.com/profile/Shiv2145', '_blank') },
  { label: 'Open OlemoN Vector DB Live App', group: 'PROJECT', run: () => window.open('https://olemo-n-ai-vector-database-local-ra.vercel.app/', '_blank') },
  { label: 'Open NAFNet-Lite GitHub Repo', group: 'PROJECT', run: () => window.open('https://github.com/Shivansh21-pixel/KLA-image-restoration', '_blank') },
  { label: 'Open WireSpy DPI Live Demo', group: 'PROJECT', run: () => window.open('https://shivansh21-pixel.github.io/WireSpy/', '_blank') }
];

function renderCommands(filter = '') {
  const q = filter.toLowerCase().trim();
  const matches = commands.filter(c => c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q));
  selectedIdx = 0;
  cmdList.innerHTML = matches.map((c, idx) => `
    <li class="cmd-item ${idx === 0 ? 'selected' : ''}" data-idx="${idx}">
      <span>${c.label}</span>
      <span class="cmd-group">${c.group}</span>
    </li>
  `).join('');
  cmdList.querySelectorAll('.cmd-item').forEach((li, idx) => {
    li.addEventListener('click', () => {
      matches[idx].run();
      closeCmd();
    });
  });
  cmdList._matches = matches;
}

function openCmd() {
  cmdModal.classList.add('open');
  cmdModal.setAttribute('aria-hidden', 'false');
  cmdInput.value = '';
  renderCommands();
  setTimeout(() => cmdInput.focus(), 20);
}
function closeCmd() {
  cmdModal.classList.remove('open');
  cmdModal.setAttribute('aria-hidden', 'true');
}

document.getElementById('openCmd').addEventListener('click', openCmd);
cmdModal.addEventListener('click', e => { if (e.target === cmdModal) closeCmd(); });
cmdInput.addEventListener('input', e => renderCommands(e.target.value));

// Resume Modal
const resumeModal = document.getElementById('resumeModal');
function openResume() {
  resumeModal.classList.add('open');
  resumeModal.setAttribute('aria-hidden', 'false');
}
function closeResume() {
  resumeModal.classList.remove('open');
  resumeModal.setAttribute('aria-hidden', 'true');
}
document.getElementById('openResumeModal').addEventListener('click', openResume);
document.getElementById('closeResumeModal').addEventListener('click', closeResume);
document.getElementById('printResume').addEventListener('click', () => window.print());
resumeModal.addEventListener('click', e => { if (e.target === resumeModal) closeResume(); });

// Global Keyboard Shortcuts (⌘K, /, T, Esc)
function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
}

window.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    cmdModal.classList.contains('open') ? closeCmd() : openCmd();
    return;
  }
  if (e.key === 'Escape') {
    closeCmd();
    closeResume();
    return;
  }
  if (cmdModal.classList.contains('open')) {
    const items = cmdList.querySelectorAll('.cmd-item');
    if (!items.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      items[selectedIdx].classList.remove('selected');
      selectedIdx = (selectedIdx + 1) % items.length;
      items[selectedIdx].classList.add('selected');
      items[selectedIdx].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      items[selectedIdx].classList.remove('selected');
      selectedIdx = (selectedIdx - 1 + items.length) % items.length;
      items[selectedIdx].classList.add('selected');
      items[selectedIdx].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter') {
      e.preventDefault();
      cmdList._matches[selectedIdx]?.run();
      closeCmd();
    }
    return;
  }

  // Single-key shortcuts when not typing in inputs
  if (!isTypingTarget(document.activeElement) && !e.metaKey && !e.ctrlKey && !e.altKey) {
    if (e.key.toLowerCase() === 't') {
      e.preventDefault();
      toggleTheme(true);
    } else if (e.key === '/') {
      e.preventDefault();
      openCmd();
    }
  }
});
