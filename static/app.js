/**
 * app.js — Agent7 Web UI Frontend
 * Handles: chat send/receive, SSE streaming, tab switching,
 *          file list, facts panel, memory panel.
 */

// ── DOM refs ──────────────────────────────────────────────────────────────
const chatMessages   = document.getElementById('chat-messages');
const chatInput      = document.getElementById('chat-input');
const sendBtn        = document.getElementById('send-btn');
const typingIndicator= document.getElementById('typing-indicator');
const outputLines    = document.getElementById('output-lines');
const filesList      = document.getElementById('files-list');
const factsList      = document.getElementById('facts-list');
const factsSearch    = document.getElementById('facts-search');
const memoryList     = document.getElementById('memory-list');
const statusDot      = document.getElementById('status-dot');
const statusText     = document.getElementById('status-text');
const modelLabel     = document.getElementById('model-label');
const factsCount     = document.getElementById('facts-count');
const sessionsCount  = document.getElementById('sessions-count');
const filesBadge     = document.getElementById('files-badge');

// ── State ─────────────────────────────────────────────────────────────────
let busy         = false;
let eventSource  = null;
let allFacts     = [];
let fileCount    = 0;

// ── Init ──────────────────────────────────────────────────────────────────
(async function init() {
  await loadStatus();
  await loadFacts();
  await loadMemory();
  await loadFiles();
})();

// ── Status polling ────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    modelLabel.textContent    = d.model || '—';
    factsCount.textContent    = d.facts  || 0;
    sessionsCount.textContent = d.sessions || 0;
  } catch(e) { /* silent */ }
}

// ── Send message ──────────────────────────────────────────────────────────
async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || busy) return;

  chatInput.value = '';
  autoResizeInput();

  // Add user bubble
  appendMessage('user', text);

  // Clear output panel for new task
  outputLines.innerHTML = '';

  setBusy(true);

  // Open SSE stream FIRST so we never miss the 'start' event
  openStream();

  // Then POST to /api/chat
  let resp;
  try {
    resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
  } catch(e) {
    appendMessage('agent', '⚠ Could not reach the server. Is web_agent.py running?');
    setBusy(false);
    return;
  }

  const data = await resp.json();
  if (data.error) {
    if (eventSource) eventSource.close();
    appendMessage('agent', '⚠ ' + data.error);
    setBusy(false);
    return;
  }
}

// ── SSE stream ────────────────────────────────────────────────────────────
function openStream() {
  if (eventSource) eventSource.close();

  eventSource = new EventSource('/api/stream');

  eventSource.onmessage = (e) => {
    let item;
    try { item = JSON.parse(e.data); }
    catch { return; }

    switch(item.type) {

      case 'start':
        switchTab('output');
        break;

      case 'session_dir':
        addFolderHeading(item.folder);
        break;

      case 'thinking':
        appendThinkingBlock(item.text);
        break;

      case 'output':
        appendOutputLine(item.text);
        break;

      case 'bash':
        appendOutputLine(item.text, 'bash-cmd');
        break;

      case 'facts_match':
        appendFactsBanner(item.text);
        break;

      case 'file':
        addFileEntry(item.file);
        break;

      case 'done':
        eventSource.close();
        appendMessage('agent', item.text);
        setBusy(false);
        loadStatus();
        loadFacts();
        loadFiles();
        break;

      case 'error':
        eventSource.close();
        appendMessage('agent', '⚠ Error: ' + item.text);
        setBusy(false);
        break;
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    setBusy(false);
  };
}

// ── Chat bubbles ──────────────────────────────────────────────────────────
function appendMessage(role, text) {
  const div  = document.createElement('div');
  div.className = `msg msg-${role}`;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'You' : 'Agent7';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  div.appendChild(label);
  div.appendChild(bubble);
  chatMessages.appendChild(div);
  scrollChat();
}

function appendFactsBanner(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-agent';

  const banner = document.createElement('div');
  banner.className = 'facts-banner';
  banner.innerHTML = '<div class="facts-banner-label">📚 Relevant facts from knowledge base</div>' 
                   + escapeHtml(text);

  div.appendChild(banner);
  chatMessages.appendChild(div);
  scrollChat();
}

function appendThinkingBlock(text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg msg-agent';

  const block = document.createElement('div');
  block.className = 'thinking-block';

  const header = document.createElement('div');
  header.className = 'thinking-header';
  header.innerHTML = '💭 <span style="flex:1">Thinking</span><span class="thinking-chevron">▼</span>';
  header.onclick = () => block.classList.toggle('collapsed');

  const body = document.createElement('div');
  body.className = 'thinking-body';
  body.textContent = text;

  block.appendChild(header);
  block.appendChild(body);
  wrapper.appendChild(block);
  chatMessages.appendChild(wrapper);
  scrollChat();
}

// ── Output panel lines ────────────────────────────────────────────────────
function appendOutputLine(text, cls = '') {
  // Remove ANSI escape codes
  const clean = text.replace(/\x1b\[[0-9;]*m/g, '').trim();
  if (!clean) return;

  // Detect step headers from smolagents output
  const extraCls = /^(Step \d+|━+|\[Step)/.test(clean) ? 'step-header' : cls;

  const line = document.createElement('div');
  line.className = 'output-line ' + extraCls;
  line.textContent = clean;
  outputLines.appendChild(line);
  outputLines.scrollTop = outputLines.scrollHeight;
}

// ── Files panel ───────────────────────────────────────────────────────────
async function loadFiles() {
  try {
    const r = await fetch('/api/files');
    const d = await r.json();
    renderFiles(d.files || []);
  } catch(e) {}
}

function addFolderHeading(folderName) {
  const empty = filesList.querySelector('.empty-state');
  if (empty) empty.remove();

  const heading = document.createElement('div');
  heading.className = 'folder-heading';
  heading.innerHTML = `<span>📂</span> ${escapeHtml(folderName)}`;
  filesList.appendChild(heading);
  switchTab('files');
}

function addFileEntry(file) {
  // Remove empty state if present
  const empty = filesList.querySelector('.empty-state');
  if (empty) empty.remove();

  filesList.appendChild(buildFileEl(file));
  fileCount++;
  filesBadge.textContent = fileCount;
  filesBadge.style.display = 'inline';
}

function renderFiles(files) {
  if (!files.length) return;
  filesList.innerHTML = '';
  files.forEach(f => filesList.appendChild(buildFileEl(f)));
  fileCount = files.length;
  filesBadge.textContent = fileCount;
  filesBadge.style.display = 'inline';
}

function buildFileEl(f) {
  const a = document.createElement('a');
  a.className = 'file-item';
  a.href = '/downloads/' + (f.folder ? encodeURIComponent(f.folder) + '/' : '') + encodeURIComponent(f.name);
  a.download = f.name;
  a.target = '_blank';

  const icon = fileIcon(f.name);

  a.innerHTML = `
    <span class="file-icon">${icon}</span>
    <div class="file-info">
      <div class="file-name">${escapeHtml(f.name)}</div>
      <div class="file-meta">${f.size} · ${f.time}</div>
    </div>
    <span class="file-dl">⬇</span>
  `;
  return a;
}

function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  const icons = {
    py:'🐍', js:'📜', ts:'📜', html:'🌐', css:'🎨',
    json:'📋', md:'📝', txt:'📄', csv:'📊',
    xlsx:'📊', xls:'📊', docx:'📃', doc:'📃',
    pdf:'📕', png:'🖼', jpg:'🖼', jpeg:'🖼', gif:'🖼',
    mp3:'🎵', wav:'🎵', mid:'🎵', midi:'🎵',
    zip:'📦', tar:'📦', gz:'📦',
    sh:'⚙', bat:'⚙',
  };
  return icons[ext] || '📄';
}

// ── Facts panel ───────────────────────────────────────────────────────────
async function loadFacts() {
  try {
    const r = await fetch('/api/facts');
    const d = await r.json();
    allFacts = d.facts || [];
    factsCount.textContent = allFacts.length;
    renderFacts(allFacts);
  } catch(e) {}
}

function renderFacts(facts) {
  if (!facts.length) {
    factsList.innerHTML = `<div class="empty-state">
      <span class="empty-icon">🧠</span>
      No facts yet. Try: <em>explore: photosynthesis</em>
    </div>`;
    return;
  }

  factsList.innerHTML = '';
  // Show newest first
  [...facts].reverse().forEach(f => {
    const div = document.createElement('div');
    div.className = 'fact-item';
    div.innerHTML = `
      <div class="fact-id">Fact #${f.id} · ${f.source || ''}</div>
      <div class="fact-topic">${escapeHtml(f.topic)}</div>
      <div class="fact-body">${escapeHtml(f.fact)}</div>
      <div class="fact-date">${f.date || ''}</div>
    `;
    factsList.appendChild(div);
  });
}

factsSearch.addEventListener('input', () => {
  const q = factsSearch.value.toLowerCase();
  if (!q) { renderFacts(allFacts); return; }
  const filtered = allFacts.filter(f =>
    f.topic.toLowerCase().includes(q) || f.fact.toLowerCase().includes(q)
  );
  renderFacts(filtered);
});

// ── Memory panel ──────────────────────────────────────────────────────────
async function loadMemory() {
  try {
    const r = await fetch('/api/memory');
    const d = await r.json();
    renderMemory(d.sessions || []);
  } catch(e) {}
}

function renderMemory(sessions) {
  if (!sessions.length) return;

  memoryList.innerHTML = '';
  [...sessions].reverse().forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'session-item';

    const prompts = (s.prompts || []).slice(0, 5);
    const promptsHtml = prompts.map(p =>
      `<div class="session-prompt-item">${escapeHtml(p.slice(0, 100))}${p.length > 100 ? '…' : ''}</div>`
    ).join('');

    div.innerHTML = `
      <div class="session-date">Session · ${s.date || 'unknown'}</div>
      <div class="session-summary">${escapeHtml(s.summary || '')}</div>
      ${prompts.length ? `<div class="session-prompts">${promptsHtml}</div>` : ''}
    `;
    memoryList.appendChild(div);
  });
}

// ── Tab switching ─────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === name)
  );
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('active', p.id === 'panel-' + name)
  );
}

// ── Busy state ────────────────────────────────────────────────────────────
function setBusy(state) {
  busy = state;
  sendBtn.disabled   = state;
  chatInput.disabled = state;
  typingIndicator.classList.toggle('visible', state);
  statusDot.classList.toggle('busy', state);
  statusText.textContent = state ? 'Working...' : 'Ready';
  if (!state) scrollChat();
}

// ── Input auto-resize ─────────────────────────────────────────────────────
chatInput.addEventListener('input', autoResizeInput);

function autoResizeInput() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

// ── Keyboard handling ─────────────────────────────────────────────────────
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

// ── Helpers ───────────────────────────────────────────────────────────────
function scrollChat() {
  requestAnimationFrame(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
