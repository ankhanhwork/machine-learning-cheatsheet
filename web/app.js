const state = { documents: [], folder: 'all', query: '', type: 'all' };
const $ = (id) => document.getElementById(id);
const encodePath = (path) => path.split('/').map(encodeURIComponent).join('/');
const pretty = (name) => name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
const formatSize = (bytes) => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
const icon = (type) => ({ ipynb: '◉', md: '▤', pdf: '▧', docx: '▧', py: '‹›', csv: '▦' }[type] || '▤');
const folderName = (doc) => doc.path.includes('/') ? doc.path.split('/')[0] : 'Tài liệu khác';
const CHAT_WEBHOOK = 'https://frannie133625.app.n8n.cloud/webhook/9fb21e8a-d1a8-4608-8343-d44f1447b3a5/chat';
const CHAT_HISTORY_KEY = 'study-ml-chat-history-v1';
const CHAT_SESSION_KEY = 'study-ml-chat-session-v1';
let chatMessages = loadChatHistory();
let chatPending = false;

function loadChatHistory() {
  try { const value = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || '[]'); return Array.isArray(value) ? value.filter((m) => m && ['user', 'assistant'].includes(m.role) && typeof m.text === 'string') : []; }
  catch { return []; }
}
function saveChatHistory() { try { localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatMessages)); } catch { showChatError('Không thể lưu lịch sử trên thiết bị này.'); } }
function getChatSession() {
  let id = localStorage.getItem(CHAT_SESSION_KEY);
  if (!id) { id = globalThis.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(36).slice(2)}`; localStorage.setItem(CHAT_SESSION_KEY, id); }
  return id;
}
function drawChat() {
  const container = $('chatMessages');
  if (!container) return;
  if (!chatMessages.length) container.innerHTML = '<div class="chat-welcome"><span>✳</span><h2>Bắt đầu cuộc trò chuyện</h2><p>Bạn đang học chủ đề nào? Hãy gửi câu hỏi để bắt đầu nhé.</p></div>';
  else { container.innerHTML = ''; for (const message of chatMessages) { const bubble = document.createElement('div'); bubble.className = `chat-message ${message.role}`; bubble.textContent = message.text; container.appendChild(bubble); } }
  container.scrollTop = container.scrollHeight;
}
function showChatError(text) {
  let error = $('chatError');
  if (!error) { error = document.createElement('p'); error.id = 'chatError'; error.className = 'chat-error'; $('chatForm').before(error); }
  error.textContent = text;
}
function responseText(data) {
  if (typeof data === 'string') return data;
  if (Array.isArray(data)) return data.map(responseText).filter(Boolean).join('\n');
  if (data && typeof data === 'object') for (const key of ['output', 'text', 'response', 'message', 'answer']) {
    if (typeof data[key] === 'string') return data[key];
    if (data[key] && typeof data[key] === 'object') { const nested = responseText(data[key]); if (nested) return nested; }
  }
  return '';
}
function openChatPage(fromHistory = false) {
  $('welcome').hidden = true; document.querySelector('.content-section').hidden = true; document.querySelector('footer').hidden = true;
  $('chatPage').hidden = false; $('crumbCurrent').textContent = 'Trợ lý học tập';
  if (!fromHistory) history.pushState({ chat: true }, '', `${location.pathname}?view=chat`);
  drawChat(); if (!fromHistory) $('chatInput').focus();
}
function closeChatPage(updateHistory = true) {
  $('chatPage').hidden = true; document.querySelector('.content-section').hidden = false; document.querySelector('footer').hidden = false;
  $('welcome').hidden = state.folder !== 'all'; $('crumbCurrent').textContent = state.folder === 'all' ? 'Tất cả tài liệu' : state.folder;
  if (updateHistory && new URLSearchParams(location.search).has('view')) history.replaceState({}, '', location.pathname);
}
async function sendChatMessage(text) {
  if (chatPending) return;
  chatPending = true; $('sendChat').disabled = true; $('chatInput').disabled = true;
  const error = $('chatError'); if (error) error.remove();
  chatMessages.push({ role: 'user', text }); saveChatHistory(); drawChat();
  const pending = document.createElement('div'); pending.className = 'chat-message assistant chat-typing'; pending.textContent = 'Đang soạn câu trả lời…'; $('chatMessages').appendChild(pending); $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
  try {
    const response = await fetch(CHAT_WEBHOOK, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json, text/plain' }, body: JSON.stringify({ action: 'sendMessage', sessionId: getChatSession(), chatInput: text }) });
    const raw = await response.text(); if (!response.ok) throw new Error(`Webhook trả về lỗi ${response.status}.`);
    let data = raw; try { data = JSON.parse(raw); } catch { /* webhook may return plain text */ }
    const answer = responseText(data).trim(); if (!answer) throw new Error('Webhook chưa trả về nội dung câu trả lời.');
    chatMessages.push({ role: 'assistant', text: answer }); saveChatHistory();
  } catch (error) { chatMessages.push({ role: 'assistant', text: `Mình chưa kết nối được với trợ lý. ${error.message || 'Vui lòng thử lại.'}` }); saveChatHistory(); }
  finally { chatPending = false; $('sendChat').disabled = false; $('chatInput').disabled = false; drawChat(); $('chatInput').focus(); }
}

async function start() {
  try {
    state.documents = await fetch('/documents.json').then((r) => { if (!r.ok) throw Error('manifest'); return r.json(); });
    $('docCount').textContent = String(state.documents.length).padStart(2, '0');
    $('welcomeCount').textContent = `${state.documents.length} tài liệu · ${state.documents.filter((d) => d.type === 'ipynb').length} notebook tương tác`;
    drawFolders(); drawDocs();
    const requested = new URLSearchParams(location.search).get('doc');
    if (requested) openDocument(requested);
  } catch {
    $('welcomeCount').textContent = 'Chưa tải được thư viện tài liệu';
    $('docGrid').innerHTML = '<p class="error-note">Không đọc được danh sách tài liệu. Hãy chạy website qua máy chủ local, không mở trực tiếp file HTML.</p>';
  }
}

function drawFolders() {
  const folders = [...new Set(state.documents.map(folderName))];
  $('folderNav').innerHTML = folders.map((name) => `<button class="nav-item folder-item" data-folder="${escapeHtml(name)}"><span class="nav-icon">▱</span> ${escapeHtml(name)}<span class="nav-count">${state.documents.filter((d) => folderName(d) === name).length}</span></button>`).join('');
  $('folderNav').querySelectorAll('[data-folder]').forEach((b) => b.addEventListener('click', () => setFolder(b.dataset.folder)));
}

function setFolder(folder) {
  state.folder = folder;
  document.querySelectorAll('.nav-item').forEach((b) => b.classList.toggle('active', b.id === 'allDocs' ? folder === 'all' : b.dataset.folder === folder));
  $('crumbCurrent').textContent = folder === 'all' ? 'Tất cả tài liệu' : folder;
  $('sectionTitle').textContent = folder === 'all' ? 'Tài liệu của bạn' : folder;
  $('welcome').hidden = folder !== 'all';
  drawDocs();
}

function visibleDocs() {
  return state.documents.filter((d) => {
    const matchesFolder = state.folder === 'all' || folderName(d) === state.folder;
    const matchesType = state.type === 'all' || (state.type === 'other' ? !['md', 'ipynb'].includes(d.type) : d.type === state.type);
    const searchable = `${d.name} ${d.path}`.toLowerCase().replace(/[_-]+/g, ' ');
    const query = state.query.toLowerCase().replace(/[_-]+/g, ' ').trim();
    return matchesFolder && matchesType && searchable.includes(query);
  });
}

function drawDocs() {
  const docs = visibleDocs();
  $('resultsCount').textContent = `${docs.length} kết quả`;
  $('emptyState').hidden = docs.length > 0;
  $('docGrid').innerHTML = docs.map((d, i) => `<article class="doc-card" style="--i:${i}"><button class="card-open" data-path="${escapeHtml(d.path)}" aria-label="Mở ${escapeHtml(d.name)}"><div class="card-top"><span class="file-icon ${d.type}">${icon(d.type)}</span><span class="file-kind">${labelType(d.type)}</span><span class="card-more">···</span></div><h3>${escapeHtml(pretty(d.name))}</h3><p class="card-folder">${escapeHtml(folderName(d))}</p><div class="card-bottom"><span>${formatSize(d.bytes)}</span><span class="open-label">Mở tài liệu <b>↗</b></span></div></button></article>`).join('');
  $('docGrid').querySelectorAll('[data-path]').forEach((b) => b.addEventListener('click', () => openDocument(b.dataset.path)));
}

function labelType(type) { return ({ ipynb: 'NOTEBOOK', md: 'GHI CHÚ', pdf: 'PDF', docx: 'WORD', py: 'PYTHON', csv: 'CSV' }[type] || type.toUpperCase()); }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]); }
function inlineMarkdown(s) {
  return escapeHtml(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*([^*]+)\*/g, '<em>$1</em>').replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}
function markdown(source) {
  const lines = source.replace(/\r/g, '').split('\n'); let out = ''; let list = ''; let code = false; let codeLines = [];
  const flushList = () => { if (list) { out += `<${list}>` + codeLines.map((x) => `<li>${inlineMarkdown(x)}</li>`).join('') + `</${list}>`; list = ''; codeLines = []; } };
  for (const line of lines) {
    if (line.trim().startsWith('```')) { flushList(); if (code) { out += `<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`; codeLines = []; code = false; } else code = true; continue; }
    if (code) { codeLines.push(line); continue; }
    const item = line.match(/^\s*([-*+] |\d+\. )(.*)$/);
    if (item) { const nextList = /^\s*\d+\./.test(line) ? 'ol' : 'ul'; if (list && list !== nextList) flushList(); list = nextList; codeLines.push(item[2]); continue; }
    flushList();
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { const level = Math.min(h[1].length + 1, 6); out += `<h${level}>${inlineMarkdown(h[2])}</h${level}>`; }
    else if (/^\s*\|.*\|\s*$/.test(line)) { out += `<div class="table-line">${inlineMarkdown(line)}</div>`; }
    else if (/^\s*>/.test(line)) out += `<blockquote>${inlineMarkdown(line.replace(/^\s*>\s?/, ''))}</blockquote>`;
    else if (/^\s*([-*_]\s*){3,}$/.test(line)) out += '<hr />';
    else if (line.trim()) out += `<p>${inlineMarkdown(line)}</p>`;
  }
  flushList(); if (code) out += `<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`;
  return out;
}

async function openDocument(path) {
  const doc = state.documents.find((d) => d.path === path); if (!doc) return;
  history.replaceState(null, '', `?doc=${encodeURIComponent(path)}`);
  $('reader').classList.add('open'); $('reader').setAttribute('aria-hidden', 'false'); document.body.classList.add('reader-open');
  $('readerPath').textContent = doc.path; $('readerHeading').textContent = pretty(doc.name); $('readerBadge').textContent = doc.type.toUpperCase();
  $('downloadLink').href = `/library/${encodePath(doc.path)}`; $('downloadLink').download = doc.name;
  $('readerContent').innerHTML = '<div class="loading">Đang mở tài liệu…</div>';
  try {
    if (doc.type === 'md') $('readerContent').innerHTML = markdown(await fetch(`/library/${encodePath(path)}`).then((r) => r.text()));
    else if (doc.type === 'ipynb') renderNotebook(JSON.parse(await fetch(`/library/${encodePath(path)}`).then((r) => r.text())));
    else if (doc.type === 'pdf') $('readerContent').innerHTML = `<iframe class="pdf-frame" title="${escapeHtml(doc.name)}" src="/library/${encodePath(path)}"></iframe>`;
    else $('readerContent').innerHTML = `<div class="download-prompt"><div class="empty-icon">${icon(doc.type)}</div><h3>Tệp này phù hợp để tải xuống</h3><p>Dùng nút “Tải xuống” phía trên để lưu bản gốc.</p></div>`;
  } catch { $('readerContent').innerHTML = '<p class="error-note">Không đọc được nội dung tệp.</p>'; }
}

function renderNotebook(notebook) {
  const cells = notebook.cells || [];
  $('readerContent').innerHTML = `<div class="notebook-meta">${cells.length} ô · ${(notebook.metadata?.kernelspec?.display_name || 'Jupyter Notebook')}</div>` + cells.map((cell, i) => {
    const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source || '';
    if (cell.cell_type === 'markdown') return `<section class="nb-cell markdown-cell"><div class="cell-count">${String(i + 1).padStart(2, '0')} <span>MARKDOWN</span></div><div class="cell-body">${markdown(source)}</div></section>`;
    const outputs = (cell.outputs || []).map((o) => {
      if (o.output_type === 'stream') return `<pre class="output"><code>${escapeHtml(Array.isArray(o.text) ? o.text.join('') : o.text || '')}</code></pre>`;
      const data = o.data || {};
      if (data['image/png']) { const img = Array.isArray(data['image/png']) ? data['image/png'].join('') : data['image/png']; return `<img class="nb-output-image" alt="Notebook output" src="data:image/png;base64,${img}" />`; }
      const text = data['text/plain'] || data['text/html'];
      if (text) return `<pre class="output"><code>${escapeHtml(Array.isArray(text) ? text.join('') : text)}</code></pre>`;
      if (o.output_type === 'error') return `<pre class="output error-output"><code>${escapeHtml([...(o.traceback || []), `${o.ename || ''}: ${o.evalue || ''}`].join('\n'))}</code></pre>`;
      return '';
    }).join('');
    return `<section class="nb-cell code-cell"><div class="cell-count">${String(i + 1).padStart(2, '0')} <span>CODE${cell.execution_count ? ` · [${cell.execution_count}]` : ''}</span></div><div class="cell-body"><pre><code>${escapeHtml(source)}</code></pre>${outputs}</div></section>`;
  }).join('');
}

function closeReader() { $('reader').classList.remove('open'); $('reader').setAttribute('aria-hidden', 'true'); document.body.classList.remove('reader-open'); history.replaceState(null, '', location.pathname); }
$('allDocs').addEventListener('click', () => setFolder('all'));
$('searchInput').addEventListener('input', (e) => { state.query = e.target.value; drawDocs(); });
$('typeFilter').addEventListener('change', (e) => { state.type = e.target.value; drawDocs(); });
$('closeReader').addEventListener('click', closeReader); $('closeIcon').addEventListener('click', closeReader); $('readerBackdrop').addEventListener('click', closeReader);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeReader(); if (!$('chatPage').hidden) closeChatPage(); } if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) { e.preventDefault(); $('searchInput').focus(); } });
$('openChat').addEventListener('click', () => openChatPage());
$('closeChat').addEventListener('click', () => closeChatPage());
$('clearChat').addEventListener('click', () => { chatMessages = []; localStorage.removeItem(CHAT_HISTORY_KEY); localStorage.removeItem(CHAT_SESSION_KEY); drawChat(); });
$('chatForm').addEventListener('submit', (event) => { event.preventDefault(); const input = $('chatInput'); const text = input.value.trim(); if (!text) return; input.value = ''; sendChatMessage(text); });
window.addEventListener('popstate', () => { if (new URLSearchParams(location.search).get('view') === 'chat') openChatPage(true); else if (!$('chatPage').hidden) closeChatPage(false); });
if (new URLSearchParams(location.search).get('view') === 'chat') openChatPage(true);
start();
