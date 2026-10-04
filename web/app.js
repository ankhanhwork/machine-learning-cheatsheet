const state = { documents: [], folder: 'all', query: '', type: 'all' };
const $ = (id) => document.getElementById(id);
const encodePath = (path) => path.split('/').map(encodeURIComponent).join('/');
const pretty = (name) => name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
const formatSize = (bytes) => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
const icon = (type) => ({ ipynb: '◉', md: '▤', pdf: '▧', docx: '▧', py: '‹›', csv: '▦' }[type] || '▤');
const folderName = (doc) => doc.path.includes('/') ? doc.path.split('/')[0] : 'Tài liệu khác';

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
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeReader(); if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) { e.preventDefault(); $('searchInput').focus(); } });
start();
