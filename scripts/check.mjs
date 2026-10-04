import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve('.');
const dist = path.join(root, 'dist');
const docs = JSON.parse(await readFile(path.join(dist, 'documents.json'), 'utf8'));
assert.ok(docs.length >= 5, `Expected at least 5 documents; found ${docs.length}`);
assert.ok(docs.some((doc) => doc.type === 'ipynb'), 'Notebook files should be indexed');
assert.ok(docs.some((doc) => doc.type === 'md'), 'Markdown files should be indexed');
for (const doc of docs) {
  const file = path.join(dist, 'library', doc.path);
  const info = await stat(file);
  assert.equal(info.size, doc.bytes, `${doc.path} should be copied intact`);
  if (doc.type === 'ipynb') {
    const notebook = JSON.parse(await readFile(file, 'utf8'));
    assert.ok(Array.isArray(notebook.cells), `${doc.path} should be a valid Jupyter notebook`);
    assert.ok(notebook.cells.length > 0, `${doc.path} should contain cells`);
  }
}
const html = await readFile(path.join(dist, 'index.html'), 'utf8');
const app = await readFile(path.join(dist, 'app.js'), 'utf8');
assert.match(html, /id="docGrid"/, 'Document list container should exist');
assert.match(html, /id="downloadLink"[^>]*download/, 'Reader should have a download link');
assert.match(app, /renderNotebook/, 'Notebook reader should be included');
assert.ok(app.includes('.replace(/[_-]+/g'), 'Search should normalize filename separators');
console.log(`Checks passed: ${docs.length} files indexed; ${docs.filter((d) => d.type === 'ipynb').length} valid notebooks.`);
