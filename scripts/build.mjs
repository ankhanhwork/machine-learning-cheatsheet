import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'dist');
const library = path.join(output, 'library');
const allowed = new Set(['.md', '.ipynb', '.pdf', '.docx', '.csv', '.py']);
const ignored = new Set(['.git', 'node_modules', 'dist', 'generator', '.docx-render', '.vercel']);
const files = [];

async function collect(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || ignored.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await collect(full);
    else if (allowed.has(path.extname(entry.name).toLowerCase())) {
      const relative = path.relative(root, full).split(path.sep).join('/');
      files.push({ path: relative, name: entry.name, type: path.extname(entry.name).slice(1).toLowerCase(), bytes: (await readFile(full)).byteLength });
    }
  }
}

await rm(output, { recursive: true, force: true });
await mkdir(library, { recursive: true });
await collect(root);
files.sort((a, b) => a.path.localeCompare(b.path));
for (const file of files) {
  const destination = path.join(library, file.path);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(path.join(root, file.path), destination);
}
await cp(path.join(root, 'web'), output, { recursive: true });
await writeFile(path.join(output, 'documents.json'), JSON.stringify(files, null, 2));
console.log(`Built ${files.length} documents into dist/`);
