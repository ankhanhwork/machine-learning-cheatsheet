import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const here = path.dirname(fileURLToPath(import.meta.url));
const build = spawn(process.execPath, [path.join(here, 'build.mjs')], { stdio: 'inherit' });
build.on('exit', (code) => {
  if (code !== 0) process.exit(code ?? 1);
  spawn(process.execPath, [path.join(here, 'server.mjs')], { stdio: 'inherit' });
});
