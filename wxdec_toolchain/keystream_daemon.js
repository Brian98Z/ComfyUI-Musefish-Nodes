// Musefish WeChat Channels keystream daemon.
//
// Long-lived Node worker that loads WeChat's own wasm_video_decode module and
// answers line-based JSON requests: {"id":N,"key":"<decode_key>"} ->
// {"id":N,"ok":true,"keystream":"<base64>"} or {"id":N,"ok":false,"error":...}.
//
// Why a daemon: the upstream repo boots a Playwright browser per service
// because emscripten's async wasm startup can stall under load; a warm
// process amortizes that cost. Measured behavior (F:/OMP/wxdec, 2026-01):
// module init dominates (~2-20s once), subsequent generate() calls take
// ~0.4s each, so the process also caches keystreams per seed.
//
// The upstream glue declares `var Module = ...` at top level. Evaluated
// inside this closure, `var` would shadow the `Module` object we inject, so
// the source is rewritten to reference our binding before vm evaluation —
// same patch the skill's gen_keystream.js applied ("var Module" -> "Module").
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const readline = require('readline');

const TOOLCHAIN_DIR = __dirname;
const GLUE_PATH = path.join(TOOLCHAIN_DIR, 'wasm_video_decode.js');
const WASM_PATH = path.join(TOOLCHAIN_DIR, 'wasm_video_decode.wasm');
const KEYSTREAM_SIZE = 131072; // only the first 128 KiB of a Channels video is encrypted

function send(message) {
  process.stdout.write(JSON.stringify(message) + '\n');
}

function patchGlue(source) {
  // `var Module = typeof Module !== 'undefined' ? Module : {};`
  // -> keep the fallback branch only, so our sandbox Module survives.
  return source.replace(
    /^var Module = typeof Module !== 'undefined' \? Module : \{ };/m,
    'Module = (typeof Module !== "undefined" && Module) ? Module : {};'
  ).replace(
    /var Module = typeof Module !== 'undefined' \? Module : \{\};/,
    'Module = (typeof Module !== "undefined" && Module) ? Module : {};'
  );
}

function createSandbox() {
  const sandbox = {
    // The glue reads these two before instantiating.
    VTS_WASM_URL: WASM_PATH,
    MAX_HEAP_SIZE: 33554432,
    Module: {},
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    WebAssembly,
    TextDecoder,
    TextEncoder,
    URL,
    performance,
    // Emscripten probes these in Node builds; keep them present but neutral.
    process: { platform: 'browser', arch: 'wasm32', env: {}, argv: [], version: '' },
    require: undefined,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  // Emscripten's environment detection reads self.location.href when it
  // decides it is running in a worker-like context; give it a browser-ish
  // stub so the file path above is used untouched.
  sandbox.location = { href: 'file://' + TOOLCHAIN_DIR.replace(/\\/g, '/') + '/' };
  return sandbox;
}

let ready = false;
let readyError = null;
let initResolve = null;
const readyPromise = new Promise((resolve) => { initResolve = resolve; });

// Filled by the wasm_isaac_generate hook for each generate() call.
let keystreamSink = null;

function boot() {
  const sandbox = createSandbox();
  const context = vm.createContext(sandbox);

  // The glue invokes this C-ctor-backed ASM_CONST whenever the wasm Isaac64
  // stream generator has `size` bytes ready at `ptr`. Upstream reverses the
  // bytes here — required, otherwise decryption yields garbage.
  context.wasm_isaac_generate = (ptr, size) => {
    const bytes = new Uint8Array(sandbox.Module.HEAPU8.buffer, ptr, size);
    if (keystreamSink) {
      keystreamSink.set(Array.from(bytes).reverse());
    }
  };

  const glue = patchGlue(fs.readFileSync(GLUE_PATH, 'utf8'));
  // Inject the wasm binary before run() executes: the glue honors
  // Module['wasmBinary'] and skips network/file fetch entirely.
  context.Module.wasmBinary = new Uint8Array(fs.readFileSync(WASM_PATH));
  try {
    vm.runInContext(glue, context, { filename: 'wasm_video_decode.js' });
  } catch (error) {
    readyError = 'glue evaluation failed: ' + (error && error.stack || error);
    initResolve(false);
    return null;
  }

  // WxIsaac64 is registered on Module via embind once __wasm_call_ctors runs.
  // Poll rather than guessing the exact onRuntimeInitialized timing.
  const startedAt = Date.now();
  const poll = setInterval(() => {
    const mod = context.Module;
    if (mod && mod.WxIsaac64) {
      clearInterval(poll);
      context.__musefishContext = context;
      global.__musefishSandbox = sandbox;
      ready = true;
      initResolve(true);
      return;
    }
    if (Date.now() - startedAt > 120000) {
      clearInterval(poll);
      readyError = 'WxIsaac64 not registered after 120s';
      initResolve(false);
    }
  }, 100);

  return { sandbox, context };
}

let runtime = null;
const cache = new Map(); // seed -> Buffer

function generate(seed) {
  const key = String(seed);
  const cached = cache.get(key);
  if (cached) return cached;

  const sandbox = runtime.sandbox;
  const context = runtime.context;
  const stream = Buffer.alloc(KEYSTREAM_SIZE);
  keystreamSink = new Uint8Array(KEYSTREAM_SIZE);

  const ctor = context.Module.WxIsaac64;
  // The wasm binding declares the seed as std::string — the RPC worker in
  // the upstream repo passes the raw JSON string too. Numbers throw
  // "Cannot pass non-string to std::string".
  const decryptor = new ctor(key);
  try {
    decryptor.generate(KEYSTREAM_SIZE);
  } finally {
    if (typeof decryptor.delete === 'function') decryptor.delete();
  }
  if (!keystreamSink) throw new Error('wasm_isaac_generate never fired');
  for (let i = 0; i < KEYSTREAM_SIZE; i++) stream[i] = keystreamSink[i];
  keystreamSink = null;

  cache.set(key, stream);
  if (cache.size > 64) cache.delete(cache.keys().next().value);
  return stream;
}

runtime = boot();

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', async (line) => {
  const text = line.trim();
  if (!text) return;
  let request;
  try {
    request = JSON.parse(text);
  } catch {
    send({ id: null, ok: false, error: 'unparseable request line' });
    return;
  }
  const id = request.id === undefined ? null : request.id;
  if (request.op === 'ping') {
    await readyPromise;
    send({ id, op: 'ping', ok: ready, ready, error: readyError });
    return;
  }
  if (request.op === 'shutdown') {
    process.exit(0);
    return;
  }
  if (request.op !== 'generate') {
    send({ id, ok: false, error: 'unknown op' });
    return;
  }
  await readyPromise;
  if (!ready) {
    send({ id, ok: false, error: readyError || 'wasm runtime not ready' });
    return;
  }
  const seed = Number(request.key);
  if (!Number.isFinite(seed) || seed <= 0) {
    send({ id, ok: false, error: `invalid decode_key: ${request.key}` });
    return;
  }
  try {
    const stream = generate(seed);
    send({ id, ok: true, keystream: stream.toString('base64') });
  } catch (error) {
    send({ id, ok: false, error: String(error && error.message || error) });
  }
});
rl.on('close', () => process.exit(0));
