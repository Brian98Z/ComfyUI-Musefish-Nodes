// Musefish Studio Electron client: spawn the local server, embed its UI.
const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn, execFile } = require("child_process");
const net = require("net");
const path = require("path");
const fs = require("fs");

function resolveRoot() {
  // electron-builder portable self-extracts to %TEMP%\...: __dirname and
  // process.execPath both point there, so ROOT must come from
  // PORTABLE_EXECUTABLE_DIR (the exe's real location). The exe may sit in
  // <root>\client\ or directly in <root>\; MUSEFISH_STUDIO_ROOT overrides.
  const candidates = [
    process.env.MUSEFISH_STUDIO_ROOT,
    process.env.PORTABLE_EXECUTABLE_DIR,
    process.env.PORTABLE_EXECUTABLE_DIR
      ? path.resolve(process.env.PORTABLE_EXECUTABLE_DIR, "..")
      : null,
    path.resolve(__dirname, ".."),
    path.resolve(path.dirname(process.execPath), ".."),
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, "server", "app.py"))) return path.resolve(c);
  }
  return path.resolve(path.dirname(process.execPath), "..");
}
const ROOT = resolveRoot();
const SERVER_DIR = path.join(ROOT, "server");
const SERVER_SCRIPT = path.join(SERVER_DIR, "app.py");
const PY_CANDIDATES = [
  process.env.MUSEFISH_PYTHON,
  path.join(ROOT, "runtime", "python", "python.exe"),
  "python.exe",
].filter(Boolean);

let win = null;
let serverProc = null;
let serverPort = 17998;
let serverUrl = `http://127.0.0.1:${serverPort}/`;
const APP_ICON = path.join(__dirname, "..", "assets", "icon.ico");

function resolvePy() {
  for (const p of PY_CANDIDATES) {
    try { if (fs.existsSync(p) && fs.statSync(p).isFile()) return p; } catch {}
  }
  return PY_CANDIDATES[PY_CANDIDATES.length - 1];
}

function sendServerLog(line) {
  if (win && !win.isDestroyed()) win.webContents.send("server-log", line);
}

function startServer() {
  const py = resolvePy();
  const env = { ...process.env, PYTHONIOENCODING: "utf-8" };
  serverProc = spawn(py, ["-u", SERVER_SCRIPT], {
    cwd: SERVER_DIR,
    env,
    windowsHide: true,
  });
  const send = sendServerLog;
  serverProc.stdout.on("data", (d) => {
    const text = d.toString();
    send(text);
    const m = text.match(/\{"server_port":\s*(\d+)\}/);
    if (m) {
      serverPort = Number(m[1]);
      serverUrl = `http://127.0.0.1:${serverPort}/`;
    }
  });
  serverProc.stderr.on("data", (d) => send(d.toString()));
  serverProc.on("error", (err) => {
    // spawn failure (missing python, bad cwd) — exit never fires for these
    send(`[server spawn failed: ${err.message}]`);
    serverProc = null;
  });
  serverProc.on("exit", (code) => {
    send(`[server exited code=${code}]`);
    serverProc = null;
  });
}

function stopServer() {
  if (!serverProc) return;
  const pid = serverProc.pid;
  serverProc = null;
  if (process.platform === "win32") {
    execFile("taskkill", ["/pid", String(pid), "/T", "/F"], () => {});
  } else {
    try { process.kill(-pid); } catch { try { serverProc.kill(); } catch {} }
  }
}

function checkPort(port, cb) {
  const s = net.connect({ port, host: "127.0.0.1", timeout: 4000 }, () => { s.destroy(); cb(true); });
  s.on("error", () => { s.destroy(); cb(false); });
  s.on("timeout", () => { s.destroy(); cb(false); });
}

// Probe whether the port already serves OUR workbench (not some other app).
function probeHealth(port, cb) {
  const req = require("http").get(
    { host: "127.0.0.1", port, path: "/api/status", timeout: 3000 },
    (res) => {
      let body = "";
      res.on("data", (c) => { body += c; if (body.length > 4096) res.destroy(); });
      res.on("end", () => cb(res.statusCode === 200 && body.includes("comfy")));
    });
  req.on("error", () => cb(false));
  req.on("timeout", () => { req.destroy(); cb(false); });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 960,
    minHeight: 600,
    title: "Musefish Studio",
    icon: APP_ICON,
    backgroundColor: "#0b0d12",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });
  win.setMenuBarVisibility(false);
  // Wait until the port serves OUR workbench before loading: a bare TCP
  // check would happily load whatever else grabbed the port (e.g. RVC变声
  // 工作室 defaults to the same port) into our window.
  const tryLoad = () => {
    probeHealth(serverPort, (ours) => {
      if (ours) {
        win.loadURL(serverUrl).catch(() => setTimeout(tryLoad, 2000));
      } else if (!win.isDestroyed()) {
        setTimeout(tryLoad, 2000);
      }
    });
  };
  // Reuse an already-running healthy workbench server on the port (e.g. one
  // started from CLI) instead of spawning a second copy — double-bind made
  // requests land on the stale process at random.
  probeHealth(serverPort, (healthy) => {
    if (healthy) {
      sendServerLog(`[client] reusing existing server on ${serverPort}`);
    } else {
      startServer();
    }
    tryLoad();
  });
  win.on("closed", () => { win = null; });
}

ipcMain.handle("check-port", () => new Promise((res) => checkPort(serverPort, res)));
ipcMain.handle("get-url", () => serverUrl);

app.whenReady().then(createWindow);
app.on("window-all-closed", () => { stopServer(); app.quit(); });
app.on("before-quit", stopServer);
