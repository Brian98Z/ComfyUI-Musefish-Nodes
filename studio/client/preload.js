const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("workbench", {
  checkPort: () => ipcRenderer.invoke("check-port"),
  getUrl: () => ipcRenderer.invoke("get-url"),
  onServerLog: (cb) => ipcRenderer.on("server-log", (_e, line) => cb(line)),
});
