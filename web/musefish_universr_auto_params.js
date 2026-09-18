/**
 * Musefish UniverSR audio nodes — auto-parameter switch UX.
 *
 * 1. The switch that governs `input_sr` / `ode_steps` / `guidance` sits first on the node (its
 *    position comes from the node schema), with those three directly below it. While it is on they
 *    are hidden, so the node shows at a glance what the material-driven matcher is going to run;
 *    switching it off reveals them again for manual selection.
 *
 * 2. Workflows saved before the switch took the first position store widget values by position only,
 *    and ComfyUI restores them by position unless `Comfy.Workflow.NamedValuesRestore` is enabled.
 *    Their legacy layout is mapped back onto the current order by widget name before the graph is
 *    configured, so earlier files keep their settings instead of shifting by several slots.
 */
import { app } from "../../../scripts/app.js";

const NODE_TYPES = ["MusefishUniverSRGeneralAudio", "MusefishUniverSRSpeechAudio"];
/** Widgets the matcher decides: hidden while the switch is on. Mirrors the governed inputs in
 *  musefish_audio.py. */
const GOVERNED_WIDGETS = ["input_sr", "ode_steps", "guidance"];
const TOGGLE_WIDGET = "auto_params";
/** Widget order each node type had before the switch moved to the front. */
const LEGACY_WIDGET_ORDER = {
  MusefishUniverSRGeneralAudio: [
    "mode", "input_sr", "channel_mode", "ode_method", "ode_steps", "guidance", "chunk_sec", "seed",
    "control_after_generate", "deess", "auto_params",
  ],
  MusefishUniverSRSpeechAudio: [
    "input_sr", "channel_mode", "ode_method", "ode_steps", "guidance", "chunk_sec", "seed",
    "control_after_generate", "deess", "auto_params",
  ],
};
/** Widget order the nodes register today; mirrors the schema in musefish_audio.py. */
const CURRENT_WIDGET_ORDER = {
  MusefishUniverSRGeneralAudio: [
    "auto_params", "input_sr", "ode_steps", "guidance", "mode", "channel_mode", "ode_method",
    "chunk_sec", "seed", "control_after_generate", "deess", "accel",
  ],
  MusefishUniverSRSpeechAudio: [
    "auto_params", "input_sr", "ode_steps", "guidance", "channel_mode", "ode_method", "chunk_sec",
    "seed", "control_after_generate", "deess", "accel",
  ],
};
/** Renamed before release: `default` was the old name of the no-flags baseline. */
const LEGACY_ACCEL_NAMES = { default: "fp32" };
/** Declared defaults, used for slots a file predates. Mirrors the schema defaults. */
const WIDGET_DEFAULTS = {
  auto_params: true,
  input_sr: "auto",
  ode_steps: 4,
  guidance: 1.5,
  mode: "auto",
  channel_mode: "auto",
  ode_method: "midpoint",
  chunk_sec: 15,
  seed: 0,
  control_after_generate: "randomize",
  deess: true,
  accel: "cuDNN TF32",
};

function isNodeType(type) {
  return NODE_TYPES.includes(type);
}

/**
 * Rewrites a stored value list from the legacy layout onto the current one by widget name. Values a
 * file never stored fall back to their declared default so every slot stays aligned with its widget;
 * lists already in the current layout (they start with the switch's boolean) pass through untouched.
 */
function migrateLegacyWidgetValues(values, nodeType) {
  if (!Array.isArray(values) || values.length === 0) return values;
  if (typeof values[0] !== "string") return values;
  const legacyOrder = LEGACY_WIDGET_ORDER[nodeType];
  const stored = Math.min(values.length, legacyOrder.length);
  return CURRENT_WIDGET_ORDER[nodeType].map((name) => {
    const legacyIndex = legacyOrder.indexOf(name);
    if (legacyIndex < 0) return WIDGET_DEFAULTS[name];
    return legacyIndex < stored ? values[legacyIndex] : WIDGET_DEFAULTS[name];
  });
}

function migrateGraphData(graphData) {
  const nodeLists = [graphData?.nodes];
  for (const subgraph of Object.values(graphData?.definitions?.subgraphs ?? {})) {
    nodeLists.push(subgraph?.nodes);
  }
  for (const nodes of nodeLists) {
    for (const node of nodes ?? []) {
      if (!isNodeType(node?.type)) continue;
      node.widgets_values = migrateLegacyWidgetValues(node.widgets_values, node.type);
    }
  }
}

function syncGovernedVisibility(node, force = false) {
  const toggle = node.widgets?.find((widget) => widget.name === TOGGLE_WIDGET);
  if (!toggle) return;
  const hide = toggle.value !== false;
  let changed = false;
  for (const widget of node.widgets ?? []) {
    if (!GOVERNED_WIDGETS.includes(widget.name)) continue;
    if (force || Boolean(widget.hidden) !== hide) changed = true;
    widget.hidden = hide;
    if (widget.options) widget.options.hidden = hide;
  }
  if (!changed) return;
  const size = node.computeSize?.();
  if (size) node.setSize([Math.max(node.size?.[0] ?? 0, size[0]), size[1]]);
  app?.graph?.setDirtyCanvas?.(true, true);
}

/**
 * Repairs values the schema would reject, so a session saved against an older node cannot wedge the
 * queue: a pre-rename accel name, or a number typed into the editable `input_sr` combo (ComfyUI
 * compares combo values by identity, so `24000` never matches `"24000"`).
 */
function repairStaleValues(node) {
  const widget = (name) => node.widgets?.find((candidate) => candidate.name === name);
  const accel = widget("accel");
  if (accel && LEGACY_ACCEL_NAMES[accel.value]) accel.value = LEGACY_ACCEL_NAMES[accel.value];
  const inputSr = widget("input_sr");
  if (!inputSr) return;
  if (typeof inputSr.value === "number") inputSr.value = String(inputSr.value);
  if (inputSr.musefishInputSrCoerced) return;
  inputSr.musefishInputSrCoerced = true;
  const previousSerialize = inputSr.serializeValue;
  inputSr.serializeValue = function (...args) {
    const value = previousSerialize ? previousSerialize.apply(this, args) : this.value;
    return typeof value === "number" ? String(value) : value;
  };
}

function installToggle(node) {
  if (!isNodeType(node?.comfyClass) && !isNodeType(node?.type)) return;
  const toggle = node.widgets?.find((widget) => widget.name === TOGGLE_WIDGET);
  if (!toggle || toggle.musefishAutoParamsHooked) return;
  toggle.musefishAutoParamsHooked = true;
  const previousCallback = toggle.callback;
  toggle.callback = function (...args) {
    const result = previousCallback?.apply(this, args);
    syncGovernedVisibility(node);
    return result;
  };
  syncGovernedVisibility(node, true);
  // Loaded graphs apply their stored widget values after the node is created.
  requestAnimationFrame(() => syncGovernedVisibility(node, true));
}

app.registerExtension({
  name: "Musefish.UniverSRAutoParams",
  beforeConfigureGraph(graphData) {
    migrateGraphData(graphData);
  },
  nodeCreated(node) {
    installToggle(node);
    repairStaleValues(node);
  },
  loadedGraphNode(node) {
    installToggle(node);
    repairStaleValues(node);
  },
});
