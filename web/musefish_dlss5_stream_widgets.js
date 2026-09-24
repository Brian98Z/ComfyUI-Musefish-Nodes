import { app } from "../../scripts/app.js";

const STREAM_TYPE = "MusefishDLSS5VideoStream";
const IMAGE_TYPE = "MusefishDLSS5NeuralRender";
const CURRENT = ["super_resolution", "style", "vsr_quality", "intensity", "local_tone",
  "local_struct", "skin_struct", "use_auto_mask", "crf", "encoder"];
const PREVIOUS = ["style", "intensity", "local_tone", "local_struct", "skin_struct",
  "use_auto_mask", "super_resolution", "vsr_quality", "crf", "encoder"];
const WITH_PREFIX = ["filename_prefix", ...PREVIOUS.slice(0, -1)];
const IMAGE_CURRENT = ["super_resolution", "style", "vsr_quality", "intensity", "local_tone",
  "local_struct", "skin_struct", "use_auto_mask", "reset_every_n_frames", "keep_session"];
const IMAGE_PREVIOUS = ["style", "intensity", "local_tone", "local_struct", "skin_struct",
  "use_auto_mask", "reset_every_n_frames", "super_resolution", "vsr_quality", "keep_session"];
const DEFAULTS = {
  super_resolution: "2× (Balance)", style: "default", vsr_quality: "ultra", reset_every_n_frames: 0, keep_session: "auto",
  intensity: 1, local_tone: 0.94, local_struct: 0.84, skin_struct: 1,
  use_auto_mask: true, crf: 19, encoder: "libx264（CPU 编码）",
};
const SCALE_VALUES = { "off": "off", "1x": "1× (Native)",
  "1x (DLAA / native)": "1× (Native)",
  "1.5x": "1.5× (Quality)", "1.5x (Quality · composite)": "1.5× (Quality)",
  "1.724x (Balanced · composite)": "1.5× (Quality)",
  "2x": "2× (Balance)", "2x (Performance)": "2× (Balance)",
  "3x (Ultra Performance · composite)": "3× (Performance)",
  "4x": "4× (Ultra)" };
const ENCODER_VALUES = { libx264: "libx264（CPU 编码）",
  h264_nvenc: "h264_nvenc（GPU 编码加速）" };

function migrate(node) {
  if (!Array.isArray(node?.widgets_values) ||
    (node.type !== STREAM_TYPE && node.type !== IMAGE_TYPE)) return;
  const values = node.widgets_values;
  let old;
  const isImage = node.type === IMAGE_TYPE;
  if (isImage) old = typeof values[0] === "string" &&
    (values[0] in SCALE_VALUES || values[0].includes("× (") ||
    ["1K", "2K", "4K", "8K"].includes(values[0])) ? IMAGE_CURRENT : IMAGE_PREVIOUS;
  else if (typeof values[0] === "string" && values[0].startsWith("Musefish/")) old = WITH_PREFIX;
  else if (typeof values[0] === "string" && (values[0] in SCALE_VALUES ||
    values[0].includes("× (") || ["1K", "2K", "4K", "8K"].includes(values[0]))) old = CURRENT;
  else old = PREVIOUS;
  const named = node.widgets_values_named || {};
  const order = isImage ? IMAGE_CURRENT : CURRENT;
  const result = order.map((key) => {
    const index = old.indexOf(key);
    const value = index >= 0 && index < values.length ? values[index] : named[key] ?? DEFAULTS[key];
    if (key === "super_resolution") return SCALE_VALUES[value] ?? value;
    if (key === "encoder") return ENCODER_VALUES[value] ?? value;
    return value;
  });
  node.widgets_values = result;
  node.widgets_values_named = Object.fromEntries(order.map((key, index) => [key, result[index]]));
  if (!isImage) node.inputs = node.inputs?.filter((input) => input.name !== "filename_prefix");
}

app.registerExtension({
  name: "Musefish.DLSS5StreamWidgets",
  beforeConfigureGraph(graphData) {
    for (const node of graphData?.nodes ?? []) migrate(node);
    for (const graph of Object.values(graphData?.definitions?.subgraphs ?? {})) {
      for (const node of graph?.nodes ?? []) migrate(node);
    }
  },
});
