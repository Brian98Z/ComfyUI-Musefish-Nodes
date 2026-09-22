/**
 * Musefish Video Download — parallel-segments UX.
 *
 * The `parallel_segments` combo only means something when the URL needs
 * concurrent fetching (direct CDN MP4s like Douyin, or fragmented
 * DASH/HLS streams like YouTube/Bilibili). This extension:
 *  1. Hides the widget while the URL field is empty or matches no
 *     segment-capable platform, so unsupported platforms never see a
 *     knob that would do nothing.
 *  2. Snaps the value to the platform's optimum (8 for Douyin/YouTube/
 *     Bilibili) the first time a supported URL appears, while still
 *     letting the user override it afterwards.
 */
import { app } from "../../../scripts/app.js";

const NODE_TYPE = "MusefishVideoDownload";
const URL_WIDGET = "url";
const SEGMENTS_WIDGET = "chunks";

/** Platforms whose downloads benefit from parallel segments. */
const SUPPORTED = [
  /v\.douyin\.com/i,
  /douyin\.com|iesdouyin\.com/i,
  /finder\.video\.qq\.com/i,
  /youtube\.com|youtu\.be/i,
  /bilibili\.com|b23\.tv/i,
];
/** Optimal segment counts per platform family (first match wins). */
const OPTIMUM = [
  [/v\.douyin\.com|douyin\.com|finder\.video\.qq\.com/i, "8"],
  [/youtube\.com|youtu\.be/i, "8"],
  [/bilibili\.com|b23\.tv/i, "8"],
];

app.registerExtension({
  name: "Musefish.VideoDownload.ParallelSegments",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const urlWidget = this.widgets.find((w) => w.name === URL_WIDGET);
      const segWidget = this.widgets.find((w) => w.name === SEGMENTS_WIDGET);
      if (!urlWidget || !segWidget) return result;

      let autoSnapped = false;
      const syncVisibility = () => {
        const url = String(urlWidget.value || "");
        const supported = url && SUPPORTED.some((re) => re.test(url));
        segWidget.hidden = !supported;
        segWidget.disabled = !supported;
        this.onResize?.(this.size);
        app.graph.setDirtyCanvas(true, true);
      };
      const snapOptimum = () => {
        const url = String(urlWidget.value || "");
        if (!autoSnapped) {
          const hit = OPTIMUM.find(([re]) => re.test(url));
          if (hit) {
            segWidget.value = hit[1];
            autoSnapped = true;
          }
        }
      };
      const originalCallback = urlWidget.callback;
      urlWidget.callback = function (...args) {
        if (originalCallback) originalCallback.apply(this, args);
        snapOptimum();
        syncVisibility();
      };
      // initial state + react to paste/serialize round-trips
      const origDraw = this.onDrawBackground;
      this.onDrawBackground = function (ctx) {
        snapOptimum();
        syncVisibility();
        if (origDraw) origDraw.apply(this, arguments);
      };
      syncVisibility();
      return result;
    };
  },
});
