# GSCodec.js — browser decoder & WebGL viewer

Decode **GSCodec `seq_hevc`** compression folders (attribute MP4s + `gop*_meta.json`) in the browser and render them with [Three.js GaussianSplat](https://threejs.org/docs/#GaussianSplat) on `WebGPURenderer`. No Node.js at runtime. The page must include the import map from `demo/index.html` (`three/webgpu`, `three/tsl`, `three/addons/`).

## Requirements

- **WebGL2** (Chrome, Firefox, Safari, Edge)
- Attribute MP4s are **HEVC (H.265)**. Native `<video>` decode works on Safari / many Edge builds. On Chrome (and others without HEVC), the library **falls back to ffmpeg.wasm** vendored under `web/gscodec-js/vendor/ffmpeg/` (served from your static HTTP server, no CDN).
- Compression layout must match Python `examples/decode_and_view.py` / `SeqHevcCompression` (PLAS grid, per-attribute MP4s, optional `unity_manifest.json`).

## Quick start

### Local folder (recommended for `demo/example`)

Browsers cannot read an arbitrary disk path from a URL. Use the demo’s **“pick a local drive folder”** control (maps files to `blob:` URLs via `createFileMapSource`):

1. Open `web/gscodec-js/demo/index.html` (HTTP server or `file://`).
2. Choose the `compression` folder under `demo/example/geohuman/.../` (or its parent).
3. Click **Load & decode**.

### Bundled example over HTTP

1. Serve the repo (modules need HTTP if you use the relative preset URL):

   ```bash
   # from repo root
   python -m http.server 8765
   ```

2. Open `http://localhost:8765/web/gscodec-js/demo/index.html`.
3. Keep the bundled preset (`./example/.../compression/`) and click **Load & decode**.

## Embed in your web app

```html
<canvas id="gs" style="width:100%;height:70vh"></canvas>
<script type="module">
  import { createViewer } from "/web/gscodec-js/src/index.js";

  const canvas = document.getElementById("gs");
  const viewer = await createViewer(canvas, {
    baseUrl: "https://cdn.example.com/MyScene/compression/",
    maxSplats: 400_000,
    frame: 0,
  });
  // viewer.setFrame(1); viewer.stop(); viewer.start();
</script>
```

Decode only (no viewer):

```javascript
import { createUrlSource, decodeSequence, activateFrame } from "/web/gscodec-js/src/index.js";

const source = createUrlSource("https://cdn.example.com/MyScene/compression/");
const frames = await decodeSequence(source, { onProgress: console.log });
const gpu = activateFrame(frames[0]);
// gpu.positions, gpu.scales, gpu.rotations, gpu.colors (Float32Array)
```

## API

| Export | Role |
|--------|------|
| `createUrlSource(baseUrl)` | HTTP / relative URL asset loader |
| `createFileMapSource(files)` | Local folder from `<input webkitdirectory>` |
| `decodeSequence(source, options)` | Full pipeline → `SplatFrame[]` |
| `activateFrame(frame)` | exp/sigmoid/normalize → render buffers |
| `GSCodecViewer` / `createViewer` | Decode + orbit camera + WebGL draw |
| `hevcLikelySupported()` | Quick WebCodecs HEVC probe |

## Pipeline (matches Python)

1. For each GOP / layer (`scene`, optional `env_`): read `gopN_meta.json`
2. Decode attribute MP4s (`means` 16-bit L/U, `quats` w+xyx, `shN` 15 bands, etc.)
3. Min–max denormalize → tensor → `deorganize` → YCbCr→RGB on `sh0`/`shN` when flagged
4. Merge env+scene, trim PLAS padding (`scales < -9.7`), optional `n_orig_per_frame`
5. **Activate** and draw with instanced Gaussian **point sprites** (preview quality; not identical to gsplat CUDA rasterizer)

## Limitations

- Large scenes (>1M splats): use `maxSplats` for interactive preview; full count decode still runs in memory.
- No view-dependent SH shading in the default renderer (DC color from `sh0` only).
- Sequential `<video>` frame seek is used instead of ffmpeg; all-intra bitstreams work best.

## Files

- `src/decode/` — MP4 → tensors → splat frames  
- `src/video/mp4-frames.js` — browser video decode  
- `src/render/` — WebGL2 splat renderer  
- `demo/index.html` — minimal UI  

Older TypeScript stubs in `src/*.ts` are superseded by the `.js` modules above.
