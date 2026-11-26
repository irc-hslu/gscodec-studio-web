/**
 * Decode attribute HEVC MP4s in the browser (no Node.js).
 *
 * These files are quantized attribute planes (positions, scales, quats), not pictures.
 * A `<video>` element runs them through the browser color pipeline and changes the
 * stored bytes, so decode always goes through vendored ffmpeg.wasm to PNG, then
 * inflates those PNG bytes. A bare `ffmpeg -i` probe is not used: a failing
 * ffmpeg call aborts the wasm runtime (`Aborted()`).
 */

let _ffmpeg = null;
let _ffmpegLoading = null;

/**
 * @param {string} url
 * @param {number} numFrames
 * @param {{ grayscale?: boolean, width?: number, height?: number, onProgress?: (msg: string) => void }} options
 * @returns {Promise<Uint8Array[]>}
 */
export async function decodeMp4ToFrames(url, numFrames, options = {}) {
  const grayscale = Boolean(options.grayscale);
  const log = options.onProgress ?? (() => {});
  return decodeWithFfmpeg(url, numFrames, {
    grayscale,
    width: options.width,
    height: options.height,
    log,
  });
}

async function decodeWithHtmlVideo(url, numFrames, grayscale) {
  const video = document.createElement("video");
  if (/^https?:/i.test(url)) video.crossOrigin = "anonymous";
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.defaultPlaybackRate = 1;

  const hevc =
    video.canPlayType('video/mp4; codecs="hev1.1.6.L93.B0"') ||
    video.canPlayType('video/mp4; codecs="hvc1.1.6.L93.B0"');
  if (!hevc) {
    // Still try — some builds report "" but decode anyway — but prefer early fallback.
    throw new Error("Browser reports no HEVC support for <video>");
  }

  video.src = url;

  await waitForEvent(video, "loadeddata", 15000, `Video load timeout: ${url}`);
  if (video.videoWidth <= 0) {
    throw new Error(`Invalid video dimensions: ${url}`);
  }

  const w = video.videoWidth;
  const h = video.videoHeight;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  let frames;
  try {
    frames = await captureBySeek(video, ctx, w, h, numFrames, grayscale);
  } catch {
    frames = await captureByPlayback(video, ctx, w, h, numFrames, grayscale);
  }

  video.pause();
  video.removeAttribute("src");
  video.load();

  if (frames.length !== numFrames) {
    throw new Error(`Expected ${numFrames} frames, got ${frames.length}`);
  }
  return frames;
}

async function captureBySeek(video, ctx, w, h, numFrames, grayscale) {
  const duration = Number.isFinite(video.duration) && video.duration > 0
    ? video.duration
    : numFrames / 25;
  const frames = [];
  for (let i = 0; i < numFrames; i++) {
    // Sample mid-frame to avoid keyframe edge issues.
    const t = Math.min(
      Math.max(0, ((i + 0.5) / numFrames) * duration - 1e-4),
      Math.max(0, duration - 1e-3),
    );
    await seekReliable(video, t);
    ctx.drawImage(video, 0, 0, w, h);
    frames.push(readFrame(ctx, w, h, grayscale));
  }
  return frames;
}

async function captureByPlayback(video, ctx, w, h, numFrames, grayscale) {
  const frames = [];
  video.currentTime = 0;
  await seekReliable(video, 0).catch(() => {});

  await new Promise((resolve, reject) => {
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };
    const fail = (err) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err instanceof Error ? err : new Error(String(err)));
    };
    const onFrame = () => {
      if (frames.length >= numFrames) {
        video.pause();
        done();
        return;
      }
      ctx.drawImage(video, 0, 0, w, h);
      frames.push(readFrame(ctx, w, h, grayscale));
      if (frames.length >= numFrames) {
        video.pause();
        done();
        return;
      }
      if (typeof video.requestVideoFrameCallback === "function") {
        video.requestVideoFrameCallback(onFrame);
      }
    };
    const onTimeUpdate = () => {
      if (typeof video.requestVideoFrameCallback === "function") return;
      if (frames.length >= numFrames) return;
      ctx.drawImage(video, 0, 0, w, h);
      frames.push(readFrame(ctx, w, h, grayscale));
      if (frames.length >= numFrames) {
        video.pause();
        done();
      }
    };
    const onEnded = () => done();
    const cleanup = () => {
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("ended", onEnded);
      video.removeEventListener("error", onErr);
      clearTimeout(timer);
    };
    const onErr = () => fail(new Error("Video playback error"));
    const timer = setTimeout(
      () => fail(new Error("Playback capture timeout")),
      Math.max(20000, numFrames * 500),
    );

    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("ended", onEnded);
    video.addEventListener("error", onErr);

    if (typeof video.requestVideoFrameCallback === "function") {
      video.requestVideoFrameCallback(onFrame);
    }
    video.playbackRate = Math.min(4, Math.max(1, numFrames / 4));
    const playPromise = video.play();
    if (playPromise && typeof playPromise.then === "function") {
      playPromise.catch((e) => fail(e));
    }
  });

  // Pad last frame if we undershot slightly.
  while (frames.length < numFrames && frames.length > 0) {
    frames.push(frames[frames.length - 1]);
  }
  return frames.slice(0, numFrames);
}

function readFrame(ctx, w, h, grayscale) {
  const data = ctx.getImageData(0, 0, w, h).data;
  if (grayscale) {
    const gray = new Uint8Array(w * h);
    for (let p = 0; p < w * h; p++) gray[p] = data[p * 4];
    gray.width = w;
    gray.height = h;
    gray.gray = true;
    return gray;
  }
  const rgb = new Uint8Array(w * h * 3);
  for (let p = 0; p < w * h; p++) {
    rgb[p * 3] = data[p * 4];
    rgb[p * 3 + 1] = data[p * 4 + 1];
    rgb[p * 3 + 2] = data[p * 4 + 2];
  }
  rgb.width = w;
  rgb.height = h;
  rgb.gray = false;
  return rgb;
}

function seekReliable(video, time) {
  return new Promise((resolve, reject) => {
    if (
      video.readyState >= 2 &&
      Math.abs(video.currentTime - time) < 0.001
    ) {
      resolve();
      return;
    }

    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };
    const fail = (msg) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error(msg));
    };
    const onSeeked = () => finish();
    // Do NOT reject on 'error' during seek — Chrome often fires transient errors
    // for HEVC; we fall back at a higher level instead.
    const timer = setTimeout(() => {
      // Some engines never fire seeked; accept current frame after wait.
      if (video.readyState >= 2) finish();
      else fail("Video seek timeout");
    }, 2500);
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      clearTimeout(timer);
    };
    video.addEventListener("seeked", onSeeked);
    try {
      video.currentTime = time;
    } catch (e) {
      fail(e.message || "Video seek failed");
    }
  });
}

function waitForEvent(target, eventName, timeoutMs, timeoutMsg) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const onOk = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };
    const onErr = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error(timeoutMsg || `${eventName} failed`));
    };
    const timer = setTimeout(onErr, timeoutMs);
    const cleanup = () => {
      target.removeEventListener(eventName, onOk);
      target.removeEventListener("error", onErr);
      clearTimeout(timer);
    };
    target.addEventListener(eventName, onOk);
    target.addEventListener("error", onErr);
  });
}

const VENDOR_ROOT = new URL("../../vendor/ffmpeg/", import.meta.url);

async function getFfmpeg(log) {
  if (_ffmpeg) return _ffmpeg;
  if (_ffmpegLoading) return _ffmpegLoading;

  _ffmpegLoading = (async () => {
    log("Loading ffmpeg.wasm from vendor/ (first run ~30s)…");
    const [{ FFmpeg }, { toBlobURL, fetchFile }] = await Promise.all([
      import(new URL("esm/index.js", VENDOR_ROOT).href),
      import(new URL("esm/util_pkg/index.js", VENDOR_ROOT).href),
    ]);
    const ffmpeg = new FFmpeg();
    ffmpeg.on("log", ({ message }) => {
      if (message && /error|Error|failed/i.test(message)) log(message);
    });
    const coreJs = new URL("core/ffmpeg-core.js", VENDOR_ROOT).href;
    const coreWasm = new URL("core/ffmpeg-core.wasm", VENDOR_ROOT).href;
    const workerJs = new URL("esm/worker.js", VENDOR_ROOT).href;
    await ffmpeg.load({
      coreURL: await toBlobURL(coreJs, "text/javascript"),
      wasmURL: await toBlobURL(coreWasm, "application/wasm"),
      workerURL: await toBlobURL(workerJs, "text/javascript"),
    });
    ffmpeg._fetchFile = fetchFile;
    _ffmpeg = ffmpeg;
    log("ffmpeg.wasm ready.");
    return ffmpeg;
  })();

  try {
    return await _ffmpegLoading;
  } catch (e) {
    _ffmpegLoading = null;
    throw e;
  }
}

/**
 * Decode via ffmpeg.wasm to a PNG sequence, then inflate the PNG bytes.
 * The command must exit 0: ffmpeg.wasm aborts the wasm runtime on a failing
 * ffmpeg (`Aborted()`), which is what a bare `-i` size probe does.
 * Inflating the PNG keeps the stored 8-bit planes. Canvas readback does not.
 */
async function decodeWithFfmpeg(url, numFrames, opts) {
  const { grayscale, log } = opts;
  const ffmpeg = await getFfmpeg(log);
  const fetchFile = ffmpeg._fetchFile;
  const logs = [];
  const onLog = ({ message }) => {
    if (message) logs.push(message);
  };
  ffmpeg.on("log", onLog);

  const inputName = "input.mp4";
  const pattern = "frame_%03d.png";

  try {
    await ffmpeg.deleteFile(inputName);
  } catch {
    /* ignore */
  }

  let data;
  if (url.startsWith("blob:") || /^https?:/i.test(url)) {
    data = await fetchFile(url);
  } else {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}`);
    data = new Uint8Array(await res.arrayBuffer());
  }
  await ffmpeg.writeFile(inputName, data);

  const pixFmt = grayscale ? "gray" : "rgb24";
  const code = await ffmpeg.exec([
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    inputName,
    "-vsync",
    "0",
    "-frames:v",
    String(numFrames),
    "-pix_fmt",
    pixFmt,
    pattern,
  ]);
  if (code !== 0) {
    throw new Error(
      `ffmpeg exited with code ${code} for ${url}${logs.length ? `: ${logs[logs.length - 1]}` : ""}`,
    );
  }

  const frames = [];
  for (let i = 1; i <= numFrames; i++) {
    const name = `frame_${String(i).padStart(3, "0")}.png`;
    try {
      const buf = await ffmpeg.readFile(name);
      const png = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
      frames.push(await pngBytesToFrame(png, grayscale));
      await ffmpeg.deleteFile(name);
    } catch (e) {
      if (frames.length === 0) throw e;
      break;
    }
  }

  try {
    await ffmpeg.deleteFile(inputName);
  } catch {
    /* ignore */
  }
  ffmpeg.off?.("log", onLog);

  if (!frames.length) {
    throw new Error(`ffmpeg produced no frames for ${url}`);
  }
  if (frames.length < numFrames) {
    log(
      `Video has ${frames.length} frame(s), metadata asked for ${numFrames}. Repeating the last frame.`,
    );
    while (frames.length < numFrames) {
      const last = frames[frames.length - 1];
      const copy = new Uint8Array(last);
      copy.width = last.width;
      copy.height = last.height;
      if (last.gray) copy.gray = true;
      frames.push(copy);
    }
  }
  return frames.slice(0, numFrames);
}

async function pngBytesToFrame(png, grayscale) {
  try {
    return await decodePngBytes(png, grayscale);
  } catch {
    return pngToFrame(png, grayscale);
  }
}

/** Non-interlaced 8-bit PNG. Reads IDAT directly so bytes are not color-managed. */
export async function decodePngBytes(png, grayscale) {
  if (png.length < 8 || png[0] !== 0x89 || png[1] !== 0x50) {
    throw new Error("ffmpeg frame is not a PNG");
  }
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = -1;
  let interlace = 0;
  const idats = [];
  while (offset + 8 <= png.length) {
    const len = view.getUint32(offset);
    const type = String.fromCharCode(
      png[offset + 4],
      png[offset + 5],
      png[offset + 6],
      png[offset + 7],
    );
    const start = offset + 8;
    if (type === "IHDR") {
      width = view.getUint32(start);
      height = view.getUint32(start + 4);
      bitDepth = png[start + 8];
      colorType = png[start + 9];
      interlace = png[start + 12];
    } else if (type === "IDAT") {
      idats.push(png.slice(start, start + len));
    } else if (type === "IEND") {
      break;
    }
    offset = start + len + 4;
  }
  if (!width || !height || bitDepth !== 8 || interlace !== 0) {
    throw new Error(`Unsupported PNG ${width}×${height} depth ${bitDepth} interlace ${interlace}`);
  }
  const channels = pngChannels(colorType);
  const inflated = await inflateZlib(idats);
  const raw = unfilterPng(inflated, width, height, channels);
  let pixels;
  let gray = false;
  if (grayscale || colorType === 0) {
    pixels = new Uint8Array(width * height);
    const step = channels;
    for (let i = 0; i < pixels.length; i++) pixels[i] = raw[i * step];
    gray = true;
  } else if (channels === 3) {
    pixels = raw;
  } else {
    pixels = new Uint8Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
      pixels[i * 3] = raw[i * channels];
      pixels[i * 3 + 1] = raw[i * channels + 1];
      pixels[i * 3 + 2] = raw[i * channels + 2];
    }
  }
  pixels.width = width;
  pixels.height = height;
  pixels.gray = gray;
  return pixels;
}

function pngChannels(colorType) {
  if (colorType === 0) return 1;
  if (colorType === 2) return 3;
  if (colorType === 4) return 2;
  if (colorType === 6) return 4;
  throw new Error(`Unsupported PNG color type ${colorType}`);
}

async function inflateZlib(parts) {
  const blob = new Blob(parts);
  const stream = blob.stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function unfilterPng(inflated, width, height, channels) {
  const stride = width * channels;
  const expected = height * (stride + 1);
  if (inflated.length < expected) {
    throw new Error(`PNG inflate is ${inflated.length} bytes, expected ${expected}`);
  }
  const out = new Uint8Array(height * stride);
  let src = 0;
  for (let y = 0; y < height; y++) {
    const filter = inflated[src++];
    const row = y * stride;
    const prev = row - stride;
    for (let x = 0; x < stride; x++) {
      const left = x >= channels ? out[row + x - channels] : 0;
      const up = y > 0 ? out[prev + x] : 0;
      const upLeft = y > 0 && x >= channels ? out[prev + x - channels] : 0;
      const raw = inflated[src++];
      let value = raw;
      if (filter === 1) value = (raw + left) & 255;
      else if (filter === 2) value = (raw + up) & 255;
      else if (filter === 3) value = (raw + ((left + up) >> 1)) & 255;
      else if (filter === 4) value = (raw + paeth(left, up, upLeft)) & 255;
      else if (filter !== 0) throw new Error(`PNG filter ${filter}`);
      out[row + x] = value;
    }
  }
  return out;
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

async function pngToFrame(pngBytes, grayscale) {
  const blob = new Blob([pngBytes], { type: "image/png" });
  const bmp = await createImageBitmap(blob);
  const w = bmp.width;
  const h = bmp.height;
  const canvas =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(w, h)
      : document.createElement("canvas");
  if (!("getContext" in canvas) || canvas.width !== w) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;
  bmp.close?.();
  const pixels = readFramePixels(data, w, h, grayscale);
  pixels.width = w;
  pixels.height = h;
  pixels.gray = grayscale;
  return pixels;
}

function readFramePixels(data, w, h, grayscale) {
  if (grayscale) {
    const gray = new Uint8Array(w * h);
    for (let p = 0; p < w * h; p++) gray[p] = data[p * 4];
    return gray;
  }
  const rgb = new Uint8Array(w * h * 3);
  for (let p = 0; p < w * h; p++) {
    rgb[p * 3] = data[p * 4];
    rgb[p * 3 + 1] = data[p * 4 + 1];
    rgb[p * 3 + 2] = data[p * 4 + 2];
  }
  return rgb;
}

export async function hevcLikelySupported() {
  const v = document.createElement("video");
  const tag =
    v.canPlayType('video/mp4; codecs="hev1.1.6.L93.B0"') ||
    v.canPlayType('video/mp4; codecs="hvc1.1.6.L93.B0"');
  return Boolean(tag);
}
