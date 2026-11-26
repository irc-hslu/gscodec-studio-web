/**
 * Load GSCodec compression assets from a local folder chosen in the browser
 * (<input type="file" webkitdirectory> or File System Access API).
 *
 * Browsers cannot read arbitrary disk paths via fetch/file://; this maps
 * selected File objects to blob: URLs so MP4 decode and JSON reads work offline.
 *
 * @param {Iterable<File>} files FileList or array of File from a directory picker
 * @param {{ compressionSubdir?: string }} [options]
 *   If the selection is a parent folder that contains `compression/`, set
 *   compressionSubdir to "compression" (default auto-detect).
 */
export function createFileMapSource(files, options = {}) {
  /** @type {Map<string, File>} basename -> File (last wins if duplicates) */
  const byBase = new Map();
  /** @type {Map<string, File>} relative posix path -> File */
  const byRel = new Map();

  for (const file of files) {
    const rel = normalizeRel(file.webkitRelativePath || file.name);
    byRel.set(rel, file);
    byBase.set(basename(rel), file);
  }

  const compressionRoot = resolveCompressionRoot(byRel, options.compressionSubdir);
  const objectUrls = new Map();

  function getFile(fileName) {
    const key = compressionRoot
      ? `${compressionRoot}/${fileName}`.replace(/\/+/g, "/")
      : fileName;
    let file = byRel.get(key);
    if (!file) file = byBase.get(basename(fileName));
    if (!file) {
      throw new Error(
        `Local folder is missing "${fileName}"` +
          (compressionRoot ? ` (looked under ${compressionRoot}/)` : ""),
      );
    }
    return file;
  }

  function resolveAsset(fileName) {
    const file = getFile(fileName);
    let url = objectUrls.get(file);
    if (!url) {
      url = URL.createObjectURL(file);
      objectUrls.set(file, url);
    }
    return url;
  }

  return {
    compressionDir: compressionRoot || "(local)",
    resolveAsset,
    async readText(fileName) {
      return getFile(fileName).text();
    },
    async readArrayBuffer(fileName) {
      return getFile(fileName).arrayBuffer();
    },
    /** Release blob URLs when the viewer is disposed. */
    dispose() {
      for (const url of objectUrls.values()) URL.revokeObjectURL(url);
      objectUrls.clear();
    },
    /** Basenames present under the compression root (debug). */
    listBasenames() {
      const out = [];
      const prefix = compressionRoot ? compressionRoot + "/" : "";
      for (const rel of byRel.keys()) {
        if (!prefix || rel.startsWith(prefix)) {
          out.push(rel.slice(prefix.length));
        }
      }
      return out.sort();
    },
  };
}

function normalizeRel(path) {
  return String(path).replace(/\\/g, "/").replace(/^\.\//, "");
}

function basename(path) {
  const parts = normalizeRel(path).split("/");
  return parts[parts.length - 1];
}

/**
 * Prefer a directory that contains unity_manifest.json or gop0_meta.json.
 * @param {Map<string, File>} byRel
 * @param {string | undefined} forcedSubdir
 */
function resolveCompressionRoot(byRel, forcedSubdir) {
  if (forcedSubdir) {
    return normalizeRel(forcedSubdir).replace(/\/$/, "");
  }

  const markers = ["unity_manifest.json", "gop0_meta.json"];
  for (const rel of byRel.keys()) {
    const name = basename(rel);
    if (!markers.includes(name)) continue;
    const dir = rel.includes("/") ? rel.slice(0, rel.lastIndexOf("/")) : "";
    return dir;
  }

  // Nested: …/something/compression/unity_manifest.json already covered above.
  for (const rel of byRel.keys()) {
    if (rel.endsWith("/compression/unity_manifest.json") || rel === "compression/unity_manifest.json") {
      return rel.slice(0, -"/unity_manifest.json".length);
    }
  }
  return "";
}
