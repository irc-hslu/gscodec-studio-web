/**
 * Load GSCodec compression assets over HTTP (static file server or CDN).
 * @param {string} baseUrl Directory containing unity_manifest.json and gop*_meta.json
 */
export function createUrlSource(baseUrl) {
  const root = baseUrl.replace(/\/?$/, "/");
  return {
    compressionDir: root,
    resolveAsset(fileName) {
      return root + fileName;
    },
    async readText(fileName) {
      const res = await fetch(root + fileName);
      if (!res.ok) throw new Error(`Failed to fetch ${root}${fileName}: ${res.status}`);
      return res.text();
    },
    async readArrayBuffer(fileName) {
      const res = await fetch(root + fileName);
      if (!res.ok) throw new Error(`Failed to fetch ${root}${fileName}: ${res.status}`);
      return res.arrayBuffer();
    },
  };
}
