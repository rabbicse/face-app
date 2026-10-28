/**
 * Fetches model files once and keeps them in the Cache Storage API, so the 16 MB of
 * models are downloaded on the first visit only (also works offline afterwards).
 * Works in windows and workers.
 */
const CACHE_NAME = "face-models-v1";

export type Progress = (loaded: number, total: number) => void;

export async function fetchModel(url: string, onProgress?: Progress): Promise<ArrayBuffer> {
    const abs = new URL(url, self.location.href).toString();
    let cache: Cache | null = null;
    try {
        cache = await caches.open(CACHE_NAME);
        const hit = await cache.match(abs);
        if (hit) {
            const buf = await hit.arrayBuffer();
            onProgress?.(buf.byteLength, buf.byteLength);
            return buf;
        }
    } catch {
        cache = null; // Cache API unavailable (e.g. http on a LAN IP) – just fetch
    }

    const res = await fetch(abs);
    if (!res.ok) throw new Error(`Failed to load ${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get("content-length") ?? 0);
    let buf: ArrayBuffer;
    if (res.body && onProgress) {
        const reader = res.body.getReader();
        const chunks: Uint8Array[] = [];
        let loaded = 0;
        for (; ;) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.byteLength;
            onProgress(loaded, total || loaded);
        }
        const all = new Uint8Array(loaded);
        let o = 0;
        for (const c of chunks) { all.set(c, o); o += c.byteLength; }
        buf = all.buffer;
    } else {
        buf = await res.arrayBuffer();
    }
    if (cache) {
        try {
            await cache.put(abs, new Response(buf.slice(0), { headers: { "content-type": "application/octet-stream" } }));
        } catch { /* quota exceeded etc. – non fatal */ }
    }
    return buf;
}

export async function clearModelCache(): Promise<void> {
    try { await caches.delete(CACHE_NAME); } catch { /* ignore */ }
}
