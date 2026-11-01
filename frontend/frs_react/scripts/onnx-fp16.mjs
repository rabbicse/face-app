/**
 * Dependency free "fp16 weight storage" for ONNX models (mirror of backend/frs/tools/compress_onnx.py
 * --mode fp16): every float32 Conv / Gemm weight initializer is stored as float16 and a Cast(to=float) node
 * is prepended. ONNX Runtime folds the casts when the session is created, so inference is unchanged while
 * the download halves (w600k_mbf: 13.6 MB -> 6.9 MB, cosine to fp32 >= 0.99998 on 636 faces).
 *
 * Only the protobuf fields involved are decoded; everything else is copied byte for byte.
 */

// ---- protobuf wire format ----
function readVarint(buf, pos) {
    let result = 0n, shift = 0n;
    for (;;) {
        const b = buf[pos++];
        result |= BigInt(b & 0x7f) << shift;
        if (!(b & 0x80)) return [result, pos];
        shift += 7n;
    }
}

/** Splits a message into [{field, wire, start, end, value}] (value = payload for length-delimited fields). */
export function parseFields(buf) {
    const fields = [];
    let pos = 0;
    while (pos < buf.length) {
        const start = pos;
        const [key, p1] = readVarint(buf, pos);
        const field = Number(key >> 3n), wire = Number(key & 7n);
        let value, end;
        if (wire === 0) {
            const [v, p2] = readVarint(buf, p1);
            value = v; end = p2;
        } else if (wire === 2) {
            const [len, p2] = readVarint(buf, p1);
            end = p2 + Number(len);
            value = buf.subarray(p2, end);
        } else if (wire === 5) {
            end = p1 + 4; value = buf.subarray(p1, end);
        } else if (wire === 1) {
            end = p1 + 8; value = buf.subarray(p1, end);
        } else {
            throw new Error(`unsupported wire type ${wire}`);
        }
        fields.push({ field, wire, start, end, value });
        pos = end;
    }
    return fields;
}

function varint(n) {
    let v = BigInt(n);
    const out = [];
    do {
        let b = Number(v & 0x7fn);
        v >>= 7n;
        if (v) b |= 0x80;
        out.push(b);
    } while (v);
    return Uint8Array.from(out);
}

const concat = parts => {
    const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
};
const lenField = (field, payload) => concat([varint((field << 3) | 2), varint(payload.length), payload]);
const varintField = (field, n) => concat([varint(field << 3), varint(n)]);
const strField = (field, s) => lenField(field, new TextEncoder().encode(s));
const str = bytes => new TextDecoder().decode(bytes);

// ---- float32 -> float16 (round to nearest even) ----
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function toHalf(value) {
    f32[0] = value;
    const x = u32[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / nan
    let e = exp - 127 + 15;
    if (e >= 0x1f) return sign | 0x7c00; // overflow -> inf
    if (e <= 0) { // subnormal half
        if (e < -10) return sign;
        mant |= 0x800000;
        const shift = 14 - e;
        let h = mant >>> shift;
        const rem = mant & ((1 << shift) - 1), half = 1 << (shift - 1);
        if (rem > half || (rem === half && (h & 1))) h++;
        return sign | h;
    }
    let h = (e << 10) | (mant >>> 13);
    const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++; // may carry into the exponent, which is correct
    return sign | h;
}

// ONNX field numbers
const MODEL_GRAPH = 7;
const GRAPH_NODE = 1, GRAPH_INITIALIZER = 5;
const NODE_INPUT = 1, NODE_OP_TYPE = 4;
const TENSOR_DIMS = 1, TENSOR_DATA_TYPE = 2, TENSOR_FLOAT_DATA = 4, TENSOR_NAME = 8, TENSOR_RAW_DATA = 9;
const FLOAT = 1, FLOAT16 = 10;
const ATTR_NAME = 1, ATTR_I = 3, ATTR_TYPE = 20, ATTR_TYPE_INT = 2;
const NODE_OUTPUT = 2, NODE_NAME = 3, NODE_ATTRIBUTE = 5;

/** @returns {{model: Uint8Array, converted: number}} */
export function convertWeightsToFp16(model, minElements = 1024) {
    const top = parseFields(model);
    const graphField = top.find(f => f.field === MODEL_GRAPH);
    if (!graphField) throw new Error("not an ONNX model (no graph)");
    const graph = parseFields(graphField.value);

    const weights = new Set();
    for (const f of graph.filter(f => f.field === GRAPH_NODE)) {
        const node = parseFields(f.value);
        const op = str(node.find(n => n.field === NODE_OP_TYPE)?.value ?? new Uint8Array());
        const inputs = node.filter(n => n.field === NODE_INPUT).map(n => str(n.value));
        if ((op === "Conv" || op === "Gemm") && inputs.length > 1) weights.add(inputs[1]);
    }

    const casts = [];
    const newGraph = [];
    let converted = 0;
    for (const f of graph) {
        if (f.field !== GRAPH_INITIALIZER) {
            newGraph.push(f);
            continue;
        }
        const tensor = parseFields(f.value);
        const name = str(tensor.find(t => t.field === TENSOR_NAME)?.value ?? new Uint8Array());
        const type = Number(tensor.find(t => t.field === TENSOR_DATA_TYPE)?.value ?? 0);
        const raw = tensor.find(t => t.field === TENSOR_RAW_DATA)?.value;
        if (!weights.has(name) || type !== FLOAT || !raw || raw.length / 4 < minElements
            || tensor.some(t => t.field === TENSOR_FLOAT_DATA)) {
            newGraph.push(f);
            continue;
        }
        const floats = new Float32Array(raw.slice().buffer);
        const halves = new Uint16Array(floats.length);
        for (let i = 0; i < floats.length; i++) halves[i] = toHalf(floats[i]);
        const rebuilt = tensor.map(t => {
            if (t.field === TENSOR_DATA_TYPE) return varintField(TENSOR_DATA_TYPE, FLOAT16);
            if (t.field === TENSOR_NAME) return strField(TENSOR_NAME, `${name}_fp16`);
            if (t.field === TENSOR_RAW_DATA) return lenField(TENSOR_RAW_DATA, new Uint8Array(halves.buffer));
            return f.value.subarray(t.start, t.end);
        });
        newGraph.push({ bytes: lenField(GRAPH_INITIALIZER, concat(rebuilt)) });
        const attr = concat([strField(ATTR_NAME, "to"), varintField(ATTR_I, FLOAT), varintField(ATTR_TYPE, ATTR_TYPE_INT)]);
        casts.push(lenField(GRAPH_NODE, concat([
            strField(NODE_INPUT, `${name}_fp16`), strField(NODE_OUTPUT, name), strField(NODE_NAME, `${name}_cast`),
            strField(NODE_OP_TYPE, "Cast"), lenField(NODE_ATTRIBUTE, attr),
        ])));
        converted++;
    }

    // casts first keeps the node list topologically sorted
    const graphBytes = concat([...casts, ...newGraph.map(f => f.bytes ?? graphField.value.subarray(f.start, f.end))]);
    const out = concat(top.map(f => (f === graphField ? lenField(MODEL_GRAPH, graphBytes) : model.subarray(f.start, f.end))));
    return { model: out, converted };
}
