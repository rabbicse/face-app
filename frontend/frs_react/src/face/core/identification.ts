import { dot, fuseEmbeddings } from "./embedding";

export interface AggregatorOptions {
    /** number of consecutive good frames fused into one query */
    windowSize: number;
    /** frames older than this are discarded */
    maxAgeMs: number;
    /** every frame in the window must have at least this cosine to the others' mean */
    minAgreement: number;
}

export const DEFAULT_AGGREGATOR_OPTIONS: AggregatorOptions = {
    windowSize: 3,
    maxAgeMs: 1500,
    minAgreement: 0.6,
};

/**
 * Fuses embeddings of several consecutive frames before querying the server: averaging reduces the
 * per-frame noise (expression, motion blur, landmark jitter) and the agreement check rejects windows
 * mixing two different faces. No head pose requirement - identification accepts any usable frame.
 */
export class IdentificationAggregator {
    private window: { embedding: Float32Array; timestamp: number }[] = [];

    constructor(private readonly options: AggregatorOptions = DEFAULT_AGGREGATOR_OPTIONS) { }

    get size(): number {
        return this.window.length;
    }

    /** @return the fused query embedding once the window is full and consistent, otherwise null */
    add(embedding: Float32Array, timestamp: number): Float32Array | null {
        this.window = this.window.filter(f => timestamp - f.timestamp <= this.options.maxAgeMs);
        this.window.push({ embedding, timestamp });
        if (this.window.length > this.options.windowSize) {
            this.window.shift();
        }
        if (this.window.length < this.options.windowSize) {
            return null;
        }

        const vectors = this.window.map(f => f.embedding);
        for (let i = 0; i < vectors.length; i++) {
            const others = fuseEmbeddings(vectors.filter((_, j) => j !== i));
            if (dot(vectors[i], others) < this.options.minAgreement) {
                // drop the oldest frame and wait for more
                this.window.shift();
                return null;
            }
        }
        this.window = [];
        return fuseEmbeddings(vectors);
    }

    reset(): void {
        this.window = [];
    }
}
