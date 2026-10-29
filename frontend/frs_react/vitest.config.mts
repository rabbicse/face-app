import path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
    resolve: {
        alias: { "@": path.resolve(__dirname, "src") },
    },
    test: {
        environment: "node",
        // the legacy *_test.ts files under src/dnn are jasmine specs copied from tfjs-models
        include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    },
});
