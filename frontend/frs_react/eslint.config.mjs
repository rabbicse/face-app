import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const config = [
    ...nextVitals,
    ...nextTs,
    { ignores: ["legacy/**", ".next/**", "public/**", "node_modules/**", "next-env.d.ts"] },
];

export default config;
