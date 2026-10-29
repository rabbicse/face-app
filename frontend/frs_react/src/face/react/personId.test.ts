import { describe, expect, it } from "vitest";
import { personIdFromEmail } from "./personId";

describe("personIdFromEmail", () => {
    it("is stable, case insensitive and a positive 31 bit integer", () => {
        const id = personIdFromEmail("Someone@Example.com ");
        expect(id).toBe(personIdFromEmail("someone@example.com"));
        expect(id).toBeGreaterThan(0);
        expect(id).toBeLessThan(2 ** 31);
        expect(personIdFromEmail("other@example.com")).not.toBe(id);
    });
});
