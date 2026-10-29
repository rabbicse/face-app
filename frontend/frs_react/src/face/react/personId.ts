/**
 * Demo only: derives a stable numeric person id from the e-mail (FNV-1a, 31 bit) so registering the same
 * e-mail again replaces the old enrolment. A real application would use the id from its user database.
 */
export function personIdFromEmail(email: string): number {
    let h = 0x811c9dc5;
    for (const ch of email.trim().toLowerCase()) {
        h ^= ch.codePointAt(0)!;
        h = Math.imul(h, 0x01000193);
    }
    return (h >>> 1) || 1;
}
