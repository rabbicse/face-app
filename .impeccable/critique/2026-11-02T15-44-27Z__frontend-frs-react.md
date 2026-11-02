---
target: critique ui/ux and polish (frs_react web app)
total_score: 14
max_score: 40
na_heuristics: 
p0_count: 2
p1_count: 2
target_identity: "file:/Users/mehmet/Projects/github/face-app/frontend/frs_react"
timestamp: 2026-11-02T15-44-27Z
slug: frontend-frs-react
---
Method: dual-agent (isolated design review A and detector/browser pass B). Fake camera only; success, failure, camera-denied and server-error states assessed from source.

## Design Health Score: 14/40 (Poor)
1 Status 2 | 2 Real world 1 | 3 Control 1 | 4 Consistency 2 | 5 Error prevention 1 | 6 Recognition 2 | 7 Flexibility 1 | 8 Aesthetic 2 | 9 Error recovery 1 | 10 Help 1

## Design Specificity
Interchangeable stock shadcn. Detector: 0 findings (regex only, cannot see classes missing from compiled CSS). Live overlay skipped for camera privacy.

## Working
Guided auto-capture; privacy architecture claim; pipeline inspector.

## Priority Issues
- [P0] Camera overlay, mirroring, click-through broken: tailwind.config.ts:6-10 does not scan src/face. Fix: add ./src/face/**/*.{ts,tsx} to content, recheck pose hints against mirrored preview. Command: polish.
- [P0] Login outcome fake: /dashboard is shadcn demo with no auth, ?name= spoofs greeting, mobile shows 404 image; LoginForm.tsx:47 button does nothing; RegistrationFrom.tsx:23-28 discards password. Fix: real signed-in page; remove password fields or label demo. Command: shape.
- [P1] No consent before camera/biometric enrolment: useCamera.ts:18 fires on load; name/email in URL (RegistrationFrom.tsx:28-29). Fix: consent screen gated by enabled param. Command: onboard.
- [P1] Camera failures are dead ends with raw errors; registration link only after 5 misses; no load progress. Command: harden.
- [P2] Face login undiscoverable/inaccessible: 24px icon, focus ring 2.56:1, input border 1.23:1, no main/h1 on camera pages, shared titles, no reduced-motion, no blink alternative. Command: audit.

## Persona Red Flags
Jordan: four equal home choices, jargon, dropped password, revenue dashboard. Sam: live-region chatter, no spoken pose guidance, Enter does not submit, faint focus. Casey: nine captures with no pause, status below video, blank dashboard.

## Minor Observations
"below to registration" copy and RegistrationFrom filename typo; demo metadata on /dashboard; .dark tokens unused; enrolment has no liveness check; Compare ignores margin.

## Questions
Demo or real sign-in product? Why identify without an intent signal? What can only exist here at recognition? Who cannot blink or turn their head?
