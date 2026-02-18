# Phase 1: Foundation and Physics Math - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

All geometry primitives — types (CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3), camera models (pinhole, fisheye), transforms, Snell's law refraction, and ray triangulation — implemented in PyTorch, device-agnostic, and verified against known values. This phase is the dependency root for all subsequent layers.

</domain>

<decisions>
## Implementation Decisions

### Guiding Principle: Unify Existing Implementations
AquaCore v1 extracts and unifies existing behavior from AquaCal and AquaMVS. The default for any gray area is to match existing implementations unless there is a clear reason to diverge (broken, inconsistent, or undocumented behavior). Decisions are made case-by-case during planning, informed by researcher examination of both codebases.

**Source repositories (same machine):**
- AquaCal: `C:\Users\tucke\PycharmProjects\AquaCal`
- AquaMVS: `C:\Users\tucke\PycharmProjects\AquaMVS`

**Pre-mapped reference docs (read these FIRST before exploring raw source):**
- `.planning/research/aquacal-map.md` — AquaCal module layout, types, functions, conventions
- `.planning/research/aquamvs-map.md` — AquaMVS module layout, types, functions, conventions
- `.planning/research/shared-patterns.md` — Cross-repo comparison, inconsistencies, and resolution decisions

The researcher should read these reference docs first, then dive into raw source only for details not covered.

### Camera Model API
- Fisheye distortion model: OpenCV fisheye (k1-k4 equidistant model)
- Camera construction: `create_camera()` factory is the only public construction API; underlying classes are internal
- Camera class design (separate vs unified, intrinsics bundling): Claude's discretion — pick what works best with the Phase 2 projection protocol

### Refraction Model
- Interface model: Simplified air-to-water (single interface with one refractive index ratio), not the full 3-layer air-glass-water chain
- Total internal reflection handling: Match existing AquaCal/AquaMVS behavior — researcher to extract current approach
- Edge case behavior: Match existing implementations — researcher to document what both repos currently do

### Error & Validation Behavior
- Validation strategy: Validate at boundaries (factory functions like `create_camera()` validate and raise); internal math functions trust their inputs for performance
- Exception style: Claude's discretion — pick what makes sense for this project's size
- Input normalization (e.g., auto-normalize direction vectors): Match existing AquaCal/AquaMVS behavior — researcher to check
- Device mismatch: Raise on mismatch with clear error message — no silent tensor moves between devices

### Claude's Discretion
- Camera class architecture (separate classes vs single class with model parameter)
- How intrinsics/extrinsics relate to camera objects
- Exception hierarchy (standard Python vs custom)
- All decisions not explicitly locked above — guided by existing AquaCal/AquaMVS patterns

</decisions>

<specifics>
## Specific Ideas

- AquaCore v1 is a unification project, not a greenfield design. The researcher should examine AquaCal and AquaMVS source to understand existing behavior before any design decisions are finalized.
- Eventual consumer migration should be smooth — API choices should minimize the delta between current AquaCal/AquaMVS usage and future AquaCore usage.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation-and-physics-math*
*Context gathered: 2026-02-18*
