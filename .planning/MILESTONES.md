# Milestones

## v1.0 MVP (Shipped: 2026-02-18)

**Phases completed:** 6 phases, 13 plans
**Lines of code:** 5,561 Python (37 files)
**Timeline:** 3 days (2026-02-16 → 2026-02-18)

**Key accomplishments:**
- Pure-PyTorch geometry foundation: types, camera models, transforms, Snell's law, and triangulation with device-parametrized tests
- Refractive ProjectionModel protocol with Newton-Raphson back-projection and batched operations
- AquaCal JSON calibration loader and cv2.remap undistortion with pinhole/fisheye dispatch
- Synchronized multi-camera I/O: FrameSet protocol, ImageSet, VideoSet, and create_frameset factory
- CI pipeline (ruff + basedpyright standard + pytest), PyPI publishing, and AquaCal/AquaMVS rewiring guide

**Delivered:** Shared PyTorch geometry library for the Aqua ecosystem — refractive multi-camera geometry, calibration loading, synchronized I/O — ready for AquaPose development.

---

