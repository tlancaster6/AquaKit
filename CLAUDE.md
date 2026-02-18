# AquaCore

Shared PyTorch geometry library for the Aqua ecosystem (AquaCal, AquaMVS, AquaPose). Implements refractive multi-camera geometry: Snell's law, camera models, triangulation, transforms, calibration loading, and synchronized I/O.

## Quick Start

```bash
pip install hatch
hatch env create
hatch run pre-commit install
hatch run pre-commit install --hook-type pre-push
```

## Commands

```bash
hatch run test                        # run tests (excludes slow)
hatch run test-all                    # run all tests including slow
hatch run lint                        # ruff lint
hatch run format                      # ruff format
hatch run typecheck                   # basedpyright
hatch run check                       # lint + typecheck
hatch run docs:build                  # build sphinx docs
hatch run pre-commit run --all-files  # all pre-commit hooks
```

## Architecture

```
src/aquacore/
├── types.py              # Foundation: shared types (no math, no deps)
├── interface.py          # Air-water plane model
├── camera.py             # Camera models + create_camera factory
├── transforms.py         # Rotation/pose utilities
├── refraction.py         # Snell's law + ray tracing
├── triangulation.py      # Batched ray intersection
├── calibration.py        # AquaCal JSON loader
├── undistortion.py       # Remap operations
├── projection/
│   ├── protocol.py       # ProjectionModel Protocol
│   └── refractive.py     # RefractiveProjectionModel (Newton-Raphson)
└── io/
    ├── frameset.py       # FrameSet Protocol
    ├── video.py          # VideoSet
    └── images.py         # ImageSet
```

Dependency flow is strictly top-down: types → math → projection → calibration → I/O.

## Domain Conventions

- **Tensor library**: PyTorch for all math. NumPy only at serialization boundaries (JSON, OpenCV).
- **Device**: follow input tensor device. No `device` param, no `.cuda()` calls.
- **Coordinate system**: world origin at reference camera optical center (+X right, +Y forward, +Z down into water). Camera frame: OpenCV convention. Extrinsics: `p_cam = R @ p_world + t`.
- **Interface normal**: `[0, 0, -1]` (upward from water surface).
- **Source references**: AquaCal at `../AquaCal/src/aquacal/`, AquaMVS at `../AquaMVS/src/aquamvs/`.
