# AquaCore

Refractive multi-camera geometry foundation for the Aqua ecosystem. Provides shared PyTorch implementations of Snell's law refraction, camera models, triangulation, pose transforms, calibration loading, and synchronized multi-camera I/O — consumed by [AquaCal](https://github.com/tlancaster6/AquaCal), [AquaMVS](https://github.com/tlancaster6/AquaMVS), and AquaPose.

## Installation

```bash
pip install aquacore
```

## Quick Start

```python
import torch
from aquacore import CameraIntrinsics, CameraExtrinsics, InterfaceParams
from aquacore import create_camera, snells_law_3d, triangulate_rays

# Load calibration from AquaCal JSON
from aquacore import load_calibration_data

calib = load_calibration_data("path/to/aquacal.json")
```

## Development

```bash
# Set up the development environment
pip install hatch
hatch env create
hatch run pre-commit install
hatch run pre-commit install --hook-type pre-push

# Run tests, lint, and type check
hatch run test
hatch run lint
hatch run typecheck
```

See [Contributing](docs/contributing.md) for full development guidelines.

## Documentation

Full documentation is available at [aquacore.readthedocs.io](https://aquacore.readthedocs.io).

## License

[MIT](LICENSE)
