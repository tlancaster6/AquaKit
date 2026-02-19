"""AquaCore: refractive multi-camera geometry foundation for the Aqua ecosystem."""

from importlib.metadata import PackageNotFoundError, version

from .calibration import CalibrationData, CameraData, load_calibration_data
from .camera import create_camera
from .interface import ray_plane_intersection
from .io import FrameSet, ImageSet, VideoSet, create_frameset
from .projection import (
    ProjectionModel,
    RefractiveProjectionModel,
    back_project_multi,
    project_multi,
)
from .refraction import (
    refractive_back_project,
    refractive_project,
    snells_law_3d,
    trace_ray_air_to_water,
    trace_ray_water_to_air,
)
from .transforms import (
    camera_center,
    compose_poses,
    invert_pose,
    matrix_to_rvec,
    rvec_to_matrix,
)
from .triangulation import point_to_ray_distance, triangulate_rays
from .types import (
    INTERFACE_NORMAL,
    CameraExtrinsics,
    CameraIntrinsics,
    InterfaceParams,
    Mat3,
    Vec2,
    Vec3,
)
from .undistortion import compute_undistortion_maps, undistort_image

__all__ = [
    "INTERFACE_NORMAL",
    "CalibrationData",
    "CameraData",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "FrameSet",
    "ImageSet",
    "InterfaceParams",
    "Mat3",
    "ProjectionModel",
    "RefractiveProjectionModel",
    "Vec2",
    "Vec3",
    "VideoSet",
    "back_project_multi",
    "camera_center",
    "compose_poses",
    "compute_undistortion_maps",
    "create_camera",
    "create_frameset",
    "invert_pose",
    "load_calibration_data",
    "matrix_to_rvec",
    "point_to_ray_distance",
    "project_multi",
    "ray_plane_intersection",
    "refractive_back_project",
    "refractive_project",
    "rvec_to_matrix",
    "snells_law_3d",
    "trace_ray_air_to_water",
    "trace_ray_water_to_air",
    "triangulate_rays",
    "undistort_image",
]

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
