import numpy as np

from instantsfm.processors.global_positioning import _camera_calibration_weight
from instantsfm.scene.defs import CameraModelId, Cameras


def test_camera_calibration_weight_distinguishes_prior_and_refined():
    cameras = Cameras(num_cameras=3)
    for cam_id in range(3):
        cameras.widths[cam_id] = 1600
        cameras.heights[cam_id] = 1200
        cameras.set_params(
            cam_id,
            np.array([800.0, 800.0, 600.0], dtype=np.float64),
            CameraModelId.SIMPLE_PINHOLE,
        )

    prior_cam = cameras[0]
    prior_cam.has_prior_focal_length = True

    refined_cam = cameras[1]
    refined_cam.has_refined_focal_length = True
    refined_cam.refined_focal_confidence = 0.5

    unknown_cam = cameras[2]

    assert _camera_calibration_weight(prior_cam, 0.75, 0.5) == 1.0
    assert _camera_calibration_weight(refined_cam, 0.75, 0.5) == 0.625
    assert _camera_calibration_weight(unknown_cam, 0.75, 0.5) == 0.5

    assert cameras[1].has_refined_focal_length
