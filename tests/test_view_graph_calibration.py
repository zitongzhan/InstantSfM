import numpy as np

from instantsfm.processors.view_graph_calibration import (
    _accept_calibrated_focals,
    _apply_camera_calibration,
    _apply_pair_acceptance,
    _collect_valid_pairs,
    _cross_validate_prior_focal_lengths,
    _essential_from_fundamental,
)
from instantsfm.scene.defs import (
    CameraModelId,
    Cameras,
    ConfigurationType,
    ImagePair,
    Images,
    ViewGraph,
)


def _make_cameras(num_cameras, prior_flags):
    cameras = Cameras(num_cameras=num_cameras)
    for cam_id, has_prior in enumerate(prior_flags):
        cameras.widths[cam_id] = 1600
        cameras.heights[cam_id] = 1200
        cameras.has_prior_focal_length[cam_id] = has_prior
        cameras.set_params(
            cam_id,
            np.array([800.0 + 25.0 * cam_id, 800.0, 600.0], dtype=np.float64),
            CameraModelId.SIMPLE_PINHOLE,
        )
    return cameras


def _make_images(cam_ids):
    images = Images(num_images=len(cam_ids))
    for image_id, cam_id in enumerate(cam_ids):
        images.cam_ids[image_id] = cam_id
    return images


def test_cross_validate_prior_focal_lengths_upgrades_supported_pairs():
    cameras = _make_cameras(4, [True, True, True, True])
    images = _make_images([0, 1, 2, 3])
    view_graph = ViewGraph()

    calibrated_F = np.array(
        [[0.0, -1.0, 0.5], [1.0, 0.0, -0.25], [-0.5, 0.25, 0.0]],
        dtype=np.float64,
    )
    upgraded_F = np.array(
        [[0.0, -0.5, 0.2], [0.5, 0.0, -0.1], [-0.2, 0.1, 0.0]],
        dtype=np.float64,
    )

    view_graph.image_pairs = {
        (0, 1): ImagePair(0, 1, F=calibrated_F.copy(), E=np.eye(3), config=ConfigurationType.CALIBRATED),
        (0, 2): ImagePair(0, 2, F=calibrated_F.copy(), E=np.eye(3), config=ConfigurationType.CALIBRATED),
        (1, 3): ImagePair(1, 3, F=calibrated_F.copy(), E=np.eye(3), config=ConfigurationType.CALIBRATED),
        (2, 3): ImagePair(2, 3, F=calibrated_F.copy(), E=np.eye(3), config=ConfigurationType.CALIBRATED),
        (0, 3): ImagePair(0, 3, F=upgraded_F.copy(), config=ConfigurationType.UNCALIBRATED),
    }

    upgraded = _cross_validate_prior_focal_lengths(
        _collect_valid_pairs(view_graph),
        cameras,
        images,
        min_calibrated_pair_ratio=0.5,
    )

    upgraded_pair = view_graph.image_pairs[(0, 3)]
    assert upgraded == 1
    assert upgraded_pair.config == ConfigurationType.CALIBRATED
    np.testing.assert_allclose(
        upgraded_pair.E,
        _essential_from_fundamental(cameras[0], cameras[3], upgraded_F),
    )


def test_accept_calibrated_focals_promotes_only_inlier_ratio_updates():
    cameras = _make_cameras(3, [True, False, False])
    initial_focals = np.array([800.0, 825.0, 850.0], dtype=np.float64)
    optimized_focals = np.array([810.0, 900.0, 20000.0], dtype=np.float64)

    final_focals, accepted_mask, rejected = _accept_calibrated_focals(
        cameras,
        initial_focals,
        optimized_focals,
        optimized_camera_ids=[0, 1, 2],
        options={"thres_lower_ratio": 0.1, "thres_higher_ratio": 10.0},
    )
    promoted, refined = _apply_camera_calibration(
        cameras,
        initial_focals,
        final_focals,
        accepted_mask,
        promote_refined_to_prior=False,
        refined_focal_min_ratio_change=1.02,
        refined_focal_full_ratio_change=1.10,
    )

    assert rejected == 1
    assert promoted == 0
    assert refined == 1
    assert cameras[0].has_prior_focal_length
    assert not cameras[1].has_prior_focal_length
    assert cameras[1].has_refined_focal_length
    assert 0.85 < cameras[1].refined_focal_confidence < 1.0
    assert not cameras[2].has_prior_focal_length
    assert not cameras[2].has_refined_focal_length
    np.testing.assert_allclose(cameras[1].focal_length, [900.0, 900.0])
    np.testing.assert_allclose(cameras[2].focal_length, [850.0, 850.0])


def test_apply_camera_calibration_gives_zero_confidence_to_unchanged_focal():
    cameras = _make_cameras(2, [True, False])
    initial_focals = np.array([800.0, 825.0], dtype=np.float64)
    final_focals = np.array([800.0, 825.0], dtype=np.float64)
    accepted_mask = np.array([True, True], dtype=bool)

    promoted, refined = _apply_camera_calibration(
        cameras,
        initial_focals,
        final_focals,
        accepted_mask,
        promote_refined_to_prior=False,
        refined_focal_min_ratio_change=1.02,
        refined_focal_full_ratio_change=1.10,
    )

    assert promoted == 0
    assert refined == 1
    assert cameras[1].has_refined_focal_length
    assert cameras[1].refined_focal_confidence == 0.0


def test_apply_pair_acceptance_marks_degenerate_pairs():
    cameras = _make_cameras(2, [True, True])
    images = _make_images([0, 1])
    F = np.array(
        [[0.0, -1.0, 0.1], [1.0, 0.0, -0.2], [-0.1, 0.2, 0.0]],
        dtype=np.float64,
    )
    accepted_pair = ImagePair(0, 1, F=F.copy(), config=ConfigurationType.UNCALIBRATED)
    rejected_pair = ImagePair(0, 1, F=F.copy(), config=ConfigurationType.UNCALIBRATED)
    input_pair_items = [((0, 1), accepted_pair), ((0, 1), rejected_pair)]

    invalid = _apply_pair_acceptance(
        input_pair_items,
        cameras,
        images,
        pair_errors_sq=[1.0, 9.0],
        max_error_sq=4.0,
        min_pair_calibrated_focal_confidence=0.5,
    )

    assert invalid == 1
    assert accepted_pair.is_valid
    assert accepted_pair.config == ConfigurationType.CALIBRATED
    np.testing.assert_allclose(
        accepted_pair.E,
        _essential_from_fundamental(cameras[0], cameras[1], F),
    )
    assert not rejected_pair.is_valid
    assert rejected_pair.config == ConfigurationType.DEGENERATE


def test_apply_pair_acceptance_keeps_low_confidence_pair_uncalibrated():
    cameras = _make_cameras(2, [False, False])
    cameras[0].has_refined_focal_length = True
    cameras[0].refined_focal_confidence = 0.3
    cameras[1].has_refined_focal_length = True
    cameras[1].refined_focal_confidence = 0.4
    images = _make_images([0, 1])
    F = np.array(
        [[0.0, -1.0, 0.1], [1.0, 0.0, -0.2], [-0.1, 0.2, 0.0]],
        dtype=np.float64,
    )
    accepted_pair = ImagePair(0, 1, F=F.copy(), config=ConfigurationType.CALIBRATED)

    invalid = _apply_pair_acceptance(
        [((0, 1), accepted_pair)],
        cameras,
        images,
        pair_errors_sq=[1.0],
        max_error_sq=4.0,
        min_pair_calibrated_focal_confidence=0.5,
    )

    assert invalid == 0
    assert accepted_pair.is_valid
    assert accepted_pair.config == ConfigurationType.UNCALIBRATED
