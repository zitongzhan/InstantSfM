import numpy as np

from instantsfm.processors.track_retriangulation import (
    complete_tracks,
    merge_tracks,
    retriangulate_underreconstructed_pairs,
)
from instantsfm.scene.defs import CameraModelId, Cameras, ImagePair, Images, Tracks, ViewGraph


def _project_point(camera_center_x, xyz, focal=1000.0):
    x = focal * (xyz[0] - camera_center_x) / xyz[2]
    y = focal * xyz[1] / xyz[2]
    return np.array([x, y], dtype=np.float64)


def _make_camera():
    cameras = Cameras(1)
    cameras.model_ids[0] = CameraModelId.SIMPLE_PINHOLE.value
    cameras.widths[0] = 2000
    cameras.heights[0] = 1500
    cameras.set_params(0, np.array([1000.0, 0.0, 0.0], dtype=np.float64), CameraModelId.SIMPLE_PINHOLE)
    return cameras


def _make_images(camera_centers_x, feature_layout):
    images = Images(len(camera_centers_x))
    images.cam_ids[:] = 0
    images.is_registered[:] = True

    for image_id, center_x in enumerate(camera_centers_x):
        world2cam = np.eye(4, dtype=np.float64)
        world2cam[0, 3] = -center_x
        images.world2cams[image_id] = world2cam
        features = np.asarray(feature_layout[image_id], dtype=np.float64)
        images.features[image_id] = features
        images.features_undist[image_id] = np.concatenate(
            [features / 1000.0, np.ones((features.shape[0], 1), dtype=np.float64)],
            axis=1,
        )

    return images


def _make_tracks(track_specs):
    tracks = Tracks(num_tracks=len(track_specs))
    for track_idx, (track_id, xyz, observations) in enumerate(track_specs):
        tracks.ids[track_idx] = track_id
        tracks.xyzs[track_idx] = np.asarray(xyz, dtype=np.float64)
        tracks.observations[track_idx] = np.asarray(observations, dtype=np.int32)
    return tracks


def test_complete_tracks_skips_observations_owned_by_other_tracks():
    xyz = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    camera_centers_x = [0.0, 0.25, 0.5]
    feature_layout = [
        [_project_point(camera_centers_x[0], xyz)],
        [_project_point(camera_centers_x[1], xyz)],
        [_project_point(camera_centers_x[2], xyz)],
    ]
    cameras = _make_camera()
    images = _make_images(camera_centers_x, feature_layout)
    tracks = _make_tracks(
        [
            (100, xyz, [(0, 0), (1, 0)]),
            (200, xyz, [(2, 0)]),
        ]
    )
    tracks_orig = {
        100: np.asarray([(0, 0), (1, 0), (2, 0)], dtype=np.int32),
    }

    num_completed = complete_tracks(
        cameras,
        images,
        tracks,
        tracks_orig,
        {"complete_max_reproj_error": 1.0},
    )

    assert num_completed == 0
    assert tracks.observations[0].tolist() == [[0, 0], [1, 0]]
    assert tracks.observations[1].tolist() == [[2, 0]]


def test_merge_tracks_requires_feature_correspondence():
    xyz = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    camera_centers_x = [0.0, 0.25, 0.5, 0.75]
    feature_layout = [
        [_project_point(camera_centers_x[0], xyz)],
        [_project_point(camera_centers_x[1], xyz)],
        [_project_point(camera_centers_x[2], xyz)],
        [_project_point(camera_centers_x[3], xyz)],
    ]
    cameras = _make_camera()
    images = _make_images(camera_centers_x, feature_layout)
    tracks = _make_tracks(
        [
            (100, xyz, [(0, 0), (1, 0)]),
            (200, xyz, [(2, 0), (3, 0)]),
        ]
    )
    view_graph = ViewGraph()

    num_merged = merge_tracks(
        view_graph,
        cameras,
        images,
        tracks,
        {"merge_max_reproj_error": 1.0},
    )

    assert num_merged == 0
    assert len(tracks) == 2


def test_merge_tracks_uses_correspondence_connected_candidates():
    xyz = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    camera_centers_x = [0.0, 0.25, 0.5, 0.75]
    feature_layout = [
        [_project_point(camera_centers_x[0], xyz)],
        [_project_point(camera_centers_x[1], xyz)],
        [_project_point(camera_centers_x[2], xyz)],
        [_project_point(camera_centers_x[3], xyz)],
    ]
    cameras = _make_camera()
    images = _make_images(camera_centers_x, feature_layout)
    tracks = _make_tracks(
        [
            (100, xyz, [(0, 0), (1, 0)]),
            (200, xyz, [(2, 0), (3, 0)]),
        ]
    )

    pair = ImagePair(image_id1=1, image_id2=2, is_valid=True, inliers=[0])
    pair.matches = np.asarray([(0, 0)], dtype=np.int32)
    view_graph = ViewGraph()
    view_graph.image_pairs[(1, 2)] = pair

    num_merged = merge_tracks(
        view_graph,
        cameras,
        images,
        tracks,
        {"merge_max_reproj_error": 1.0},
    )

    assert num_merged == 4
    assert len(tracks) == 1
    assert tracks.observations[0].tolist() == [[0, 0], [1, 0], [2, 0], [3, 0]]


def test_retriangulation_continue_skips_duplicate_image_observation():
    xyz = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    camera_centers_x = [0.0, 0.25]
    feature_layout = [
        [_project_point(camera_centers_x[0], xyz)],
        [
            _project_point(camera_centers_x[1], xyz),
            _project_point(camera_centers_x[1], xyz),
        ],
    ]
    cameras = _make_camera()
    images = _make_images(camera_centers_x, feature_layout)
    tracks = _make_tracks(
        [
            (100, xyz, [(0, 0), (1, 0)]),
        ]
    )

    pair = ImagePair(image_id1=0, image_id2=1, is_valid=True, inliers=[0])
    pair.matches = np.asarray([(0, 1)], dtype=np.int32)
    view_graph = ViewGraph()
    view_graph.image_pairs[(0, 1)] = pair

    num_retriangulated = retriangulate_underreconstructed_pairs(
        view_graph,
        cameras,
        images,
        tracks,
        {
            "complete_max_reproj_error": 1.0,
            "re_max_angle_error": 3.0,
            "re_min_ratio": 1.0,
            "re_max_trials": 1,
            "filter_min_tri_angle": 1.5,
            "ignore_two_view_tracks": True,
        },
        {},
    )

    assert num_retriangulated == 0
    assert tracks.observations[0].tolist() == [[0, 0], [1, 0]]
