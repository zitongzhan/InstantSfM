import numpy as np
import pyceres
import torch
import tqdm

from instantsfm.scene.defs import ConfigurationType, ViewGraph
from instantsfm.utils.cost_function import (
    FetzerFocalLengthCostFunction,
    FetzerFocalLengthSameCameraCostFunction,
    fetzer_cost,
    fetzer_ds,
)
from instantsfm.utils.optimization_models import FetzerCalibrationModel

# used by torch LM
from torch import nn
import pypose as pp
from pypose.optim.kernel import Cauchy
from bae.utils.pysolvers import PCG
from bae.optim import LM
from bae.autograd.function import TrackingTensor


def _collect_valid_pairs(view_graph: ViewGraph):
    return [
        (pair_id, image_pair)
        for pair_id, image_pair in view_graph.image_pairs.items()
        if image_pair.is_valid
        and image_pair.config
        in [ConfigurationType.CALIBRATED, ConfigurationType.UNCALIBRATED]
    ]


def _essential_from_fundamental(cam1, cam2, F):
    return cam2.get_K().T @ F @ cam1.get_K()


def _fundamental_from_essential(cam1, cam2, E):
    K1_inv = np.linalg.inv(cam1.get_K())
    K2_inv = np.linalg.inv(cam2.get_K())
    return K2_inv.T @ E @ K1_inv


def _cross_validate_prior_focal_lengths(
    valid_pair_items,
    cameras,
    images,
    min_calibrated_pair_ratio,
):
    camera_counter_total = np.zeros(len(cameras), dtype=np.int32)
    camera_counter_calib = np.zeros(len(cameras), dtype=np.int32)

    for _, image_pair in valid_pair_items:
        cam_id1 = images[image_pair.image_id1].cam_id
        cam_id2 = images[image_pair.image_id2].cam_id
        camera1 = cameras[cam_id1]
        camera2 = cameras[cam_id2]
        if not camera1.has_prior_focal_length or not camera2.has_prior_focal_length:
            continue
        camera_counter_total[cam_id1] += 1
        camera_counter_total[cam_id2] += 1
        if image_pair.config == ConfigurationType.CALIBRATED:
            camera_counter_calib[cam_id1] += 1
            camera_counter_calib[cam_id2] += 1

    camera_validity = np.zeros(len(cameras), dtype=bool)
    nonzero_mask = camera_counter_total > 0
    camera_validity[nonzero_mask] = (
        camera_counter_calib[nonzero_mask] / camera_counter_total[nonzero_mask]
        > min_calibrated_pair_ratio
    )

    num_upgraded_pairs = 0
    for _, image_pair in valid_pair_items:
        if image_pair.config != ConfigurationType.UNCALIBRATED:
            continue
        cam_id1 = images[image_pair.image_id1].cam_id
        cam_id2 = images[image_pair.image_id2].cam_id
        if not (camera_validity[cam_id1] and camera_validity[cam_id2]):
            continue
        if image_pair.F is None:
            continue
        cam1 = cameras[cam_id1]
        cam2 = cameras[cam_id2]
        image_pair.E = _essential_from_fundamental(cam1, cam2, image_pair.F)
        image_pair.config = ConfigurationType.CALIBRATED
        num_upgraded_pairs += 1
    return num_upgraded_pairs


def _refresh_calibrated_pair_fundamentals(valid_pair_items, cameras, images):
    for _, image_pair in valid_pair_items:
        if image_pair.config != ConfigurationType.CALIBRATED or image_pair.E is None:
            continue
        cam1 = cameras[images[image_pair.image_id1].cam_id]
        cam2 = cameras[images[image_pair.image_id2].cam_id]
        image_pair.F = _fundamental_from_essential(cam1, cam2, image_pair.E)


def _accept_calibrated_focals(
    cameras,
    initial_focals,
    optimized_focals,
    optimized_camera_ids,
    options,
):
    final_focals = initial_focals.copy()
    accepted_mask = np.array(
        [cam.has_prior_focal_length for cam in cameras], dtype=bool
    )
    rejected_counter = 0

    lower_ratio = options["thres_lower_ratio"]
    upper_ratio = options["thres_higher_ratio"]

    for cam_id in optimized_camera_ids:
        cam = cameras[cam_id]
        if cam.has_prior_focal_length:
            continue

        focal = float(optimized_focals[cam_id])
        ratio = focal / max(initial_focals[cam_id], 1e-12)
        if ratio < lower_ratio or ratio > upper_ratio:
            rejected_counter += 1
            continue

        final_focals[cam_id] = focal
        accepted_mask[cam_id] = True

    return final_focals, accepted_mask, rejected_counter


def _apply_camera_calibration(
    cameras,
    initial_focals,
    final_focals,
    accepted_mask,
    promote_refined_to_prior,
    refined_focal_min_ratio_change,
    refined_focal_full_ratio_change,
):
    promoted_counter = 0
    refined_counter = 0
    cameras.has_refined_focal_length[:] = False
    cameras.refined_focal_confidence[:] = 0.0

    min_log_change = np.log(max(refined_focal_min_ratio_change, 1.0))
    full_log_change = np.log(max(refined_focal_full_ratio_change, refined_focal_min_ratio_change))
    log_change_denom = max(full_log_change - min_log_change, 1e-12)

    for cam_id, cam in enumerate(cameras):
        if cam.has_prior_focal_length:
            continue

        if accepted_mask[cam_id]:
            focal = final_focals[cam_id]
            cam.focal_length = np.array([focal, focal], dtype=np.float64)
            ratio_log_change = abs(np.log(max(focal, 1e-12) / max(initial_focals[cam_id], 1e-12)))
            confidence = np.clip(
                (ratio_log_change - min_log_change) / log_change_denom,
                0.0,
                1.0,
            )
            cam.has_refined_focal_length = True
            cam.refined_focal_confidence = float(confidence)
            refined_counter += 1
            if promote_refined_to_prior:
                cam.has_prior_focal_length = True
                promoted_counter += 1
        else:
            focal = initial_focals[cam_id]
            cam.focal_length = np.array([focal, focal], dtype=np.float64)

    return promoted_counter, refined_counter


def _apply_pair_acceptance(
    input_pair_items,
    cameras,
    images,
    pair_errors_sq,
    max_error_sq,
    min_pair_calibrated_focal_confidence,
):
    invalid_counter = 0

    for (pair_id, image_pair), error_sq in zip(input_pair_items, pair_errors_sq):
        _ = pair_id
        if error_sq > max_error_sq:
            image_pair.config = ConfigurationType.DEGENERATE
            image_pair.is_valid = False
            invalid_counter += 1
            continue

        cam1 = cameras[images[image_pair.image_id1].cam_id]
        cam2 = cameras[images[image_pair.image_id2].cam_id]

        cam1_conf = 1.0 if cam1.has_prior_focal_length else cam1.refined_focal_confidence
        cam2_conf = 1.0 if cam2.has_prior_focal_length else cam2.refined_focal_confidence

        if min(cam1_conf, cam2_conf) >= min_pair_calibrated_focal_confidence:
            image_pair.E = _essential_from_fundamental(cam1, cam2, image_pair.F)
            image_pair.config = ConfigurationType.CALIBRATED
        else:
            image_pair.config = ConfigurationType.UNCALIBRATED
        image_pair.is_valid = True

    return invalid_counter


def SolveViewGraphCalibration(view_graph:ViewGraph, cameras, images, VIEW_GRAPH_CALIBRATOR_OPTIONS):
    valid_pair_items = _collect_valid_pairs(view_graph)
    if not valid_pair_items:
        print('No image pairs to calibrate')
        return

    cameras.has_refined_focal_length[:] = False
    cameras.refined_focal_confidence[:] = 0.0

    if VIEW_GRAPH_CALIBRATOR_OPTIONS.get('cross_validate_prior_focal_lengths', True):
        num_upgraded_pairs = _cross_validate_prior_focal_lengths(
            valid_pair_items,
            cameras,
            images,
            VIEW_GRAPH_CALIBRATOR_OPTIONS.get('min_calibrated_pair_ratio', 0.5),
        )
        print(
            f'Upgraded {num_upgraded_pairs} / {len(valid_pair_items)} pairs to calibrated through cross-validation'
        )

    _refresh_calibrated_pair_fundamentals(valid_pair_items, cameras, images)

    initial_focals = np.array([np.mean(cam.focal_length) for cam in cameras], dtype=np.float64)
    focals = initial_focals.copy()
    parameter_blocks = [focals[idx:idx + 1] for idx in range(len(cameras))]

    problem = pyceres.Problem()
    options = pyceres.SolverOptions()
    loss_function = pyceres.CauchyLoss(VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'])
    if len(cameras) < 50:
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_NORMAL_CHOLESKY
    else:
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY

    input_pair_items = []
    optimized_camera_ids = set()
    for pair_id, image_pair in valid_pair_items:
        if image_pair.F is None or np.asarray(image_pair.F).shape != (3, 3):
            continue
        image1, image2 = images[image_pair.image_id1], images[image_pair.image_id2]
        cam1, cam2 = cameras[image1.cam_id], cameras[image2.cam_id]
        cam_id1, cam_id2 = image1.cam_id, image2.cam_id
        if cam_id1 == cam_id2:
            cost_function = FetzerFocalLengthSameCameraCostFunction(image_pair.F, cam1.principal_point)
            problem.add_residual_block(cost_function, loss_function, [parameter_blocks[cam_id1]])
        else:
            cost_function = FetzerFocalLengthCostFunction(image_pair.F, cam1.principal_point, cam2.principal_point)
            problem.add_residual_block(
                cost_function,
                loss_function,
                [parameter_blocks[cam_id1], parameter_blocks[cam_id2]],
            )
        input_pair_items.append((pair_id, image_pair))
        optimized_camera_ids.add(cam_id1)
        optimized_camera_ids.add(cam_id2)

    if not input_pair_items:
        print('No valid fundamental matrices found for view graph calibration')
        return

    num_cameras_to_optimize = 0
    for cam_id in sorted(optimized_camera_ids):
        parameter_block = parameter_blocks[cam_id]
        problem.set_parameter_lower_bound(parameter_block, 0, 1e-3)
        if cameras[cam_id].has_prior_focal_length:
            problem.set_parameter_block_constant(parameter_block)
        else:
            num_cameras_to_optimize += 1

    if num_cameras_to_optimize == 0:
        print('No cameras to optimize in view graph calibration')
        return

    options.max_num_iterations = VIEW_GRAPH_CALIBRATOR_OPTIONS['max_num_iterations']
    options.function_tolerance = VIEW_GRAPH_CALIBRATOR_OPTIONS['function_tolerance']

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    print(summary.BriefReport())
    if not summary.IsSolutionUsable():
        print('View graph calibration solver failed')
        return

    final_focals, accepted_mask, rejected_counter = _accept_calibrated_focals(
        cameras,
        initial_focals,
        focals,
        sorted(optimized_camera_ids),
        VIEW_GRAPH_CALIBRATOR_OPTIONS,
    )
    promoted_counter, refined_counter = _apply_camera_calibration(
        cameras,
        initial_focals,
        final_focals,
        accepted_mask,
        VIEW_GRAPH_CALIBRATOR_OPTIONS.get('promote_refined_to_prior', False),
        VIEW_GRAPH_CALIBRATOR_OPTIONS.get('refined_focal_min_ratio_change', 1.02),
        VIEW_GRAPH_CALIBRATOR_OPTIONS.get('refined_focal_full_ratio_change', 1.10),
    )
    print(f'{rejected_counter} cameras are rejected in view graph calibration')
    print(f'{promoted_counter} cameras are promoted to calibrated priors')
    print(f'{refined_counter} cameras keep accepted refined focals')

    eval_options = pyceres.EvaluateOptions()
    eval_options.apply_loss_function = False
    residuals = problem.evaluate_residuals(eval_options)
    thres_two_view_error_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_two_view_error'] ** 2

    pair_errors_sq = []
    for idx in range(len(input_pair_items)):
        residual = residuals[2 * idx:2 * idx + 2]
        pair_errors_sq.append(residual[0]**2 + residual[1]**2)

    invalid_counter = _apply_pair_acceptance(
        input_pair_items,
        cameras,
        images,
        pair_errors_sq,
        thres_two_view_error_sq,
        VIEW_GRAPH_CALIBRATOR_OPTIONS.get('min_pair_calibrated_focal_confidence', 0.5),
    )
    print(f'invalid / total number of two view geometry: {invalid_counter} / {len(input_pair_items)}')

class TorchVGC():
    def __init__(self, device='cuda:0'):
        self.device = device

    def Optimize(self, view_graph:ViewGraph, cameras, images, VIEW_GRAPH_CALIBRATOR_OPTIONS):
        cost_fn = fetzer_cost

        valid_image_pairs = {pair_id: image_pair for pair_id, image_pair in view_graph.image_pairs.items()
                             if image_pair.is_valid and image_pair.config in [ConfigurationType.CALIBRATED, ConfigurationType.UNCALIBRATED]}
        focals = torch.tensor(np.array([np.mean(cam.focal_length) for cam in cameras]), dtype=torch.float64).to(self.device).unsqueeze(-1)
        self.camera_has_prior = torch.tensor([cam.has_prior_focal_length for cam in cameras], dtype=torch.bool).to(self.device)
        # TODO: Only support all cameras have prior focal length. If some cameras have prior focal length while others do not, 
        # they will be optimized together, which is not a good idea.
        if torch.all(self.camera_has_prior):
            print('All cameras have prior focal length, skipping view graph calibration')
            return

        ds_list = []
        camera_indices1_list = []
        camera_indices2_list = []
        for image_pair in valid_image_pairs.values():
            # add both directions
            image1, image2 = images[image_pair.image_id1], images[image_pair.image_id2]
            cam_id1, cam_id2 = image1.cam_id, image2.cam_id
            cam1, cam2 = cameras[cam_id1], cameras[cam_id2]
            principal_point0, principal_point1 = cam1.principal_point, cam2.principal_point
            K0 = np.array([[1, 0, principal_point0[0]], [0, 1, principal_point0[1]], [0, 0, 1]])
            K1 = np.array([[1, 0, principal_point1[0]], [0, 1, principal_point1[1]], [0, 0, 1]])
            i1_G_i0 = K1.T @ image_pair.F @ K0
            ds = fetzer_ds(i1_G_i0)
            ds_list.append(ds)
            camera_indices1_list.append(cam_id1)
            camera_indices2_list.append(cam_id2)
            i0_G_i1 = i1_G_i0.T
            ds = fetzer_ds(i0_G_i1)
            ds_list.append(ds)
            camera_indices1_list.append(cam_id2)
            camera_indices2_list.append(cam_id1)
        
        ds = torch.tensor(np.array(ds_list), dtype=torch.float64).to(self.device).unsqueeze(1)
        camera_indices1 = torch.tensor(np.array(camera_indices1_list), dtype=torch.int64).to(self.device).flatten()
        camera_indices2 = torch.tensor(np.array(camera_indices2_list), dtype=torch.int64).to(self.device).flatten()

        model = FetzerCalibrationModel(focals, cost_fn)
        strategy = pp.optim.strategy.TrustRegion(radius=1e2, max=1e6, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5) # cuSolverSP()
        cauchy_kernel = Cauchy(VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=cauchy_kernel, reject=30)

        input = {
            "ds": ds,
            "camera_indices1": camera_indices1,
            "camera_indices2": camera_indices2
        }

        window_size = 3
        loss_history = []
        progress_bar = tqdm.trange(VIEW_GRAPH_CALIBRATOR_OPTIONS['max_num_iterations'])
        for _ in progress_bar:
            loss = optimizer.step(input)
            torch.set_printoptions(threshold=torch.inf)
            print(f'focals: {model.focals}, loss: {loss.item()}')
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < VIEW_GRAPH_CALIBRATOR_OPTIONS['function_tolerance']:
                    break
            progress_bar.set_postfix({"loss": loss.item()})
        progress_bar.close()

        focals_ = focals.detach().cpu().numpy().squeeze()
        counter = 0
        for cam, focal in zip(cameras, focals_):
            if (focal / np.mean(cam.focal_length) < VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_lower_ratio'] or 
                focal / np.mean(cam.focal_length) > VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_higher_ratio']):
                counter += 1
                continue
            cam.has_refined_focal_length = True
            cam.focal_length = np.array([focal, focal])
        
        print(f'{counter} cameras are rejected in view graph calibration')

        thres_two_view_error_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_two_view_error'] ** 2

        # manually calculate the residuals
        loss = model.forward(ds, camera_indices1, camera_indices2).detach().cpu().numpy()
        invalid_counter = 0
        loss_sq = np.sum(loss ** 2, axis=-1)
        for idx, (pair_id, image_pair) in enumerate(valid_image_pairs.items()):
            if loss_sq[idx*2] > thres_two_view_error_sq:
                invalid_counter += 1
                view_graph.image_pairs[pair_id].is_valid = False
        print(f'invalid / total number of two view geometry: {invalid_counter} / {len(valid_image_pairs)}')
