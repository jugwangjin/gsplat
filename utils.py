import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
import tqdm
import math
from gsplat.utils import save_ply, depth_to_points
from torch.utils.data import Dataset
from gsplat.optimizers import SelectiveAdam

from gsplat.strategy.ops import duplicate, remove, reset_opa, split, remove_return

from torch.cuda.amp import autocast


import time


@torch.no_grad()
def create_splats_and_optimizers_from_data(
    xyzs: Tensor,
    rgbs: Tensor,
    runner=None,
    device: str = "cuda",
    batch_size: int = 1,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    world_size: int = 1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    shN = None,
    scales=None,
    quats=None,
    opacities=None,
    optimizers=None,
    keep_feats=False,
    indices=None,
    keep_mean_sh0=False,
):
    shN_channels = 15 if runner is None else runner.splats["shN"].shape[1]

    # xyzs shape of N, 3
    # rgbs shape of N, 3
    means = xyzs 
    if len(rgbs.shape) == 2:
        rgbs = rgbs[:, None]
    sh0 = rgb_to_sh(rgbs)
    if shN is None:
        shN = torch.zeros((sh0.shape[0], shN_channels, 3), device=device)


    if scales is None:
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(means, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    if quats is None:
        quats = torch.rand((means.shape[0], 4), device=device)  # [N, 4]\
    if opacities is None:
        opacities = torch.logit(torch.full((means.shape[0],), 0.1, device=device))  # [N,]


    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(means.clone().detach().data), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales.clone().detach().data), 5e-3),
        ("quats", torch.nn.Parameter(quats.clone().detach().data), 1e-3),
        ("opacities", torch.nn.Parameter(opacities.clone().detach().data), 5e-2),
        ("sh0", torch.nn.Parameter(sh0.clone().detach().data), 2.5e-3),
        ("shN", torch.nn.Parameter(shN.clone().detach().data), 2.5e-3 / 20),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    new_n_gaussians = means.shape[0]

    if optimizers is None:
        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent, see
        # https://arxiv.org/pdf/2402.18824v1
        BS = batch_size * world_size
        optimizer_class = None
        if sparse_grad:
            optimizer_class = torch.optim.SparseAdam
        elif visible_adam:
            optimizer_class = SelectiveAdam
        else:
            optimizer_class = torch.optim.Adam
        optimizers = {
            name: optimizer_class(
                [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, _, lr in params
        }
    else:
        if not keep_feats and keep_mean_sh0:
            def optimizer_fn(key, v):
                # shape : [new n gaussians, v shape [1:]]
                # I don't know the length of v shape
                if key != "sh0" and key != "means":
                    num_v = torch.zeros(new_n_gaussians, *v.shape[1:], device=device)
                    return num_v
                else:
                    return v[indices]
        if not keep_feats:
            def optimizer_fn(key, v):
                # shape : [new n gaussians, v shape [1:]]
                # I don't know the length of v shape

                num_v = torch.zeros(new_n_gaussians, *v.shape[1:], device=device)
                return num_v
        else:
            def optimizer_fn(key, v):
                # shape : [new n gaussians, v shape [1:]]
                # I don't know the length of v shape
                return v[indices]

        for name in [n for n, _, _ in params]:
            param = runner.splats[name]
            new_param = splats[name]
            if name not in optimizers:
                assert not param.requires_grad, (
                    f"Optimizer for {name} is not found, but the parameter is trainable."
                    f"Got requires_grad={param.requires_grad}"
                )
                continue
            optimizer = optimizers[name]
            for i in range(len(optimizer.param_groups)):
                param_state = optimizer.state[param]
                del optimizer.state[param]
                for key in param_state.keys():
                    if key != "step":
                        v = param_state[key]
                        param_state[key] = optimizer_fn(key, v)
                optimizer.param_groups[i]["params"] = [new_param]
                optimizer.state[new_param] = param_state



    # # Scale learning rate based on batch size, reference:
    # # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # # Note that this would not make the training exactly equivalent, see
    # # https://arxiv.org/pdf/2402.18824v1
    # BS = batch_size * world_size
    # optimizer_class = None
    # if sparse_grad:
    #     optimizer_class = torch.optim.SparseAdam
    # elif visible_adam:
    #     optimizer_class = SelectiveAdam
    # else:
    #     optimizer_class = torch.optim.Adam
    # optimizers = {
    #     name: optimizer_class(
    #         [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
    #         eps=1e-15 / math.sqrt(BS),
    #         # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
    #         betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
    #     )
    #     for name, _, lr in params
    # }
    return splats, optimizers


@torch.no_grad()
def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
    
    prune_mask = ~non_prune_mask

    return prune_mask


@torch.no_grad()
def simplification(
    trainset: Dataset, 
    runner,
    parser,
    cfg,
    sampling_factor: float = 0.1,
    cdf_threshold: float = 0.99,
    use_cdf_mask: bool = False,
    keep_sh0: bool = True,
    keep_feats: bool = False,
    batch_size: int = 1,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    world_size: int = 1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    optimizers=None,
    abs_ratio=False,
    trainloader=None,
):
    # if trainloader is None:
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
    )

    n_gaussian = runner.splats["means"].shape[0]

    importance_scores = torch.zeros(n_gaussian, device=runner.splats["means"].device)
    pixels_per_gaussian = torch.zeros(n_gaussian, device=runner.splats["means"].device, dtype=torch.int)

    trainloader_iter = iter(trainloader)

    for step in tqdm.tqdm(range(len(trainloader))):
        data = next(trainloader_iter)

        pixel = data["image"].to(runner.splats["means"].device) / 255.0
        camtoworlds = camtoworlds_gt = data["camtoworld"].to(runner.splats["means"].device)
        Ks = data["K"].to(runner.splats["means"].device)
        image_ids = data["image_id"].to(runner.splats["means"].device)
        masks = data["mask"].to(runner.splats["means"].device) if "mask" in data else None

        width, height = pixel.shape[1:3]

        # forward
        _, _, info = runner.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=0,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+IW",
            masks=masks,
        )

        max_ids = info["max_ids"]

        max_ids_valid_mask = max_ids >= 0

        max_ids = max_ids % n_gaussian

        batch_pixels_per_gaussian = torch.zeros_like(pixels_per_gaussian)
        batch_pixels_per_gaussian.index_add_(
            0, max_ids[max_ids_valid_mask].flatten(), torch.ones_like(max_ids[max_ids_valid_mask]).flatten()
        )

        pixels_per_gaussian += batch_pixels_per_gaussian

        accumulated_weights_value = info["accumulated_weights_value"] # C N
        accumulated_weights_count = info["accumulated_weights_count"] # C N


        # add importance score to impotance_scores, but only where accumulated_weights_count > 0
        for i in range(accumulated_weights_value.shape[0]):
            importance_score = accumulated_weights_value[i] / (accumulated_weights_count[i].clamp(min=1))
            accumulated_weights_valid_mask = batch_pixels_per_gaussian > 0
            # importance_scores += importance_score
            importance_scores[accumulated_weights_valid_mask] += importance_score[accumulated_weights_valid_mask]
 
    importance_scores[pixels_per_gaussian == 0] = 0
    # print(pixels_per_gaussian.shape, (pixels_per_gaussian==0).sum())

    if use_cdf_mask:
        non_prune_mask = init_cdf_mask(importance_scores, cdf_threshold)
        indices = torch.nonzero(~non_prune_mask, as_tuple=True)[0]

    else:
        prob = importance_scores / importance_scores.sum()
        prob = prob.cpu().numpy()
        if abs_ratio:
            n_sample = int(n_gaussian * sampling_factor)
        else:
            n_sample = int(n_gaussian * sampling_factor * ((prob !=0).sum()/prob.shape[0]))

        indices = np.random.choice(n_gaussian, n_sample, p=prob, replace=False)

        # indices = np.unique(indices)

        sampling_mask = torch.ones(n_gaussian, device=runner.splats["means"].device)
        sampling_mask[indices] = 0

    # if keep_gradients:
    #     remove(runner.splats, runner.optimizers, strategy.state, sampling_mask)
    #     splats, optimizers = runner.splats, runner.optimizers

    # else

    if keep_feats:
        # invert indices
        mask = torch.zeros(n_gaussian, device=runner.splats["means"].device)
        mask[indices] = 1
        mask = 1 - mask
        mask = mask.bool()
        splats, optimizers = remove_return(runner.splats, runner.optimizers, runner.strategy_state, mask)

    else:
        splats, optimizers = create_splats_and_optimizers_from_data(
            runner.splats["means"][indices],
            sh0_to_rgb(runner.splats["sh0"][indices]),
            runner=runner,
            device=runner.splats["means"].device,
            batch_size=batch_size,
            sparse_grad=sparse_grad,
            visible_adam=visible_adam,
            world_size=world_size,
            init_scale=init_scale,
            scene_scale=scene_scale,
            shN=runner.splats["shN"][indices] if keep_feats else None,
            scales=runner.splats["scales"][indices] if keep_feats else None,
            quats=runner.splats["quats"][indices] if keep_feats else None,
            opacities=runner.splats["opacities"][indices] if keep_feats else None,
            optimizers=optimizers,
            keep_feats=keep_feats,
            indices=indices,
        )

    return splats, optimizers


@torch.no_grad()
def simplification_from_mesh_simp(
    trainset: Dataset, 
    runner,
    parser,
    cfg,
    sampling_factor: float = 0.1,
    keep_sh0: bool = True,
    keep_feats: bool = False,
    batch_size: int = 1,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    world_size: int = 1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    optimizers=None,
    abs_ratio=False,
    ascending=False,
    use_mean=False,
    sampling=False,
    iterations=1,
    apply_opacity=False,
    trainloader=None,
):
    """
    Use a mesh simplification-like strategy to prune splats based on their 
    contribution to potential loss. This function calculates a potential loss 
    for each splat, then removes those with the smallest contribution over 
    multiple iterations.
    """
    # If no data loader is provided, create one
    # if trainloader is None:
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
    )

    # Initial number of Gaussians and how many to remove each iteration
    n_gaussian = runner.splats["means"].shape[0]
    target_n_gaussian = int(n_gaussian * sampling_factor)
    pruned_gaussian_per_iteration = max(0, (n_gaussian - target_n_gaussian) // iterations)

    for iteration in range(iterations):
        n_gaussian = runner.splats["means"].shape[0]
        gaussians_to_keep = n_gaussian - pruned_gaussian_per_iteration

        trainloader_iter = iter(trainloader)
        accumulated_increased_losses = torch.zeros(n_gaussian, device=runner.splats["means"].device)
        accumulated_weights_counts = torch.zeros(n_gaussian, device=runner.splats["means"].device)

        for _ in tqdm.tqdm(range(len(trainloader))):
            data = next(trainloader_iter)

            pixel = data["image"].to(runner.splats["means"].device) / 255.0
            camtoworlds = data["camtoworld"].to(runner.splats["means"].device)
            Ks = data["K"].to(runner.splats["means"].device)
            image_ids = data["image_id"].to(runner.splats["means"].device)
            masks = data["mask"].to(runner.splats["means"].device) if "mask" in data else None
            width, height = pixel.shape[1:3]

            _, _, info = runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=0,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+IW",
                masks=masks,
                gt_image=pixel
            )

            increased_losses = info["accumulated_potential_loss"]
            count_ = info["accumulated_weights_count"]
            potential_loss = increased_losses.sum(dim=0) / (width * height)
            if use_mean:
                potential_loss /= count_.sum(dim=0).clamp(min=1)
            accumulated_increased_losses += potential_loss
            accumulated_weights_counts += count_.sum(dim=0)

            if torch.isnan(increased_losses).any():
                break

        # # Shift losses so there are no negative values
        # accumulated_increased_losses -= torch.amin(accumulated_increased_losses)

        # Choose which Gaussians to keep
        if not sampling:
            # Sort based on accumulated loss (ascending or descending)
            indices = torch.argsort(accumulated_increased_losses, descending=not ascending)[:gaussians_to_keep]
        else:
            # Randomly sample based on normalized potential loss
            # prob_mesh_simp = accumulated_increased_losses / accumulated_increased_losses.sum()
            indices = torch.multinomial(accumulated_increased_losses, gaussians_to_keep, replacement=False)

        # If we want to keep further features or do multiple pruning iterations, 
        # we directly remove splats in place. Otherwise, we create new splats.
        if keep_feats or iterations != 1:
            pruning_mask = torch.zeros(n_gaussian, device=runner.splats["means"].device)
            pruning_mask[indices] = 1
            pruning_mask = 1 - pruning_mask
            pruning_mask = pruning_mask.bool()

            # Remove splats in place
            splats, optimizers = remove_return(runner.splats, optimizers, runner.strategy_state, pruning_mask)
        else:
            # Re-create splats and optimizers from the chosen indices
            splats, optimizers = create_splats_and_optimizers_from_data(
                runner.splats["means"][indices],
                sh0_to_rgb(runner.splats["sh0"][indices]),
                runner=runner,
                device=runner.splats["means"].device,
                batch_size=batch_size,
                sparse_grad=sparse_grad,
                visible_adam=visible_adam,
                world_size=world_size,
                init_scale=init_scale,
                scene_scale=scene_scale,
                shN=runner.splats["shN"][indices] if keep_feats else None,
                scales=runner.splats["scales"][indices] if keep_feats else None,
                quats=runner.splats["quats"][indices] if keep_feats else None,
                opacities=runner.splats["opacities"][indices] if keep_feats else None,
                optimizers=optimizers,
                keep_feats=keep_feats,
                indices=indices,
            )

        runner.splats = splats
        print(f"Iteration {iteration} done. Reduced to {len(indices)} splats.")

    return splats, optimizers


@torch.no_grad()
def simplification_with_approx(
    trainset: Dataset, 
    runner,
    parser,
    cfg,
    sampling_factor: float = 0.1,
    keep_sh0: bool = True,
    keep_feats: bool = False,
    batch_size: int = 1,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    world_size: int = 1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    optimizers=None,
    abs_ratio=False,
    ascending=False,
    use_mean=False,
    sampling=False,
    iterations=1,
    apply_opacity=False,
    distance_sigma: float = 2e0,
    distance_alpha: float = 1e-1,
    removal_chunk: int = 100,
    max_removal_iterations: int = 1000,
    chunk_size_alive: int = 1e6,
    chunk_size_removed: int = 5e1,
    trainloader=None,
):
    """가우시안 단순화를 수행하는 함수 (progressive 방식 + 정확도 향상)
    
    Args:
        trainset: 학습 데이터셋
        runner: 학습 러너
        parser: 파서
        cfg: 설정
        sampling_factor: 샘플링 비율
        keep_sh0: sh0 유지 여부
        keep_feats: 특징 유지 여부 (True: 점진적 제거, False: 한번에 제거)
        batch_size: 배치 크기
        sparse_grad: 희소 그래디언트 사용 여부
        visible_adam: visible adam 사용 여부
        world_size: 세계 크기
        init_scale: 초기 스케일
        scene_scale: 장면 스케일
        optimizers: 옵티마이저
        abs_ratio: 절대 비율 사용 여부
        ascending: 오름차순 정렬 여부
        use_mean: 평균 사용 여부
        sampling: 샘플링 사용 여부 (True: 확률 기반 샘플링, False: 손실 기반 선택)
        iterations: 반복 횟수
        apply_opacity: 투명도 적용 여부
        distance_sigma: 거리 시그마
        distance_alpha: 거리 알파
        removal_chunk: 제거 청크 크기
        max_removal_iterations: 최대 제거 반복 횟수
        chunk_size_alive: 생존 청크 크기
        chunk_size_removed: 제거된 청크 크기
        trainloader: 학습 데이터로더
    """
    device = runner.splats["means"].device
    n_gaussians = runner.splats["means"].shape[0]
    gaussians_to_keep = int(n_gaussians * sampling_factor)
    gaussians_to_remove = n_gaussians - gaussians_to_keep

    if trainloader is None:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )

    num_camera = len(trainloader)

    # Progressive removal loop
    for it_iter in range(iterations):
        print(f"Simplification iteration {it_iter + 1}/{iterations}")
        
        current_gaussians_to_remove = gaussians_to_remove // iterations 
        n_gaussians = runner.splats["means"].shape[0]

        # 누적값 초기화 - 메모리 효율적인 방식으로
        accumulated_values = {
            "gt_colors": torch.zeros(n_gaussians, 3, device=device),
            "final_colors": torch.zeros(n_gaussians, 3, device=device),
            "cur_colors": torch.zeros(n_gaussians, 3, device=device),
            "one_minus_alphas": torch.zeros(n_gaussians, device=device),
            "colors": torch.zeros(n_gaussians, 3, device=device),
            "weights_count": torch.zeros(n_gaussians, device=device)
        }
        
        # 카메라별 데이터 저장용 텐서
        camera_data = {
            "radiis": torch.zeros(len(trainloader), n_gaussians, device=device),
            "means2ds": torch.zeros(len(trainloader), n_gaussians, 2, device=device)
        }

        gaussians_max_accumulated_weights = torch.zeros(n_gaussians, device=device)
        gaussians_max_accumulated_weights_index = torch.zeros(n_gaussians, device=device, dtype=torch.int32)

        # 데이터 수집 단계
        for batch_idx, data in enumerate(tqdm.tqdm(trainloader, desc="Computing potential loss")):
            # 데이터 준비
            pixel = data["image"].to(device) / 255.0
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None

            # 렌더링
            _, _, info = runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=pixel.shape[2],
                height=pixel.shape[1],
                sh_degree=0,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+IW",
                masks=masks,
                gt_image=pixel,
                use_approx=True,
            )

            # 누적값 업데이트
            for key in ["gt_colors", "final_colors", "cur_colors", "colors"]:
                accumulated_values[key] += info[f"accumulated_{key}"].sum(dim=0)
            accumulated_values["one_minus_alphas"] += info["accumulated_one_minus_alphas"].sum(dim=0)
            accumulated_values["weights_count"] += info["accumulated_weights_count"].sum(dim=0)

            # 카메라별 데이터 저장
            camera_data["radiis"][batch_idx] = info["radii"][0]
            camera_data["means2ds"][batch_idx] = info["means2d"][0]

            accumulated_weights_value = info["accumulated_weights_value"].sum(dim=0)
            # accumulated_weights_count = info["accumulated_weights_count"].sum(dim=0)

            # for current camera, if the accumulated_weights_value > gaussians_max_accumulated_weights, update it on the index and apply batch_idx to the gaussians_max_accumulated_weights_index
            gaussians_max_accumulated_weights_index = torch.where(accumulated_weights_value > gaussians_max_accumulated_weights, batch_idx, gaussians_max_accumulated_weights_index)
            gaussians_max_accumulated_weights = torch.where(accumulated_weights_value > gaussians_max_accumulated_weights, accumulated_weights_value, gaussians_max_accumulated_weights)

            # 중간 메모리 정리
            torch.cuda.empty_cache()

        # 누적값 정규화
        weights_count = accumulated_values["weights_count"].clamp(min=1).unsqueeze(-1)
        for key in ["gt_colors", "final_colors", "cur_colors", "colors"]:
            accumulated_values[key] /= weights_count
        accumulated_values["one_minus_alphas"] /= accumulated_values["weights_count"].clamp(min=1)

        # 가우시안 제거 단계
        alive_mask = torch.ones(n_gaussians, dtype=torch.bool, device=device)
        n_alive = n_gaussians
        current_target_gaussians = n_alive - current_gaussians_to_remove

        # # FIX REMOVAL_CHUNK TO 1
        # removal_chunk = 1

        total_to_remove = n_alive - current_target_gaussians
        with tqdm.tqdm(total=total_to_remove, desc="Removing Gaussians") as pbar:
            # 미리 계산된 값들
            alive_indices = alive_mask.nonzero(as_tuple=True)[0]
            n_alive = len(alive_indices)
            
            # 메모리 재사용을 위한 텐서
            distances = torch.empty((n_alive, removal_chunk), device=device)
            similarities = torch.empty((n_alive, removal_chunk), device=device)
            alive_potential_loss_increase = torch.empty(n_alive, device=device)
            
            while n_alive > current_target_gaussians:
                # 살아있는 가우시안 데이터 추출 - 딕셔너리 대신 직접 접근
                alive_colors = accumulated_values["colors"][alive_mask]
                alive_final_colors = accumulated_values["final_colors"][alive_mask]
                alive_cur_colors = accumulated_values["cur_colors"][alive_mask]
                alive_gt_colors = accumulated_values["gt_colors"][alive_mask]
                alive_one_minus_alphas = accumulated_values["one_minus_alphas"][alive_mask]
                
                # 잠재적 색상 계산 - 메모리 효율적으로
                alive_potential_final_colors = (
                    alive_colors + 
                    (alive_final_colors - alive_colors - alive_cur_colors) / 
                    (alive_one_minus_alphas.unsqueeze(-1) + 1e-6)
                ).clamp(min=0, max=1)

                # 손실 계산 - 벡터화
                alive_current_loss = torch.abs(alive_gt_colors - alive_final_colors)
                alive_potential_loss = torch.abs(alive_gt_colors - alive_potential_final_colors)
                alive_potential_loss_increase = (alive_potential_loss - alive_current_loss).sum(dim=1)

                # 제거할 가우시안 선택 - 최적화된 샘플링
                chunk_size = min(removal_chunk, n_alive - gaussians_to_keep)
                if chunk_size <= 0:
                    break
                    
                if sampling:
                    alive_potential_loss_increase -= torch.amin(alive_potential_loss_increase)
                    probs = 1 / (alive_potential_loss_increase + 1e-6)
                    indices_to_remove = torch.multinomial(probs, num_samples=chunk_size, replacement=False)
                else:
                    indices_to_remove = torch.argsort(alive_potential_loss_increase, descending=ascending)[:chunk_size]

                # # 인덱스 범위 체크
                # if indices_to_remove.max() >= len(alive_indices):
                #     print(f"Warning: Invalid index detected. Max index: {indices_to_remove.max()}, Alive indices length: {len(alive_indices)}")
                #     break

                removal_mask = torch.zeros_like(alive_mask)    
                indices_to_remove_global = alive_indices[indices_to_remove]
                removal_mask[indices_to_remove_global] = True
                
                # 각 가우시안의 카메라 인덱스 가져오기
                removed_camera_indices = gaussians_max_accumulated_weights_index[indices_to_remove_global]
                
                # 카메라별 투표 수 계산
                camera_votes = torch.bincount(removed_camera_indices, minlength=len(camera_data["means2ds"]))
                
                # 가장 많은 투표를 받은 카메라 하나만 선택
                cam_idx = camera_votes.argmax()

                # 카메라 데이터 추출
                cam_means2ds = camera_data["means2ds"][cam_idx]
                cam_radiis = camera_data["radiis"][cam_idx]
                
                # 유효한 가우시안만 선택 (반경 > 0)
                valid_mask = cam_radiis > 0
                if not valid_mask.any():
                    continue
                    
                valid_alive_mask = alive_mask & valid_mask
                valid_removed_mask = removal_mask & valid_mask
                
                if not valid_alive_mask.any() or not valid_removed_mask.any():
                    continue
                
                # 거리 계산 - 메모리 효율적으로
                alive_means = cam_means2ds[valid_alive_mask]
                removed_means = cam_means2ds[valid_removed_mask]
                
                # 브로드캐스팅을 활용한 거리 계산
                diff = alive_means.unsqueeze(1) - removed_means.unsqueeze(0)
                distances = (diff * diff).sum(dim=-1)
                
                # 반경 계산 - 벡터화
                alive_radiis = cam_radiis[valid_alive_mask]
                removed_radiis = cam_radiis[valid_removed_mask]
                
                radii_sum = (alive_radiis.unsqueeze(1) + removed_radiis.unsqueeze(0)).pow(2)
                radii_diff = (alive_radiis.unsqueeze(1) - removed_radiis.unsqueeze(0)).pow(2)
                
                # 유사도 계산 - 메모리 재사용
                normalized_distances = ((distances - radii_diff) / (radii_sum - radii_diff + 1e-10)).clamp(min=0, max=1)
                similarities = (1 - normalized_distances) / chunk_size
                
                # 색상 업데이트 - 벡터화 (유효한 가우시안만)
                valid_alive_indices = valid_alive_mask.nonzero(as_tuple=True)[0]
                valid_removed_indices = valid_removed_mask.nonzero(as_tuple=True)[0]
                
                print(accumulated_values["final_colors"].shape, similarities.shape, alive_potential_final_colors.shape, accumulated_values["final_colors"][valid_alive_indices].shape, )
                print(alive_potential_final_colors[valid_removed_indices].shape)
                # 유효한 가우시안에 대해서만 색상 업데이트
                accumulated_values["final_colors"][valid_alive_indices] = (
                    accumulated_values["final_colors"][valid_alive_indices] * (1 - similarities.sum(dim=1)).unsqueeze(-1) + 
                    torch.mm(similarities, alive_potential_final_colors[valid_removed_indices])
                )

                # 마스크 업데이트 - 안전하게
                try:
                    alive_mask[indices_to_remove_global] = False
                    alive_indices = alive_mask.nonzero(as_tuple=True)[0]
                    n_alive = len(alive_indices)
                except RuntimeError as e:
                    print(f"Warning: Error updating mask. Error: {e}")
                    break

                # 진행 상황 업데이트
                pbar.update(chunk_size)
                pbar.set_postfix({
                    'target': current_target_gaussians,
                    'remaining': n_alive
                })

                if n_alive <= 0:
                    raise ValueError("No Gaussians left to remove")

                # 메모리 정리
                torch.cuda.empty_cache()

        # 물리적 가우시안 제거
        pruning_mask = ~alive_mask
        if keep_feats or iterations != 1:
            splats, optimizers = remove_return(
                runner.splats, 
                optimizers,
                runner.strategy_state,
                pruning_mask
            )
        else:
            splats, optimizers = create_splats_and_optimizers_from_data(
                runner.splats["means"][alive_mask],
                sh0_to_rgb(runner.splats["sh0"][alive_mask]),
                runner=runner,
                device=device,
                batch_size=batch_size,
                sparse_grad=sparse_grad,
                visible_adam=visible_adam,
                world_size=world_size,
                init_scale=init_scale,
                scene_scale=scene_scale,
                shN=runner.splats["shN"][alive_mask] if keep_feats else None,
                scales=runner.splats["scales"][alive_mask] if keep_feats else None,
                quats=runner.splats["quats"][alive_mask] if keep_feats else None,
                opacities=runner.splats["opacities"][alive_mask] if keep_feats else None,
                optimizers=optimizers,
                keep_feats=keep_feats,
                indices=alive_mask.nonzero(as_tuple=True)[0],
            )
        runner.splats = splats

    return runner.splats, optimizers


@torch.no_grad()
def simplification_progressive(
    trainset,
    runner,
    parser,
    cfg,
    sampling_factor: float = 0.1,
    keep_sh0: bool = True,
    keep_feats: bool = False,
    batch_size: int = 1,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    world_size: int = 1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    optimizers=None,
    ascending: bool = False,
    use_mean: bool = False,
    sampling: bool = True,
    iterations: int = 1,
    # Synergy (local update) parameters:
    distance_sigma: float = 2e0,
    distance_alpha: float = 1e-1,
    removal_chunk: int = 1000,  # default fallback
    max_removal_iterations: int = 1000,
    chunk_size_alive: int = 1e6,
    chunk_size_removed: int = 5e1,
    trainloader=None,
):
    """
    Progressive simplification of Gaussians:
      1. Compute 'true' potential loss by a full forward pass (across the entire dataset).
      2. Decide how many Gaussians to prune this iteration.
      3. Repeatedly remove small chunks of Gaussians (size = removal_chunk or a dynamic calculation),
         while performing synergy updates to partially reflect that newly removed Gaussians
         have further 'exposed' certain behind Gaussians.
      4. Prune them physically from runner.splats.

    The synergy aggregator is done using a "sum of all removed" approach in the snippet below.
    """
    # Chunk sizes for sub-batches
    chunk_size_alive = int(chunk_size_alive)
    chunk_size_removed = int(chunk_size_removed)


    device = runner.splats["means"].device
    n_gaussian_initial = runner.splats["means"].shape[0]

    # 1) Compute target_n_gaussian
    if sampling_factor > 1:
        # if user provides sampling_factor>1, do nothing
        return runner.splats, optimizers
    target_n_gaussian = int(n_gaussian_initial * sampling_factor)
    if target_n_gaussian < 1:
        target_n_gaussian = 1

    # 2) Overall number to remove
    total_to_remove = n_gaussian_initial - target_n_gaussian

    # 3) How many to remove "per iteration"
    pruned_gaussian_per_iteration = total_to_remove // iterations

    print(f"Initial Gaussians: {n_gaussian_initial}, "
          f"Target: {target_n_gaussian}, "
          f"Removing ~{pruned_gaussian_per_iteration} each iteration, "
          f"for {iterations} iterations.")

    # Some logic for synergy aggregator scale:
    means = runner.splats["means"].detach()  # shape [N, 3]
    # Suppose we compute an average 'dist_avg' from 3-nn
    dist2_avg = (knn(means, 4)[:, 1:] ** 2).mean(dim=-1)  # shape [N]
    dist_avg = torch.sqrt(dist2_avg.clamp(1e-6)).mean().item()
    print(f"dist_avg ~ {dist_avg:.4f} (scene scale)")

    exp_factor = distance_sigma / (dist_avg * dist_avg)

    # Timers
    simplification_start = time.time()

    # ---------------------------------------------------------------
    # Outer Loop: repeated iteration
    # ---------------------------------------------------------------
    for iteration in tqdm.tqdm(range(iterations), desc="ProgressiveSimplify"):
        # (A) Recompute potential_loss
        n_gaussian = runner.splats["means"].shape[0]
        gaussians_to_keep = n_gaussian - pruned_gaussian_per_iteration
        expected_removals = max(0, n_gaussian - gaussians_to_keep)

        # 1) Full forward pass => potential_loss
        # Setup the DataLoader:
        if trainloader is None:
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                persistent_workers=True,
                pin_memory=True,
            )
        trainloader_iter = iter(trainloader)

        potential_loss = torch.zeros(n_gaussian, device=device)
        weights_count = torch.zeros(n_gaussian, device=device)

        # Full pass: accumulate potential_loss
        for _ in tqdm.tqdm(range(len(trainloader)), desc="Compute potential loss"):
            data = next(trainloader_iter)

            pixel = data["image"].to(device) / 255.0
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None

            height, width = pixel.shape[1:3]

            _, _, info = runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=0,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+IW",
                masks=masks,
                gt_image=pixel
            )
            incr_loss = info["accumulated_potential_loss"]
            incr_count = info["accumulated_weights_count"]

            # If shape is [C, N], sum over cameras
            if incr_loss.dim() == 2:
                incr_loss = incr_loss.sum(dim=0)
                incr_count = incr_count.sum(dim=0)

            potential_loss += incr_loss
            weights_count += incr_count

        # Possibly do the averaging
        if use_mean:
            potential_loss /= weights_count.clamp_min(1)

        # Shift up to ensure no negative
        min_val = potential_loss.min()
        if min_val < 0:
            potential_loss -= min_val

        # (B) synergy loop: remove small chunks
        alive_mask = torch.ones(n_gaussian, dtype=torch.bool, device=device)
        means = runner.splats["means"].detach()
        # opacities = torch.sigmoid(runner.splats["opacities"].detach())

        synergy_update = torch.zeros(n_gaussian, device=device)
        n_gaussians_current = n_gaussian

        relative_removal_chunk = max(1, expected_removals // max_removal_iterations)

        # # Ensure all tensors are contiguous
        # potential_loss = potential_loss.contiguous()
        # synergy_update = synergy_update.contiguous()
        # alive_mask = alive_mask.contiguous()

        pbar = tqdm.tqdm(total=expected_removals, desc="Simplification progress")

        while n_gaussians_current > gaussians_to_keep:
            chunk_to_remove = min(relative_removal_chunk, n_gaussians_current - gaussians_to_keep)
            if chunk_to_remove <= 0:
                break

            effective_loss = potential_loss * (1.0 + synergy_update)

            global_alive_indices = alive_mask.nonzero(as_tuple=True)[0]
            # global_alive_indices = global_alive_indices.sort().values
            alive_loss = effective_loss.gather(0, global_alive_indices)

            if sampling:
                chosen_local_idx = torch.multinomial(alive_loss, num_samples=chunk_to_remove, replacement=False)
                # chosen_local_idx = torch.multinomial(alive_loss - alive_loss.amin(), num_samples=chunk_to_remove, replacement=False)
            else:
                _, chosen_local_idx = torch.topk(alive_loss, k=chunk_to_remove, largest=False)

            chosen_global_idx = global_alive_indices[chosen_local_idx]

            # Aggregator synergy approach
            removed_positions = means[chosen_global_idx]
            # removed_opacities = opacities[chosen_global_idx]
            alive_positions = means[global_alive_indices]
            M = alive_positions.shape[0]


            # Outside the loop, create a GradScaler
            # scaler = torch.cuda.amp.GradScaler()
            # with autocast(dtype=torch.float16):
            synergy_total = torch.zeros(M, device=device)
            startA = 0
            while startA < M:
                endA = min(startA + chunk_size_alive, M)
                batch_positions = alive_positions[startA:endA]
                A = batch_positions.shape[0]

                synergy_sub = torch.zeros(A, device=device)

                # Chunk over removed Gaussians
                R = removed_positions.shape[0]
                startR = 0
                while startR < R:
                    endR = min(startR + chunk_size_removed, R)
                    sub_removed_positions = removed_positions[startR:endR]
                    # sub_removed_opacities = removed_opacities[startR:endR]

                    # dist = torch.cdist(batch_positions, sub_removed_positions, p=2)
                    # Replace cdist with squared norm calculation

                    diff = batch_positions.unsqueeze(1) - sub_removed_positions.unsqueeze(0)
                    dist_sq = (diff * diff).sum(dim=2)  # Avoid sqrt until needed
                    # dist = dist_sq.sqrt()  # Only if absolutely required
                
                    factor = 1 / (exp_factor * dist_sq + 1)
                    # alpha_exp = sub_removed_opacities / (exp_factor * dist_sq + 1)
                    # torch.exp(exp_factor * dist)
                    # alpha_exp = sub_removed_opacities * torch.exp(exp_factor * dist)
                    
                    # synergy_sub += alpha_exp.sum(dim=1)
                    # print(factor.sum(dim=1).amin(), factor.sum(dim=1).amax())
                    synergy_sub.add_(factor.sum(dim=1))

                    startR = endR

                synergy_total[startA:endA] = synergy_sub
                startA = endA

            # # If this is part of a larger training loop, use the scaler
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # Update synergy and remove Gaussians
            synergy_update[global_alive_indices] += distance_alpha * synergy_total.float()

            # Remove Gaussians
            alive_mask[chosen_global_idx] = False

            # Update number of Gaussians
            n_gaussians_current -= chunk_to_remove
            pbar.update(chunk_to_remove)
            pbar.set_postfix(removed=chunk_to_remove, syn_min=synergy_update.min().item(), syn_max=synergy_update.max().item(), syn_mean=synergy_update.mean().item())

            # Timing for the block
            n_gaussians_sum = n_gaussians_current

            torch.cuda.empty_cache()

        pbar.close()




        # (C) physical prune
        pruning_mask = ~alive_mask
        if keep_feats or iterations != 1:
            splats, optimizers = remove_return(runner.splats, optimizers,
                                               runner.strategy_state,
                                               pruning_mask)
        else:
            splats, optimizers = create_splats_and_optimizers_from_data(
                runner.splats["means"][alive_mask],
                sh0_to_rgb(runner.splats["sh0"][alive_mask]),
                runner=runner,
                device=device,
                batch_size=batch_size,
                sparse_grad=sparse_grad,
                visible_adam=visible_adam,
                world_size=world_size,
                init_scale=init_scale,
                scene_scale=scene_scale,
                shN=runner.splats["shN"][alive_mask] if keep_feats else None,
                scales=runner.splats["scales"][alive_mask] if keep_feats else None,
                quats=runner.splats["quats"][alive_mask] if keep_feats else None,
                opacities=runner.splats["opacities"][alive_mask] if keep_feats else None,
                optimizers=optimizers,
                keep_feats=keep_feats,
                indices=alive_mask.nonzero(as_tuple=True)[0],
            )
        runner.splats = splats

    n_gaussians_after = runner.splats["means"].shape[0]
    simplification_time = time.time() - simplification_start
    print(f"Simplification: from {n_gaussian_initial} to {n_gaussians_after} gaussians, in {simplification_time:.2f} sec.")
    return splats, optimizers






@torch.no_grad()
def compare_simplifications(
    trainset: Dataset, 
    runner,
    parser,
    cfg,
    sampling_factor: float = 0.1,
    cdf_threshold: float = 0.99,
    use_cdf_mask: bool = False,
    keep_sh0: bool = True,
    keep_feats: bool = False,
    batch_size: int = 1,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    world_size: int = 1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    optimizers=None,
    abs_ratio=False,
    trainloader=None,
):
    if trainloader is None:
        # Get data from first method (simplification)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )

    n_gaussian = runner.splats["means"].shape[0]
    device = runner.splats["means"].device

    # Calculate importance scores for simplification method
    importance_scores = torch.zeros(n_gaussian, device=device)
    pixels_per_gaussian = torch.zeros(n_gaussian, device=device, dtype=torch.int)

    trainloader_iter = iter(trainloader)

    print("Calculating importance scores for simplification method...")
    for step in tqdm.tqdm(range(len(trainloader))):
        data = next(trainloader_iter)

        pixel = data["image"].to(device) / 255.0
        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None

        width, height = pixel.shape[1:3]

        # forward
        _, _, info = runner.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=0,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+IW",
            masks=masks,
        )

        max_ids = info["max_ids"]
        max_ids_valid_mask = max_ids >= 0
        max_ids = max_ids % n_gaussian

        batch_pixels_per_gaussian = torch.zeros_like(pixels_per_gaussian)
        batch_pixels_per_gaussian.index_add_(
            0, max_ids[max_ids_valid_mask].flatten(), torch.ones_like(max_ids[max_ids_valid_mask]).flatten()
        )

        pixels_per_gaussian += batch_pixels_per_gaussian

        accumulated_weights_value = info["accumulated_weights_value"]
        accumulated_weights_count = info["accumulated_weights_count"]

        for i in range(accumulated_weights_value.shape[0]):
            importance_score = accumulated_weights_value[i] / (accumulated_weights_count[i].clamp(min=1))
            accumulated_weights_valid_mask = batch_pixels_per_gaussian > 0
            importance_scores[accumulated_weights_valid_mask] += importance_score[accumulated_weights_valid_mask]
 
    # importance_scores[pixels_per_gaussian == 0] = 0
    
    # Calculate probability for simplification method
    prob_simplification = importance_scores / importance_scores.sum()

    
    # Reset trainloader for second method
    if trainloader is not None:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )
    trainloader_iter = iter(trainloader)
    
    # Calculate accumulated increased losses for simplification_from_mesh_simp method
    accumulated_increased_losses = torch.zeros(n_gaussian, device=device)
    accumulated_weights_counts = torch.zeros(n_gaussian, device=device)

    print("Calculating accumulated increased losses for simplification_from_mesh_simp method...")
    for _ in tqdm.tqdm(range(len(trainloader))):
        data = next(trainloader_iter)

        pixel = data["image"].to(device) / 255.0
        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None

        width, height = pixel.shape[1:3]

        # forward
        _, _, info = runner.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=0,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+IW",
            masks=masks,
            gt_image=pixel
        )

        increased_losses = info["accumulated_potential_loss"]
        accumulated_weights_count = info["accumulated_weights_count"]

        accumulated_increased_losses += (increased_losses.sum(dim=0) / (width * height))
        accumulated_weights_counts += accumulated_weights_count.sum(dim=0)

    # Mean of accumulated increased losses
    # accumulated_increased_losses /= accumulated_weights_counts.clamp(min=1)
    
    # For simplification_from_mesh_simp, we use inverse of accumulated_increased_losses
    # since lower loss means higher importance
    
    print(accumulated_increased_losses.min(), accumulated_increased_losses.max(), accumulated_increased_losses.mean(), accumulated_increased_losses.std())
    print("Zeros: ", (accumulated_increased_losses == 0).sum(), " out of ", accumulated_increased_losses.shape[0])

    # Calculate probability for simplification_from_mesh_simp method
    accumulated_increased_losses = accumulated_increased_losses - torch.amin(accumulated_increased_losses)  # since accumulated_increased_losses can have negative values
    prob_mesh_simp = accumulated_increased_losses / accumulated_increased_losses.sum()
    
    # before plotting, normalize both to have range of [0, 1]
    # prob_simplification = prob_simplification / torch.amax(prob_simplification)
    # prob_mesh_simp = prob_mesh_simp / torch.amax(prob_mesh_simp)

    # Convert to numpy for plotting
    prob_simplification_np = prob_simplification.cpu().numpy()
    prob_mesh_simp_np = prob_mesh_simp.cpu().numpy()
    
    # Plot the correlation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(prob_simplification_np, prob_mesh_simp_np, alpha=0.5, s=0.5)
    plt.xlabel('Probability from simplification')
    plt.ylabel('Probability from simplification_from_mesh_simp')
    plt.title('Correlation between simplification methods')
    plt.xlim(0, np.amax(prob_simplification_np))  # Limit to reasonable range
    plt.ylim(0, np.amax(prob_mesh_simp_np))  # Limit to reasonable range
    
    # Add diagonal line for perfect correlation reference
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.savefig('simplification_comparison.png', dpi=300)
    plt.close()
    
    print(f"Comparison plot saved as 'simplification_comparison.png'")
    
    # Calculate correlation coefficient
    from scipy.stats import pearsonr
    
    # Filter out zeros for better correlation analysis
    mask = (prob_simplification_np > 0) & (prob_mesh_simp_np > 0)
    if mask.sum() > 1:  # Need at least 2 points for correlation
        correlation, p_value = pearsonr(prob_simplification_np, prob_mesh_simp_np)
        print(f"Pearson correlation coefficient: {correlation:.4f} (p-value: {p_value:.4e})")
    else:
        print("Not enough non-zero values to calculate correlation")
    
    return {
        "prob_simplification": prob_simplification,
        "prob_mesh_simp": prob_mesh_simp,
        "correlation_plot_path": "simplification_comparison.png"
    }


@torch.no_grad()
def depth_reinitialization(
        trainset: Dataset,
        num_depth: int,
        runner,
        parser,
        cfg,
        batch_size: int = 1,
        sparse_grad: bool = False,
        visible_adam: bool = False,
        world_size: int = 1,
        init_scale: float = 1.0,
        init_extent: float = 3.0,
        scene_scale: float = 1.0,
        init_num_pts: int = 100_000,
        device: str = "cuda",
        init_type: str = "sfm",
        optimizers=None,
        replace=False,
        trainloader=None,
):
    
    # if trainloader is None:
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
    )

    out_pts_list = []
    gt_list = []
    
    trainloader_iter = iter(trainloader)

    # Training loop.
    pbar = tqdm.tqdm(range(len(trainloader)))
    for step in pbar:
        data = next(trainloader_iter)

        pixel = data["image"].to(device) / 255.0  # [1, H, W, 3]
        camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
    
        width, height = pixel.shape[1:3]
        
        # forward
        renders, alphas, info = runner.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=0,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED+IW",
            masks=masks,
        )

        # depths = renders[..., -1:]
        depths = info["max_weight_depth"] # [b, H, W, 1]
        world_coords = depth_to_points(depths=depths, Ks=Ks, camtoworlds=camtoworlds)  # [b, H, W, 3]
        
        prob = alphas # shape of [b, H, W, 1]
        # prob = 1 - alphas # shape of [b, H, W, 1]

        # if info["max_ids"] < 0: prob = 0
        # max_ids = info["max_ids"] # shape of [b, H, W, 1]
        # prob[max_ids < 0] = 0

        # prob to zero where depths < 0
        prob[depths < 0] = 0

        prob = prob / prob.sum() 
        # print(prob.shape, depths.shape, world_coords.shape, alphas.shape)
        prob = prob.reshape(-1).cpu().numpy() 

        factor = 1 / (width * height * len(trainloader) / num_depth)

        N_xyz = prob.shape[0]
        num_sampled = min(N_xyz, int(N_xyz * factor))

        indices = np.random.choice(N_xyz, num_sampled, p=prob, replace=replace)
        if replace: 
            indices = np.unique(indices)
        

        world_coords = world_coords.reshape(-1, 3)
        gt = pixel.reshape(-1, 3)

        out_pts_list.append(world_coords[indices])
        gt_list.append(gt[indices])
    

    out_pts_merged = torch.cat(out_pts_list, dim=0)
    gt_merged = torch.cat(gt_list, dim=0)

    splats, optimizers = create_splats_and_optimizers_from_data(
        out_pts_merged,
        gt_merged,
        runner=runner,
        device=device,
        batch_size=batch_size,
        sparse_grad=sparse_grad,
        visible_adam=visible_adam,
        world_size=world_size,
        init_scale=init_scale,
        scene_scale=scene_scale,
        optimizers=optimizers
    )

    return splats, optimizers


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def sh0_to_rgb(sh0: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return sh0 * C0 + 0.5

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


def visualize_id_maps(max_ids: torch.Tensor) -> torch.Tensor:
    B, H, W = max_ids.shape
    output = torch.zeros((B, H, W, 3), dtype=torch.float32, device=max_ids.device)
    
    r_value = max_ids % 256
    g_value = (max_ids // 256) % 256
    b_value = (max_ids // (256 * 256)) % 256
    
    output[:, :, :, 0] = r_value / 255.0
    output[:, :, :, 1] = g_value / 255.0
    output[:, :, :, 2] = b_value / 255.0

    return output


@torch.no_grad()
def recover_covariance_2d(conics: torch.Tensor) -> torch.Tensor:
    """2D 공분산 행렬을 conics로부터 복원합니다.
    
    Args:
        conics: [N, 3] 형태의 텐서, 각 행은 [a, b, c] 형태로 
               covar2d_inv의 상삼각 부분을 저장
               
    Returns:
        [N, 2, 2] 형태의 2D 공분산 행렬
    """
    # 2x2 행렬의 역행렬 공식 사용
    det = conics[:, 0] * conics[:, 2] - conics[:, 1] * conics[:, 1]
    zero_mask = torch.abs(det) < 1e-8
    
    # 브로드캐스팅을 활용한 한 번의 나눗셈
    conics_div_det = conics / det.unsqueeze(-1)
    
    # 한 번의 메모리 할당으로 cov2d 생성
    cov2d = torch.stack([
        torch.stack([conics_div_det[:, 2], -conics_div_det[:, 1]], dim=1),
        torch.stack([-conics_div_det[:, 1], conics_div_det[:, 0]], dim=1)
    ], dim=1)
    
    return cov2d, zero_mask, det


@torch.no_grad()
def update_final_colors_with_correlation(
    means2ds, radiis,
    accumulated_final_colors, removed_potential_colors,
    indices_to_remove, alive_mask,
    distance_sigma=2e0, distance_alpha=1e-1,
    num_camera=1,
):
    """KL 발산 기반 최종 색상 업데이트 함수"""
    device = means2ds.device
    N = means2ds.shape[1]  # 가우시안 수
    n_remove = len(indices_to_remove)
    n_alive = torch.sum(alive_mask)
    
    # 모든 카메라의 유사도 누적
    # similarities_sum = torch.zeros(n_alive, device=device)
    
    # 카메라별로 처리하되 내부 연산은 최적화
    for cam_idx in num_camera:
        # 현재 카메라의 데이터 추출
        cam_means2ds = means2ds[cam_idx]  # [N, 2]
        cam_radiis = radiis[cam_idx]  # [N]

        # 살아있는 가우시안과 제거할 가우시안의 데이터
        # alive_means = cam_means2ds[alive_mask]  # [n_alive, 2]
        # removed_means = cam_means2ds[indices_to_remove]  # [n_remove, 2]
        alive_radiis = cam_radiis[alive_mask]  # [n_alive]
        removed_radiis = cam_radiis[indices_to_remove]  # [n_remove]

        # 거리 계산 [n_alive, n_remove]
        distances = torch.pow(
            cam_means2ds[alive_mask].unsqueeze(1) - cam_means2ds[indices_to_remove].unsqueeze(0), 
            2
        ).sum(dim=-1)

        # 반경 관련 계산 [n_alive, n_remove]
        radii_sum = (alive_radiis.unsqueeze(1) + removed_radiis.unsqueeze(0)).pow(2).sum(dim=-1)
        radii_diff = (torch.abs(alive_radiis.unsqueeze(1) - removed_radiis.unsqueeze(0))).pow(2).sum(dim=-1)

        # 정규화된 거리와 유사도 계산
        normalized_distances = ((distances - radii_diff) / (radii_sum - radii_diff + 1e-10)).clamp(min=0, max=1)
        similarities = (1 - normalized_distances) / (len(num_camera) * n_remove)
        
        # 유사도 누적
        # similarities_sum = similarities.sum(dim=1)

        # 중간 텐서 메모리 해제
        # del distances, radii_sum, radii_diff, normalized_distances
        torch.cuda.empty_cache()

        # 색상 보간 - 한 번에 처리
        accumulated_final_colors[alive_mask] = (
            accumulated_final_colors[alive_mask] * (1 - similarities.sum(dim=1)).unsqueeze(-1) + 
            torch.mm(similarities, removed_potential_colors)
        )

    return accumulated_final_colors


@torch.no_grad()
def calculate_gaussian_kl_batch(means2ds1, conics1, means2ds2, conics2, alive_valid_mask, removed_valid_mask):
    """가우시안 간의 KL 발산을 계산하는 함수
    
    Args:
        means2ds1: 첫 번째 가우시안들의 2D 평균 위치 [N1, 2]
        conics1: 첫 번째 가우시안들의 conic 행렬 [N1, 3] (공분산 행렬의 역행렬)
        means2ds2: 두 번째 가우시안들의 2D 평균 위치 [N2, 2]
        conics2: 두 번째 가우시안들의 conic 행렬 [N2, 3] (공분산 행렬의 역행렬)
        alive_valid_mask: 유효한 생존 가우시안의 마스크
        removed_valid_mask: 유효한 제거 가우시안의 마스크
        
    Returns:
        KL 발산 값 [N1, N2]
    """
    N1, _ = means2ds1.shape
    N2, _ = means2ds2.shape
    similarity = torch.zeros(N1, N2, device=means2ds1.device)
    
    # 유효한 인덱스에 대해서만 계산
    valid_means2ds1 = means2ds1[alive_valid_mask]
    valid_means2ds2 = means2ds2[removed_valid_mask]
    valid_conics1 = conics1[alive_valid_mask]
    valid_conics2 = conics2[removed_valid_mask]
    
    # n_valid1 = len(alive_valid_indices)
    # n_valid2 = len(removed_valid_indices)

    # 2D 공분산 행렬 복원과 행렬식 계산을 한 번에
    det1 = valid_conics1[:, 0] * valid_conics1[:, 2] - valid_conics1[:, 1] * valid_conics1[:, 1]
    det2 = valid_conics2[:, 0] * valid_conics2[:, 2] - valid_conics2[:, 1] * valid_conics2[:, 1]
    zero_mask = torch.logical_or(
        (torch.abs(det1) < 1e-8).unsqueeze(1),
        (torch.abs(det2) < 1e-8).unsqueeze(0)
    )
    det_cov1 = 1 / det1
    det_cov2 = 1 / det2
    
    # conics1을 공분산 행렬로 변환
    cov1 = torch.stack([
        torch.stack([valid_conics1[:, 2] / det1, -valid_conics1[:, 1] / det1], dim=1),
        torch.stack([-valid_conics1[:, 1] / det1, valid_conics1[:, 0] / det1], dim=1)
    ], dim=1)  # [n_valid1, 2, 2]
    
    # 차이 계산 [n_valid1, n_valid2, 2]
    diff = valid_means2ds1.unsqueeze(1) - valid_means2ds2.unsqueeze(0)
    
    # 마할라노비스 항 계산 최적화
    # (x-y)^T Σ^-1 (x-y) = (x-y)^T [a b; b c] (x-y)
    # = a(x1-y1)^2 + 2b(x1-y1)(x2-y2) + c(x2-y2)^2
    diff_sq = diff * diff  # [n_valid1, n_valid2, 2]
    mahalanobis_term = (
        valid_conics2[:, 0].unsqueeze(0) * diff_sq[..., 0] +  # a(x1-y1)^2
        2 * valid_conics2[:, 1].unsqueeze(0) * diff[..., 0] * diff[..., 1] +  # 2b(x1-y1)(x2-y2)
        valid_conics2[:, 2].unsqueeze(0) * diff_sq[..., 1]  # c(x2-y2)^2
    )
    
    # 대각합 항 계산 최적화
    # tr(Σ2^-1 Σ1) = tr([a2 b2; b2 c2] [a1 b1; b1 c1])
    # = a2*a1 + 2*b2*b1 + c2*c1
    trace_term = (
        valid_conics2[:, 0].unsqueeze(0) * cov1[:, 0, 0].unsqueeze(1) +  # a2*a1
        2 * valid_conics2[:, 1].unsqueeze(0) * cov1[:, 0, 1].unsqueeze(1) +  # 2*b2*b1
        valid_conics2[:, 2].unsqueeze(0) * cov1[:, 1, 1].unsqueeze(1)  # c2*c1
    )
    
    # 로그 행렬식 항 계산
    log_det_term = torch.log((det_cov2.unsqueeze(0) / det_cov1.unsqueeze(1)).abs().clamp(min=1e-10))
    
    # 최종 KL 발산 계산
    kl_div = 0.5 * (trace_term + mahalanobis_term - 2 + log_det_term)
    kl_div = kl_div.clamp(min=0)

    valid_similarities = torch.exp(-kl_div)
    valid_similarities[zero_mask] = 0

    valid_radiis = torch.logical_and(alive_valid_mask.unsqueeze(1), removed_valid_mask.unsqueeze(0))

    # 유효한 인덱스에 대해서만 결과 저장
    similarity[valid_radiis] = valid_similarities

    return similarity

