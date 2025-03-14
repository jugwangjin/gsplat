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
):
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
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
    use_mean = False,
    sampling = False,
):
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
    )

    n_gaussian = runner.splats["means"].shape[0]

    importance_scores = torch.zeros(n_gaussian, device=runner.splats["means"].device)
    pixels_per_gaussian = torch.zeros(n_gaussian, device=runner.splats["means"].device, dtype=torch.int)

    trainloader_iter = iter(trainloader)

    accumulated_increased_losses = torch.zeros(n_gaussian, device=runner.splats["means"].device)
    accumulated_weights_counts = torch.zeros(n_gaussian, device=runner.splats["means"].device)

    for _ in tqdm.tqdm(range(len(trainloader))):
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
            gt_image=pixel
        )

        increased_losses = info["accumulated_potential_loss"]
        accumulated_weights_count = info["accumulated_weights_count"]

        accumulated_increased_losses += (increased_losses.sum(dim=0) / (width * height))
        accumulated_weights_counts += accumulated_weights_count.sum(dim=0)

        if torch.isnan(increased_losses).any():
            exit()

    # Add fixed pruning indices: that accumulated_increased_losses < 0
    fixed_pruning_indices = accumulated_increased_losses < 0
    fixed_pruning_indices = torch.nonzero(fixed_pruning_indices, as_tuple=True)[0]

    accumulated_increased_losses = accumulated_increased_losses - torch.amin(accumulated_increased_losses)  # since accumulated_increased_losses can have negative values

    if use_mean:
        accumulated_increased_losses /= accumulated_weights_counts.clamp(min=1)

    # indices to keep
    if not sampling:
        indices = torch.argsort(accumulated_increased_losses, descending=False if ascending else True)[:int(n_gaussian * sampling_factor)]
        
    else:
        # same as simplification - making prob 
        # inverse_losses = 1.0 / (accumulated_increased_losses + 1e-10)  # Add small epsilon to avoid division by zero

        prob_mesh_simp = accumulated_increased_losses / accumulated_increased_losses.sum()
        prob_mesh_simp = prob_mesh_simp.cpu().numpy()

        n_sample = int(n_gaussian * sampling_factor)

        indices = np.random.choice(n_gaussian, n_sample, p=prob_mesh_simp, replace=False)
    
    # remove fixed pruning indices
    # the indices is the indices for keeping
    indices = [i for i in indices if i not in fixed_pruning_indices]

    # indices_ = torch.argsort(accumulated_increased_losses, descending=False)[:int(n_gaussian * sampling_factor)]
    # print("descending", torch.sigmoid(runner.splats["opacities"][indices_]).min(), torch.sigmoid(runner.splats["opacities"][indices_]).max(),\
    #     torch.sigmoid(runner.splats["opacities"][indices_]).mean(), torch.sigmoid(runner.splats["opacities"][indices_]).std())
    # indices_ = torch.argsort(accumulated_increased_losses, descending=True)[:int(n_gaussian * sampling_factor)]
    # print("ascending", torch.sigmoid(runner.splats["opacities"][indices_]).min(), torch.sigmoid(runner.splats["opacities"][indices_]).max(),\
    #     torch.sigmoid(runner.splats["opacities"][indices_]).mean(), torch.sigmoid(runner.splats["opacities"][indices_]).std())

    if keep_feats:
        # simply prune the gaussians
        # pruning mask has value True where we want to prune
        pruning_mask = torch.zeros(n_gaussian, device=runner.splats["means"].device)
        pruning_mask[indices] = 1
        pruning_mask = 1 - pruning_mask
        
        pruning_mask = pruning_mask.bool()

        splats, optimizers = remove_return(runner.splats, optimizers, runner.strategy_state, pruning_mask)
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
    abs_ratio=False
):
    # Get data from first method (simplification)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
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
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
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
        correlation, p_value = pearsonr(prob_simplification_np[mask], prob_mesh_simp_np[mask])
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
        replace=False
):
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
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



# def visualize_id_maps(max_ids: torch.Tensor) -> torch.Tensor:
#     B, H, W = max_ids.shape
#     output = torch.zeros((B, 3, H, W), dtype=torch.float32)
    
#     # Define color palette (RGB values) with about 30 colors
#     colors = torch.tensor([
#         [0.12, 0.47, 0.71],  # blue
#         [0.98, 0.50, 0.45],  # red
#         [0.17, 0.63, 0.17],  # green
#         [0.58, 0.40, 0.74],  # purple
#         [0.55, 0.34, 0.29],  # brown
#         [0.89, 0.47, 0.76],  # pink
#         [0.74, 0.74, 0.13],  # yellow
#         [0.09, 0.75, 0.81],  # cyan
#         [0.40, 0.40, 0.40],  # gray
#         [0.91, 0.54, 0.76],  # light pink
#         [0.65, 0.81, 0.89],  # light blue
#         [0.99, 0.75, 0.44],  # orange
#         [0.68, 0.92, 0.67],  # light green
#         [0.84, 0.66, 0.87],  # lavender
#         [0.75, 0.55, 0.53],  # tan
#         [0.98, 0.78, 0.95],  # light magenta
#         [0.94, 0.94, 0.55],  # pale yellow
#         [0.55, 0.87, 0.87],  # aqua
#         [0.75, 0.75, 0.75],  # silver
#         [1.00, 0.60, 0.60],  # salmon
#         [0.60, 1.00, 0.60],  # light lime
#         [0.60, 0.60, 1.00],  # periwinkle
#         [0.80, 0.80, 0.40],  # olive
#         [0.40, 0.80, 0.80],  # teal
#         [0.35, 0.70, 0.90],  # sky blue
#         [0.90, 0.35, 0.70],  # hot pink
#         [0.70, 0.90, 0.35],  # lime green
#         [0.35, 0.90, 0.70],  # sea green
#         [0.90, 0.70, 0.35],  # amber
#         [0.70, 0.35, 0.90],  # violet
#     ], dtype=torch.float32)
    
#     for b in range(B):
#         # Cluster similar regions
#         unique_ids = torch.unique(max_ids[b])
#         print(len(unique_ids))
#         color_mapping = unique_ids % len(colors)
#         print(color_mapping)
#         # Assign colors
#         for idx, color_idx in enumerate(color_mapping):
#             mask = (max_ids[b] == unique_ids[idx])
#             output[b, :, mask] = colors[color_idx].view(3, 1)
#         print("done")
#     return output