import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, visualize_id_maps, depth_reinitialization, simplification
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy, DistillationStrategy, Distill2DStrategy
from gsplat.optimizers import SelectiveAdam
from gsplat.utils import save_ply

import math
import random

from teacher_trainer import Runner as TeacherRunner
from teacher_trainer import create_splats_with_optimizers, fix_all_seeds

def fix_all_seeds(seed: int = 42):
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU and CUDA seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For hash-based functions in Python (like set or dict order)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # # Make cuDNN deterministic and disable benchmark to ensure reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

fix_all_seeds(20202464)


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    use_novel_view: bool = True
    novel_view_weight: float = 0.25
    teacher_ckpt: str = None
    # The teacher's sampling ratio. If 0.1, the 10% of the teacher's GSs are used for the student.
    start_sampling_ratio: float = 0.75

    target_sampling_ratio: float = 0.9

    num_cycles: int = 1

    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    use_abs_ratio: bool = False

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 10_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [10_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [10_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [10_000])


    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy, DistillationStrategy, Distill2DStrategy] = field(
        default_factory=Distill2DStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss

    distill_colors_lambda: float = 0
    distill_depth_lambda: float = 0
    distill_xyzs_lambda: float = 0    
    distill_quats_lambda: float = 0    
    distill_scales_lambda: float = 0
    distill_opacities_lambda: float = 0
    distill_sh_lambda: float = 0
    
    


    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy) or isinstance(strategy, Distill2DStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, DistillationStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
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
    return splats, optimizers


class Runner(TeacherRunner):
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:

        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # assert teacher_ckpt is not in the subdirectory of cfg.result_dir - to avoid overwriting
        if cfg.teacher_ckpt is not None:
            assert not os.path.commonpath([cfg.teacher_ckpt, cfg.result_dir]) == cfg.result_dir, "teacher_ckpt is in the subdirectory of result_dir"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )

        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None

        # load teacher checkpoint
        teacher_ckpt = torch.load(cfg.teacher_ckpt, map_location="cpu")
        self.teacher_splats = {}
        for k in teacher_ckpt["splats"].keys():
            self.teacher_splats[k] = teacher_ckpt["splats"][k].to(self.device).data


        self.splats = {}
        for k in teacher_ckpt["splats"].keys():
            self.splats[k] = torch.nn.Parameter(teacher_ckpt["splats"][k].to(self.device).data)

        target_n_gaussians = int(cfg.start_sampling_ratio * len(self.teacher_splats["means"]))
        current_n_gaussians = len(self.splats["means"])
        sampling_factor = min(1, target_n_gaussians / current_n_gaussians)


        print(target_n_gaussians, current_n_gaussians, sampling_factor)
        
        self.splats, self.optimizers = simplification(
            trainset = self.trainset,
            runner = self,
            parser = self.parser,
            cfg = self.cfg,
            sampling_factor = sampling_factor,
            cdf_threshold = 0.99,
            use_cdf_mask = False,
            keep_sh0 = True,
            keep_feats = True,
            batch_size=cfg.batch_size,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            world_size=world_size,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            abs_ratio = cfg.use_abs_ratio,
        )

        self.target_num_gaussians = int(cfg.target_sampling_ratio * len(self.teacher_splats["means"]))

        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy) or isinstance(self.cfg.strategy, Distill2DStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale,  target_num_gaussians=self.target_num_gaussians
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        elif isinstance(self.cfg.strategy, DistillationStrategy):   
            self.strategy_state = self.cfg.strategy.initialize_state()
            self.cfg.strategy.teacher_sampling_ratio = self.cfg.teacher_sampling_ratio
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")


        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        cfg.distill = cfg.distill_colors_lambda > 0.0 or \
                        cfg.distill_depth_lambda > 0.0 or \
                        cfg.distill_xyzs_lambda > 0.0 or \
                        cfg.distill_quats_lambda > 0.0 or \
                        cfg.distill_scales_lambda > 0.0 or \
                        cfg.distill_opacities_lambda > 0.0 or \
                        cfg.distill_sh_lambda > 0.0 
                        # cfg.distill_sh0_lambda > 0.0 or \
                        # cfg.distill_shN_lambda > 0.0


    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )


        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()

        for cycle in range(cfg.num_cycles):

            pbar = tqdm.tqdm(range(init_step, max_steps))
            for step in pbar:
                if not cfg.disable_viewer:
                    while self.viewer.state.status == "paused":
                        time.sleep(0.01)
                    self.viewer.lock.acquire()
                    tic = time.time()
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)

                camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
                Ks = data["K"].to(device)  # [1, 3, 3]
                pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
                num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                )
                image_ids = data["image_id"].to(device)
                masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
                
                height, width = pixels.shape[1:3]

                # sh schedule
                sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

                if cfg.pose_noise:
                    camtoworlds = self.pose_perturb(camtoworlds, image_ids)

                if cfg.pose_opt:
                    camtoworlds = self.pose_adjust(camtoworlds, image_ids)

                render_mode = "NS" if cfg.distill and cfg.distill_sh_lambda == 0 else "F" if cfg.distill else "RGB+ED+IW"
                

                with torch.no_grad():
                    teacher_renders, _, _ = self.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=image_ids,
                        render_mode=render_mode,
                        masks=masks,
                        splats = self.teacher_splats
                    )
                    teacher_rgb = teacher_renders[..., 0:3].detach().data # [1, H, W, 3]
                    teacher_xyzs = teacher_renders[..., 3:6].detach().data # [1, H, W, 3]
                    teacher_quats = teacher_renders[..., 6:10].detach().data # [1, H, W, 4]
                    teacher_scales = teacher_renders[..., 10:13].detach().data # [1, H, W, k*3]
                    teacher_opacities = teacher_renders[..., 13:14].detach().data # [1, H, W, 1]
                    teacher_sh = teacher_renders[..., 14:-1].detach().data # [1, H, W, k*3]
                    teacher_depths = teacher_renders[..., -1:].detach().data # [1, H, W, 1]

                    del teacher_renders
        
                if cfg.use_novel_view:

                    data_ = self.trainset.sample_novel_views(batch_size=cfg.batch_size)
                    Ks_ = data_["K"].to(device)  # [1, 3, 3]
                    camtoworlds_ = data_["camtoworld"].to(device)
                    image_ids_ = data_["image_id"].to(device)

                    with torch.no_grad():
                        teacher_renders, _, _ = self.rasterize_splats(
                            camtoworlds=camtoworlds_,
                            Ks=Ks_,
                            width=width,
                            height=height,
                            sh_degree=sh_degree_to_use,
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                            image_ids=image_ids_,
                            render_mode=render_mode,
                            masks=masks,
                            splats = self.teacher_splats
                        )
                        teacher_rgb_ = teacher_renders[..., 0:3].detach().data # [1, H, W, 3]
                        teacher_xyzs_ = teacher_renders[..., 3:6].detach().data # [1, H, W, 3]
                        teacher_quats_ = teacher_renders[..., 6:10].detach().data # [1, H, W, 4]
                        teacher_scales_ = teacher_renders[..., 10:13].detach().data # [1, H, W, k*3]
                        teacher_opacities_ = teacher_renders[..., 13:14].detach().data # [1, H, W, 1]
                        teacher_sh_ = teacher_renders[..., 14:-1].detach().data # [1, H, W, k*3]
                        teacher_depths_ = teacher_renders[..., -1:].detach().data # [1, H, W, 1]

                        del teacher_renders

                        pixels_ = teacher_rgb_

                        pixels = torch.cat([pixels, pixels_], dim=0).detach().data

                        teacher_rgb = torch.cat([teacher_rgb, teacher_rgb_], dim=0)
                        teacher_xyzs = torch.cat([teacher_xyzs, teacher_xyzs_], dim=0)
                        teacher_quats = torch.cat([teacher_quats, teacher_quats_], dim=0)
                        teacher_scales = torch.cat([teacher_scales, teacher_scales_], dim=0)
                        teacher_opacities = torch.cat([teacher_opacities, teacher_opacities_], dim=0)
                        teacher_sh = torch.cat([teacher_sh, teacher_sh_], dim=0)
                        teacher_depths = torch.cat([teacher_depths, teacher_depths_], dim=0)

        
                    Ks = torch.cat([Ks, Ks_], dim=0)
                    camtoworlds = torch.cat([camtoworlds, camtoworlds_], dim=0)
                    image_ids = torch.cat([image_ids, image_ids_], dim=0)

                # forward
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode=render_mode,
                    masks=masks,
                )

                colors = renders[..., 0:3]
                depths = renders[..., -1:] if renders.shape[-1] > 3 else None
                if cfg.distill:
                    xyzs = renders[..., 3:6]
                    quats = renders[..., 6:10]
                    scales = renders[..., 10:13]
                    opacities = renders[..., 13:14]
                    sh = renders[..., 14:-1]


                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

                if cfg.use_bilateral_grid:
                    grid_y, grid_x = torch.meshgrid(
                        (torch.arange(height, device=self.device) + 0.5) / height,
                        (torch.arange(width, device=self.device) + 0.5) / width,
                        indexing="ij",
                    )
                    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                    colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)

                # if novel_view, multiply the novel view dimensions by 0.2, to reduce the weight of the novel view
                if cfg.use_novel_view:
                    colors[-cfg.batch_size:] *= cfg.novel_view_weight
                    depths[-cfg.batch_size:] *= cfg.novel_view_weight

                    pixels[-cfg.batch_size:] *= cfg.novel_view_weight
                    teacher_rgb[-cfg.batch_size:] *= cfg.novel_view_weight
                    teacher_depths[-cfg.batch_size:] *= cfg.novel_view_weight

                    if cfg.distill:
                        xyzs[-cfg.batch_size:] *= cfg.novel_view_weight
                        quats[-cfg.batch_size:] *= cfg.novel_view_weight
                        sh[-cfg.batch_size:] *= cfg.novel_view_weight

                        teacher_xyzs[-cfg.batch_size:] *= cfg.novel_view_weight
                        teacher_quats[-cfg.batch_size:] *= cfg.novel_view_weight
                        teacher_sh[-cfg.batch_size:] *= cfg.novel_view_weight

                    # loss
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
                if cfg.distill:
                    colorloss = F.l1_loss(colors, teacher_rgb) * cfg.distill_colors_lambda
                    loss += colorloss

                    xyzloss = F.l1_loss(xyzs, teacher_xyzs) * cfg.distill_xyzs_lambda 
                    loss += xyzloss

                    mask = (depths > 0) & (teacher_depths > 0)
                    depths = torch.where(mask, depths, torch.zeros_like(depths))
                    teacher_depths = torch.where(mask, teacher_depths, torch.zeros_like(teacher_depths))
                    depthloss = F.l1_loss(depths, teacher_depths) * cfg.distill_depth_lambda
                    loss += depthloss

                    quatloss = F.l1_loss(quats, teacher_quats) * cfg.distill_quats_lambda
                    loss += quatloss

                    scaleloss = F.l1_loss(scales, teacher_scales) * cfg.distill_scales_lambda
                    loss += scaleloss

                    opacityloss = F.l1_loss(opacities, teacher_opacities) * cfg.distill_opacities_lambda
                    loss += opacityloss

                    shloss = F.l1_loss(sh.sum(dim=-1) /3., teacher_sh.sum(dim=-1)/3.) * cfg.distill_sh_lambda
                    loss += shloss

                if cfg.use_bilateral_grid:
                    tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                    loss += tvloss

                if cfg.use_novel_view:
                    loss = loss * (cfg.novel_view_weight + 1) # to balance the loss of the novel view

                # regularizations
                if cfg.opacity_reg > 0.0:
                    loss = (
                        loss
                        + cfg.opacity_reg
                        * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                    )
                if cfg.scale_reg > 0.0:
                    loss = (
                        loss
                        + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                    )

                loss.backward()
                

                desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
                if cfg.pose_opt and cfg.pose_noise:
                    # monitor the pose error if we inject noise
                    pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                    desc += f"pose err={pose_err.item():.6f}| "
                if cfg.distill:
                    if cfg.distill_colors_lambda > 0.0:
                        desc += f"col={colorloss.item():.3f}| "
                    if cfg.distill_depth_lambda > 0.0:
                        desc += f"dep={depthloss.item():.3f}| "
                    if cfg.distill_xyzs_lambda > 0.0:
                        desc += f"xyz={xyzloss.item():.3f}| "
                    if cfg.distill_quats_lambda > 0.0:
                        desc += f"qua={quatloss.item():.3f}| "
                    if cfg.distill_scales_lambda > 0.0:
                        desc += f"sca={scaleloss.item():.3f}| "
                    if cfg.distill_opacities_lambda > 0.0:
                        desc += f"op={opacityloss.item():.3f}| "
                    if cfg.distill_sh_lambda > 0.0:
                        desc += f"sh={shloss.item():.3f}| "
                key_for_gradient = self.cfg.strategy.key_for_gradient
                desc += f"key = {key_for_gradient[:3]}|"

                pbar.set_description(desc)

                if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    self.writer.add_scalar("train/loss", loss.item(), step)
                    self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                    self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                    self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                    self.writer.add_scalar("train/mem", mem, step)
                    if cfg.depth_loss:
                        self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                    if cfg.use_bilateral_grid:
                        self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                    if cfg.tb_save_image:
                        canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                        canvas = canvas.reshape(-1, *canvas.shape[2:])
                        self.writer.add_image("train/render", canvas, step)
                    self.writer.flush()

                # Turn Gradients into Sparse Tensor before running optimizer
                if cfg.sparse_grad:
                    assert cfg.packed, "Sparse gradients only work with packed mode."
                    gaussian_ids = info["gaussian_ids"]
                    for k in self.splats.keys():
                        grad = self.splats[k].grad
                        if grad is None or grad.is_sparse:
                            continue
                        self.splats[k].grad = torch.sparse_coo_tensor(
                            indices=gaussian_ids[None],  # [1, nnz]
                            values=grad[gaussian_ids],  # [nnz, ...]
                            size=self.splats[k].size(),  # [N, ...]
                            is_coalesced=len(Ks) == 1,
                        )

                if cfg.visible_adam:
                    gaussian_cnt = self.splats.means.shape[0]
                    if cfg.packed:
                        visibility_mask = torch.zeros_like(
                            self.splats["opacities"], dtype=bool
                        )
                        visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                    else:
                        visibility_mask = (info["radii"] > 0).any(0)

                # Run post-backward steps after backward and optimizer
                if isinstance(self.cfg.strategy, DefaultStrategy) or isinstance(self.cfg.strategy, Distill2DStrategy):

                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )

                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
                elif isinstance(self.cfg.strategy, DistillationStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                        teacher_splats=self.teacher_splats,
                        trainset=self.trainset,
                        runner=self,
                        cfg=self.cfg
                    )
                else:
                    assert_never(self.cfg.strategy)

                # optimize
                for optimizer in self.optimizers.values():
                    if cfg.visible_adam:
                        optimizer.step(visibility_mask)
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.pose_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.app_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.bil_grid_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()

                if step % 50 == 0:
                    torch.cuda.empty_cache()

            total_step = step + cycle * max_steps
            # eval the full set
            self.eval(total_step, include_ids=True)
            self.render_traj(total_step)


            mem = torch.cuda.max_memory_allocated() / 1024**3
            stats = {
                "mem": mem,
                "ellipse_time": time.time() - global_tic,
                "num_GS": len(self.splats["means"]),
            }
            print("Step: ", total_step, stats)
            with open(
                f"{self.stats_dir}/train_step{total_step:04d}_rank{self.world_rank}.json",
                "w",
            ) as f:
                json.dump(stats, f)
            data = {"step": total_step, "splats": self.splats.state_dict()}
            if cfg.pose_opt:
                if world_size > 1:
                    data["pose_adjust"] = self.pose_adjust.module.state_dict()
                else:
                    data["pose_adjust"] = self.pose_adjust.state_dict()
            if cfg.app_opt:
                if world_size > 1:
                    data["app_module"] = self.app_module.module.state_dict()
                else:
                    data["app_module"] = self.app_module.state_dict()
            torch.save(
                data, f"{self.ckpt_dir}/ckpt_{total_step}_rank{self.world_rank}.pt"
            )
            
            if (
                cfg.save_ply
            ):
                rgb = None
                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0)

                save_ply(self.splats, f"{self.ply_dir}/point_cloud_{total_step}.ply", rgb)

                if not cfg.disable_viewer:
                    self.viewer.lock.release()
                    num_train_steps_per_sec = 1.0 / (time.time() - tic)
                    num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                    )
                    # Update the viewer state.
                    self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                    # Update the scene.
                    self.viewer.update(total_step, num_train_rays_per_step)

            # it this was not a last cycle, simplify the model

            if cycle < cfg.num_cycles - 1:

                sampling_n_gaussians = int(cfg.start_sampling_ratio * len(self.splats["means"]))
                target_n_gaussians = int(cfg.target_sampling_ratio * len(self.splats["means"]))
                current_n_gaussians = len(self.splats["means"])
                sampling_factor = min(1, sampling_n_gaussians / current_n_gaussians)


                print(target_n_gaussians, current_n_gaussians, sampling_factor)
                
                self.splats, self.optimizers = simplification(
                    trainset = self.trainset,
                    runner = self,
                    parser = self.parser,
                    cfg = self.cfg,
                    sampling_factor = sampling_factor,
                    cdf_threshold = 0.99,
                    use_cdf_mask = False,
                    keep_sh0 = True,
                    keep_feats = True,
                    batch_size=cfg.batch_size,
                    sparse_grad=cfg.sparse_grad,
                    visible_adam=cfg.visible_adam,
                    world_size=world_size,
                    init_scale=cfg.init_scale,
                    scene_scale=self.scene_scale,
                    abs_ratio = cfg.use_abs_ratio,
                )

                self.target_num_gaussians = target_n_gaussians

                print("Model initialized. Number of GS:", len(self.splats["means"]))

                # Densification Strategy
                self.cfg.strategy.check_sanity(self.splats, self.optimizers)

                if isinstance(self.cfg.strategy, DefaultStrategy) or isinstance(self.cfg.strategy, Distill2DStrategy):
                    self.strategy_state = self.cfg.strategy.initialize_state(
                        scene_scale=self.scene_scale,  target_num_gaussians=self.target_num_gaussians
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.strategy_state = self.cfg.strategy.initialize_state()
                elif isinstance(self.cfg.strategy, DistillationStrategy):   
                    self.strategy_state = self.cfg.strategy.initialize_state()
                    self.cfg.strategy.teacher_sampling_ratio = self.cfg.teacher_sampling_ratio
                else:
                    assert_never(self.cfg.strategy)


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step, include_ids=True)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
        "distillation": (
            "Gaussian splatting training with teacher-student distillation.",
            Config(
                strategy=DistillationStrategy(verbose=True),
            ),
        ),
        "distill2d": (
            "Gaussian splatting training with 2D teacher-student distillation.",
            Config(
                strategy=Distill2DStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)









    '''
    testing training views
    '''

    # import open3d as o3d
    # def create_camera_visualization(*args, **kwargs) -> o3d.geometry.LineSet:
    #     """
    #     Overloaded factory function to create a camera visualization LineSet from intrinsic and extrinsic camera matrices.

    #     Overload 1:
    #     create_camera_visualization(view_width_px: int, view_height_px: int, intrinsic: np.ndarray, extrinsic: np.ndarray, scale: float = 1.0)
        
    #     Overload 2:
    #     create_camera_visualization(intrinsic: o3d.camera.PinholeCameraIntrinsic, extrinsic: np.ndarray, scale: float = 1.0)
    #     """
    #     # Overload 1: arguments: int, int, np.ndarray, np.ndarray, (optional scale)
    #     if len(args) >= 4 and isinstance(args[0], int):
    #         view_width_px, view_height_px, intrinsic, extrinsic = args[:4]
    #         scale = kwargs.get("scale", 1.0)
    #         # Convert the provided intrinsic matrix to an Open3D pinhole intrinsic.
    #         fx = intrinsic[0, 0]
    #         fy = intrinsic[1, 1]
    #         cx = intrinsic[0, 2]
    #         cy = intrinsic[1, 2]
    #         pinhole = o3d.camera.PinholeCameraIntrinsic(view_width_px, view_height_px, fx, fy, cx, cy)
    #         # Call the second overload.
    #         return create_camera_visualization(pinhole, extrinsic, scale=scale)
        
    #     # Overload 2: arguments: o3d.camera.PinholeCameraIntrinsic, np.ndarray, (optional scale)
    #     elif len(args) >= 2 and isinstance(args[0], o3d.camera.PinholeCameraIntrinsic):
    #         intrinsic_obj = args[0]
    #         extrinsic = args[1]
    #         scale = kwargs.get("scale", 1.0)
    #         # Define a canonical camera frustum in camera space.
    #         # Here we use five points: the camera center and four corners at a given distance.
    #         s = scale
    #         points = np.array([
    #             [0, 0, 0],             # camera center
    #             [-s, -s, 1.5 * s],      # bottom-left
    #             [ s, -s, 1.5 * s],      # bottom-right
    #             [ s,  s, 1.5 * s],      # top-right
    #             [-s,  s, 1.5 * s]       # top-left
    #         ])
    #         lines = [
    #             [0, 1], [0, 2], [0, 3], [0, 4],
    #             [1, 2], [2, 3], [3, 4], [4, 1]
    #         ]
    #         # Transform the canonical frustum from camera to world coordinates.
    #         num_points = points.shape[0]
    #         homogeneous_points = np.hstack([points, np.ones((num_points, 1))])
    #         transformed_points = (extrinsic @ homogeneous_points.T).T[:, :3]
    #         ls = o3d.geometry.LineSet()
    #         ls.points = o3d.utility.Vector3dVector(transformed_points)
    #         ls.lines = o3d.utility.Vector2iVector(lines)
    #         return ls
    #     else:
    #         raise ValueError("Invalid arguments provided to create_camera_visualization")


    # # ------------------------------------------------------------------------------
    # # Assume that 'self.trainset' is already created and initialized as in your code.
    # # ------------------------------------------------------------------------------
    # # 1. Visualize all training views as green LineSets.
    # train_linesets = []
    # for idx in self.trainset.indices:
    #     extrinsic = self.trainset.parser.camtoworlds[idx]   # 4x4 camera-to-world matrix.
    #     camera_id = self.trainset.parser.camera_ids[idx]
    #     intrinsic_np = self.trainset.parser.Ks_dict[camera_id]  # numpy intrinsic matrix.
    #     view_width, view_height = self.trainset.parser.imsize_dict[camera_id]
    #     # Create a camera frustum using the first overload.
    #     ls = create_camera_visualization(view_width, view_height, intrinsic_np, extrinsic, scale=0.1)
    #     # Color all lines green.
    #     num_lines = np.asarray(ls.lines).shape[0]
    #     ls.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (num_lines, 1)))
    #     train_linesets.append(ls)

    # # Merge all training view LineSets into a single LineSet.
    # combined_train_points = []
    # combined_train_lines = []
    # offset = 0
    # for ls in train_linesets:
    #     pts = np.asarray(ls.points)
    #     lines = np.asarray(ls.lines)
    #     combined_train_points.append(pts)
    #     combined_train_lines.append(lines + offset)
    #     offset += pts.shape[0]
    # combined_train_points = np.vstack(combined_train_points)
    # combined_train_lines = np.vstack(combined_train_lines)
    # train_lineset = o3d.geometry.LineSet()
    # train_lineset.points = o3d.utility.Vector3dVector(combined_train_points)
    # train_lineset.lines = o3d.utility.Vector2iVector(combined_train_lines)
    # num_lines = combined_train_lines.shape[0]
    # train_lineset.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (num_lines, 1)))

    # o3d.io.write_line_set("train_view_visualization.ply", train_lineset)

    # # 2. Sample 100 novel views and visualize them as red LineSets.
    # novel_views = self.trainset.sample_novel_views(batch_size=100)
    # novel_poses = novel_views["camtoworld"].numpy()  # shape: (100, 4, 4)
    # # For simplicity, we use the intrinsic of the first training camera for all novel views.
    # first_camera_id = self.trainset.parser.camera_ids[self.trainset.indices[0]]
    # intrinsic_np = self.trainset.parser.Ks_dict[first_camera_id]
    # view_width, view_height = self.trainset.parser.imsize_dict[first_camera_id]
    # novel_linesets = []
    # for extrinsic in novel_poses:
    #     ls = create_camera_visualization(view_width, view_height, intrinsic_np, extrinsic, scale=0.1)
    #     num_lines = np.asarray(ls.lines).shape[0]
    #     ls.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (num_lines, 1)))
    #     novel_linesets.append(ls)

    # # Merge all novel view LineSets.
    # combined_novel_points = []
    # combined_novel_lines = []
    # offset = 0
    # for ls in novel_linesets:
    #     pts = np.asarray(ls.points)
    #     lines = np.asarray(ls.lines)
    #     combined_novel_points.append(pts)
    #     combined_novel_lines.append(lines + offset)
    #     offset += pts.shape[0]
    # combined_novel_points = np.vstack(combined_novel_points)
    # combined_novel_lines = np.vstack(combined_novel_lines)
    # novel_lineset = o3d.geometry.LineSet()
    # novel_lineset.points = o3d.utility.Vector3dVector(combined_novel_points)
    # novel_lineset.lines = o3d.utility.Vector2iVector(combined_novel_lines)
    # num_lines = combined_novel_lines.shape[0]
    # novel_lineset.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (num_lines, 1)))

    # o3d.io.write_line_set("novel_view_visualization.ply", novel_lineset)

    # # 3. Combine training and novel view LineSets.
    # train_pts = np.asarray(train_lineset.points)
    # novel_pts = np.asarray(novel_lineset.points)
    # combined_points = np.vstack([train_pts, novel_pts])
    # train_lines = np.asarray(train_lineset.lines)
    # novel_lines = np.asarray(novel_lineset.lines) + train_pts.shape[0]  # offset for novel indices.
    # combined_lines = np.vstack([train_lines, novel_lines])
    # combined_colors = np.vstack([np.asarray(train_lineset.colors), np.asarray(novel_lineset.colors)])

    # combined_lineset = o3d.geometry.LineSet()
    # combined_lineset.points = o3d.utility.Vector3dVector(combined_points)
    # combined_lineset.lines = o3d.utility.Vector2iVector(combined_lines)
    # combined_lineset.colors = o3d.utility.Vector3dVector(combined_colors)

    # # 4. Save the combined visualization as an OBJ file.
    # # Note: OBJ format may have limited support for lines. If you run into issues, consider saving as a PLY.
    # # o3d.io.write_line_set("novel_view_visualization.ply", combined_lineset)



    # exit()