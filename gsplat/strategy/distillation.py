from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split, rescale
from typing_extensions import Literal
from datasets.colmap import Dataset, Parser
import tqdm
import math

@dataclass
class DistillationStrategy(Strategy):
    """
    A teacher-guided densification strategy.
    
    At every densification interval, we render both the teacher model and the student model from
    all training views. For every pixel in every view, we collect the pair:
      (student Gaussian index, teacher SH coefficients [concatenated from sh0 and shN]).
    
    For each student Gaussian, we then compute a difference measure:
    
        diff_i = sum_{pixels with student index i} || teacher_sh - mean(teacher_sh) ||₁
        
    The student Gaussians are then sorted by this difference and the top ones
    (as determined by teacher_sampling_ratio) are densified.
    
    Args:
        teacher_model: An object representing the pretrained teacher model.
                       It must implement a method `render_all_views(training_views)`
                       that returns, for each view, a dict with keys "sh0" and "shN".
        training_views: A list of training view information. Each element should at least
                        contain the view height, width, and device info.
        teacher_sampling_ratio: Fraction of student Gaussians to densify at each interval.
        grow_scale3d: Threshold for the 3D scale to decide between duplication and splitting.
        revised_opacity: Whether to use the revised opacity heuristic for splitting.
        verbose: If True, prints debugging information.
    """
    teacher_sampling_ratio: float = 0.1
    grow_scale3d: float = 0.01
    revised_opacity: bool = False
    verbose: bool = False

    initial_pruning: float = 0.5
    final_pruning: float = 0.01

    initial_scaling: float = 2.0

    disable_pruning: bool = False
    prune_opa: float = 0.005
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 10_000
    reset_every: int = 3000
    refine_every: int = 500
    pause_refine_after_reset: int = 0
    revised_opacity: bool = False
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state


    @torch.no_grad()
    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        # We do not need to accumulate gradients here.
        pass

    @torch.no_grad()
    def _grow_gs(
            self, 
            student_splats: Any,
            runner: Any,
            cfg: Any,
            step: int,
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            teacher_splats: Any,
            trainset: Any,
    ):
        

        if not self.disable_pruning:
            print('Pruning')
            # 4.1. before splitting students, we prune some of them.
            # Calculate the pruning ratio (linearly decreased from initial_pruning to final_pruning)
            pruning_ratio = self.initial_pruning - (step - self.refine_start_iter) / (self.refine_stop_iter - self.refine_start_iter) * (self.initial_pruning - self.final_pruning)
            num_prune = int(pruning_ratio * student_splats['means'].shape[0])
            
            # Sample uniformly without weighting
            total_students = student_splats['means'].shape[0]
            indices = torch.randperm(total_students, device=student_splats['means'].device)[:num_prune]
            pruning_mask = torch.zeros(total_students, dtype=torch.bool, device=student_splats['means'].device)
            pruning_mask[indices] = True

            n_prune = pruning_mask.sum().item()
            n_before_prune = total_students
            if n_prune > 0:
                remove(params=student_splats, optimizers=optimizers, state=state, mask=pruning_mask)

            print('Pruned from ', n_before_prune, ' to ', student_splats["means"].shape[0], ' students,', ' pruned ', n_prune, ' students')

            # For remaining students, scale them based on the pruning ratio: scaling varies from initial_scaling to 1.
            new_scaling = self.initial_scaling - (step - self.refine_start_iter) / (self.refine_stop_iter - self.refine_start_iter) * (self.initial_scaling - 1)
            new_scaling = math.log(new_scaling)
            sel = torch.ones(student_splats['means'].shape[0], dtype=torch.bool, device=student_splats['means'].device)
            rescale(student_splats, optimizers, state, sel, new_scaling)
                

        
        # 1. render teacher from all views, collect all pixels
            
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )
        # trainloader_iter = iter(trainloader)

        all_teacher_gaussian_ids = []
        all_student_gaussian_ids = []

        device = torch.device("cuda")

        # # Students M, # Teachers N

        splats_teacher_sh0 = teacher_splats["sh0"] # [N, 1, 3]
        splats_teacher_shN = teacher_splats["shN"] # [N, k, 3]

        # print('Rendering training views')

        for data in trainloader:
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]

            image_ids = data["image_id"].to(device)

            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
                
            _, _, info = runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+IW",
                masks=masks,
                splats = teacher_splats
            )
            # Precompute batch size (b) from the rendered info.
            b = info["max_ids"].shape[0]  # [b, H, W, 1]

            # Extract required tensors
            gaussian_ids = info["max_ids"]         # [b, H, W, 1]

            # Store the computed inverse-projected values.
            all_teacher_gaussian_ids.append(gaussian_ids.reshape(b, -1, 1))  # [b, H*W, 1]

            _, _, info = runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+IW",
                masks=masks,
                splats = student_splats
            )

            # Extract required tensors
            gaussian_ids = info["max_ids"]         # [b, H, W, 1]

            # Store the computed inverse-projected values.
            all_student_gaussian_ids.append(gaussian_ids.reshape(b, -1, 1))  # [b, H*W, 1]

        del info

        # concatenate all the results
        all_teacher_gaussian_ids = torch.cat(all_teacher_gaussian_ids, dim=0).reshape(-1).long() # [B*H*W]
        all_student_gaussian_ids = torch.cat(all_student_gaussian_ids, dim=0).reshape(-1).long() # [B*H*W]
        
        # if student < 0, remove item for both teacher and student
        mask = all_student_gaussian_ids >= 0
        all_teacher_gaussian_ids = all_teacher_gaussian_ids[mask]
        all_student_gaussian_ids = all_student_gaussian_ids[mask]

        mask = all_teacher_gaussian_ids >= 0
        all_teacher_gaussian_ids = all_teacher_gaussian_ids[mask]
        all_student_gaussian_ids = all_student_gaussian_ids[mask]

        # # 2. collect SH coefficients for all teacher Gaussians
        # teacher_sh0 = teacher_sh0[all_teacher_gaussian_ids] # [B*H*W, 1, 3]
        # teacher_shN = teacher_shN[all_teacher_gaussian_ids] # [B*H*W, 1, 3]

        # 3. collect paired student-teacher SH coefficients
        num_accumulated = torch.zeros_like(student_splats["sh0"][:, 0, 0], dtype=torch.float32) # [M]
        accumulated_sh0 = torch.zeros_like(student_splats["sh0"]) # [M, 1, 3]
        accumulated_shN = torch.zeros_like(student_splats["shN"]) # [M, k, 3]

        num_teacher = torch.ones_like(all_student_gaussian_ids, dtype=torch.float32) # [B*H*W]

        # chunk all_student_gaussian_ids into B chunks
        chunk_size = 5000000

        # print('Accumulating')
        # for i in tqdm.tqdm(range(0, len(all_student_gaussian_ids), chunk_size)):
        for i in range(0, len(all_student_gaussian_ids), chunk_size):
            chunk = all_student_gaussian_ids[i:i+chunk_size]
            
            teacher_sh0_chunk = splats_teacher_sh0[all_teacher_gaussian_ids[i:i+chunk_size]]
            teacher_shN_chunk = splats_teacher_shN[all_teacher_gaussian_ids[i:i+chunk_size]]
            num_accumulated = num_accumulated.scatter_add(0, chunk, num_teacher)
            for d in range(teacher_sh0_chunk.shape[2]):
                # Extract the slice (shape [M])
                acc_slice = accumulated_sh0[:, 0, d]
                # Perform scatter_add on the slice using 'chunk' as index (shape [chunk_size])
                acc_slice = acc_slice.scatter_add(0, chunk, teacher_sh0_chunk[:, 0, d])
                # Write the result back to accumulated_sh0
                accumulated_sh0[:, 0, d] = acc_slice
                for k in range(teacher_shN_chunk.shape[1]):
                    acc_slice_n = accumulated_shN[:, k, d]
                    acc_slice_n = acc_slice_n.scatter_add(0, chunk, teacher_shN_chunk[:, k, d])
                    accumulated_shN[:, k, d] = acc_slice_n

        mean_sh0 = accumulated_sh0.sum(dim=0, keepdim=True) / (num_accumulated.clamp_min(1)[:, None, None]) # [M, 1, 3]
        mean_shN = accumulated_shN.sum(dim=0, keepdim=True) / (num_accumulated.clamp_min(1)[:, None, None]) # [M, k, 3]

        # print('Accumulating differences')
        student_differences = torch.zeros_like(num_accumulated, dtype=torch.float32) # [M]
        # for i in tqdm.tqdm(range(0, len(all_student_gaussian_ids), chunk_size)):
        for i in range(0, len(all_student_gaussian_ids), chunk_size):
            chunk = all_student_gaussian_ids[i:i+chunk_size]
            teacher_sh0_chunk = splats_teacher_sh0[all_teacher_gaussian_ids[i:i+chunk_size]] # [chunk_size, 1, 3]
            teacher_shN_chunk = splats_teacher_shN[all_teacher_gaussian_ids[i:i+chunk_size]] # [chunk_size, k, 3]
            student_differences = student_differences.scatter_add(0, chunk, 
                                                                  torch.pow(teacher_sh0_chunk - mean_sh0[chunk], 2).sum(1).sum(1) +\
                                                                torch.pow(teacher_shN_chunk - mean_shN[chunk], 2).sum(1).sum(1))

        student_differences = (student_differences / num_accumulated.clamp_min(1)) * (num_accumulated > 0).float()
        del all_teacher_gaussian_ids, all_student_gaussian_ids, num_accumulated, accumulated_sh0, accumulated_shN, num_teacher, mean_sh0, mean_shN

            
        # 4. sort student Gaussians by difference
        target_num_students = int(teacher_splats["means"].shape[0] * self.teacher_sampling_ratio)


        sorted_indices = torch.argsort(student_differences, descending=True)


        # from the sorted indices, we sample the indices to prune - with probability pruning_
        num_students = student_splats["means"].shape[0]
        num_to_densify = max(target_num_students - num_students, 0)
        
        # gradually densify the top student Gaussians
        if self.disable_pruning:
            num_to_densify = int(min(num_students * 0.2, num_to_densify * 0.2))

        # print('current number of students: ', num_students, ' target number of students: ', target_num_students, ' densify ', num_to_densify, ' students')

        is_split = torch.zeros(num_students, dtype=torch.bool, device=device)
        is_split[sorted_indices[:num_to_densify]] = True

        # 5. densify top student Gaussians
        if num_to_densify > 0:

            split(
                params=student_splats,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )

        return num_to_densify

    @torch.no_grad()
    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        runner: Any,
        cfg: Any,
        packed: bool = False,
        trainset: Any = None,
        teacher_splats: Any = None,
    ):
        
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        # --- Refine GSs ---
        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):

            # We reorder to prune and grow, because we fit the number of GSs to the target

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # grow GSs
            n_split = self._grow_gs(params, runner, cfg, step, optimizers, state, teacher_splats, trainset)
            if self.verbose:
                print(
                    f"Step {step}: {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            torch.cuda.empty_cache()

        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )



    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        # if step > self.reset_every:
            # is_too_big = (
            #     torch.exp(params["scales"]).max(dim=-1).values
            #     > self.prune_scale3d * state["scene_scale"]
            # )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            # if step < self.refine_scale2d_stop_iter:
                # is_too_big |= state["radii"] > self.prune_scale2d

            # is_prune = is_prune
            # is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
