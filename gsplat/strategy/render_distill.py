from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split
from typing_extensions import Literal


@dataclass
class Distill2DStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.student_trainer.py
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 5_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False

    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs", "quat", "depths", "rendered_sh_coeffs","depths_and_sh"] = "means2d"

    sh_coeffs_mult: float = 50
    depths_mult: float = 10

    use_blur_split: bool = False
    blur_threshold: float = 4e-4

    def initialize_state(self, scene_scale: float = 1.0, target_num_gaussians = 1000000) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale, "blur_mask": None}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        self.target_num_gaussians = target_num_gaussians
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        if self.key_for_gradient == 'sh_coeffs':
            pass 
        elif self.key_for_gradient != 'depths_and_sh':
            assert (
                self.key_for_gradient in info
            ), "The 2D means of the Gaussians is required but missing."
            info[self.key_for_gradient].retain_grad()
            info['means2d'].retain_grad()
        else:
            assert (
                'rendered_sh_coeffs' in info and 'depths' in info
            ), "The SH coefficients of the Gaussians is required but missing."
            info['rendered_sh_coeffs'].retain_grad()
            info['depths'].retain_grad()
            info['means2d'].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_dupli, n_split, n_blur = self._grow_gs(params, optimizers, state, step)
            
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split ({n_blur} GSs dup/split for blur) "
                    f"Now having {len(params['means'])} GSs, target: {self.target_num_gaussians}. key: {self.key_for_gradient}"
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs. key: {self.key_for_gradient}"
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.use_blur_split:
                state["blur_mask"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        # if step % self.reset_every == 0:
        #     reset_opa(
        #         params=params,
        #         optimizers=optimizers,
        #         state=state,
        #         value=self.prune_opa * 2.0,
        #     )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        if self.key_for_gradient == 'sh_coeffs':
            for key in [
                "width",
                "height",
                "n_cameras",
                "radii",
                "gaussian_ids",
            ]:
                assert key in info, f"{key} is required but missing."

        elif self.key_for_gradient != 'depths_and_sh':
            for key in [
                "width",
                "height",
                "n_cameras",
                "radii",
                "gaussian_ids",
                self.key_for_gradient,
            ]:
                assert key in info, f"{key} is required but missing."
        else:
            for key in [
                "width",
                "height",
                "n_cameras",
                "radii",
                "rendered_sh_coeffs",
                "depths",
            ]:
                assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen spac
        if self.key_for_gradient == "rendered_sh_coeffs":
            grads = info[self.key_for_gradient].grad.clone() * self.sh_coeffs_mult
            grads = grads.norm(dim=-1)
        elif self.key_for_gradient == "depths":
            grads = info[self.key_for_gradient].grad.clone()[..., None] * self.depths_mult
            grads = grads.norm(dim=-1)
        elif self.key_for_gradient == "depths_and_sh":
            sh_grads = info["rendered_sh_coeffs"].grad.clone() * self.sh_coeffs_mult
            sh_grads_norm = sh_grads.norm(dim=-1)
            depths_grads = info["depths"].grad.clone()[..., None] * self.depths_mult
            depths_grads_norm = depths_grads.norm(dim=-1)
            
            grads = sh_grads_norm + depths_grads_norm

            # grads = torch.cat([sh_grads, depths_grads[..., None]], dim=-1)
        else:
            if self.absgrad:
                grads = info[self.key_for_gradient].absgrad.clone()
            else:
                grads = info[self.key_for_gradient].grad.clone()
            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
            grads = grads.norm(dim=-1)

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])


        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)
        
        if self.use_blur_split:
            if state["blur_mask"] is None:
                state["blur_mask"] = torch.zeros(n_gaussian, device=grads.device,)
            for c in range(info["max_ids"].shape[0]):
                num_pixels = torch.zeros(n_gaussian, device=grads.device)  # shape of N
                max_ids = info["max_ids"][c]  # shape of C, H, W

                valid_mask = max_ids >= 0

                max_ids = max_ids % n_gaussian
            
                num_pixels.index_add_(
                    0, 
                    max_ids[valid_mask].flatten(), 
                    torch.ones_like(max_ids[valid_mask].flatten(), dtype=torch.float32)
                )

                state["blur_mask"] = torch.logical_or(
                    state["blur_mask"], 
                    num_pixels > (info["width"] * info["height"] * self.blur_threshold)
                )

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads)


        if self.key_for_gradient != "means2d":
            means2d_grad = info["means2d"].grad.clone()
            means2d_grad[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            means2d_grad[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
            means2d_grad = means2d_grad 

            if not packed:
                means2d_grad = means2d_grad[sel]
            state["grad2d"].index_add_(0, gs_ids, means2d_grad.norm(dim=-1))


        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )


        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )


    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        grads[count == 0] = 0.0
        device = grads.device

        # 2. Determine the current number of Gaussians
        n_gaussians = len(params["means"])

        # 3. Compute the remaining number of grow_gs calls.
        # Since grow_gs is called every `self.refine_every` steps, we compute:
        calls_left = max(1, (self.refine_stop_iter - step) // self.refine_every)
        
        # 4. Calculate how many Gaussians are still needed and determine the ideal increasement per call.
        gaussians_needed = self.target_num_gaussians - n_gaussians
        ideal_increase_count = max(1, int(round(gaussians_needed / calls_left)))
        
        # 5. Sort gradients in descending order and select the top indices.
        sorted_indices = torch.argsort(grads, descending=True)
        top_indices = sorted_indices[:ideal_increase_count]
        
        # Create a boolean mask for the selected indices.
        highest_grad = torch.zeros_like(grads, dtype=torch.bool)
        highest_grad[top_indices] = True

        is_grad_high = grads > self.grow_grad2d

        is_grad_high = torch.logical_and(is_grad_high, highest_grad)

        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large

        if self.use_blur_split:
            is_split = torch.logical_or(is_split, state["blur_mask"])

        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()


        # first duplicate
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )


        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        n_blur = state["blur_mask"].sum().item() if self.use_blur_split else 0
        return n_dupli, n_split, n_blur

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
