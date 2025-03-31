#include "bindings.h"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_fwd_approx_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]

    const S *__restrict__ gt_image, // [C, image_height, image_width, 3]

    S *__restrict__ render_colors, // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    int32_t *__restrict__ max_ids, // [C, image_height, image_width]
    S *__restrict__ accumulated_weights_value, // [C, N]
    int32_t *__restrict__ accumulated_count, // [C, N]
    S *__restrict__ max_weight_depths, // [C, image_height, image_width, 1]
    S *__restrict__ accumulated_cur_colors, // [C, N, 3] - cur_color RGB 누적
    S *__restrict__ accumulated_final_colors, // [C, N, 3] - final rendered color RGB 누적
    S *__restrict__ accumulated_one_minus_alphas, // [C, N] - (1-alpha) 누적
    S *__restrict__ accumulated_colors, // [C, N, 3] - 누적된 색상 값
    S *__restrict__ accumulated_gt_colors // [C, N, 3] - GT 이미지 색상 누적
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * COLOR_DIM;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    max_ids += camera_id * image_height * image_width;
    max_weight_depths += camera_id * image_height * image_width;
    
    if (gt_image != nullptr) {
        gt_image += camera_id * image_height * image_width * 3;
    }
    
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    const int32_t pix_id = i * image_width + j;

    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] = backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        max_ids[pix_id] = -1;
        return;
    }

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
        ? n_isects
        : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec3<S> *xy_opacity_batch = reinterpret_cast<vec3<float> *>(&id_batch[block_size]);
    vec3<S> *conic_batch = reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]);

    S T = 1.0f;
    uint32_t cur_idx = 0;
    uint32_t tr = block.thread_rank();
    S pix_out[COLOR_DIM] = {0.f};
    S max_depth = 0.0f;
    int32_t max_id = -1;
    S max_weight = 0.0f;

    int32_t max_index = 0;

    // First pass: accumulate colors and find max depth
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
        }

        block.sync();
        uint32_t batch_size = min(block_size, range_end - batch_start);

        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
            const S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
            S alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) {
                done = true;
                max_index = batch_start + t;
                break;
            }

            int32_t g = id_batch[t];
            const S vis = alpha * T;
            const S *c_ptr = colors + g * COLOR_DIM;

            // // Store cur_color contribution (RGB only)
            // GSPLAT_PRAGMA_UNROLL
            // for (uint32_t k = 0; k < 3; ++k) {
            // }

            // Store (1-alpha)
            atomicAdd(&accumulated_one_minus_alphas[g], 1.0f - alpha);

            // accumulate colors (full dimension)
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < 3; ++k) {
                atomicAdd(&accumulated_colors[g * 3 + k], pix_out[k]);
            }

            // Accumulate colors
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                S cur_color = c_ptr[k] * vis;
                if (k < 3) {
                    atomicAdd(&accumulated_cur_colors[g * 3 + k], cur_color);
                }

                pix_out[k] += cur_color;
            }

            // Update max depth and weight
            if (vis > max_weight) {
                max_id = g;
                max_weight = vis;
                max_depth = c_ptr[COLOR_DIM - 1];
            }

            // Update accumulated weights and counts
            atomicAdd(&accumulated_weights_value[g], vis);
            atomicAdd(&accumulated_count[g], 1);

            cur_idx = batch_start + t;
            T = next_T;
        }
    }


    // second pass:
    // 1. update final colors for contributing Gaussians
    // 2. update gt colors for contributing Gaussians

    S T_ = 1.0f;  // Initialize to 1.0 like in first pass
    bool done_ = false;

    // First pass: accumulate colors and find max depth
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done_) >= block_size) {
            break;
        }

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
        }

        block.sync();
        uint32_t batch_size = min(block_size, range_end - batch_start);

        for (uint32_t t = 0; (t < batch_size) && !done_; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
            const S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
            S alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_T_ = T_ * (1.0f - alpha);
            if (next_T_ <= 1e-4) {
                done_ = true;
                break;
            }

            int32_t g = id_batch[t];
            const S vis = alpha * T_;

            cur_idx = batch_start + t;
            T_ = next_T_;

            // update final colors for contributing Gaussians
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < 3; ++k) {
                atomicAdd(&accumulated_final_colors[g * 3 + k], pix_out[k]);
                atomicAdd(&accumulated_gt_colors[g * 3 + k], gt_image[pix_id * 3 + k]);
            }
        }
    }

    if (inside) {
        render_alphas[pix_id] = 1.0f - T;
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] = backgrounds == nullptr ? pix_out[k]
                                                       : (pix_out[k] + T * backgrounds[k]);
        }

        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
        max_ids[pix_id] = max_id;
        max_weight_depths[pix_id] = max_depth;
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> call_kernel_with_dim_approx(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]

    // GT image
    const at::optional<torch::Tensor> &gt_image // [C, image_height, image_width, 3]
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor renders = torch::zeros(
        {C, image_height, image_width, channels},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor alphas = torch::zeros(
        {C, image_height, image_width, 1},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor last_ids = torch::empty(
        {C, image_height, image_width}, means2d.options().dtype(torch::kInt32)
    );

    torch::Tensor max_ids = torch::empty(
        {C, image_height, image_width, 1}, means2d.options().dtype(torch::kInt32)
    );

    torch::Tensor accumulated_weights_value = torch::zeros(
        {C, N}, means2d.options().dtype(torch::kFloat32)
    );

    torch::Tensor accumulated_weights_count = torch::zeros(
        {C, N}, means2d.options().dtype(torch::kInt32)
    );

    torch::Tensor max_weight_depths = torch::full(
        {C, image_height, image_width, 1}, -1, means2d.options().dtype(torch::kFloat32)
    );

    torch::Tensor accumulated_cur_colors = torch::zeros({C, N, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor accumulated_final_colors = torch::zeros({C, N, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor accumulated_one_minus_alphas = torch::zeros({C, N}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor accumulated_colors = torch::zeros({C, N, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor accumulated_gt_colors = torch::zeros({C, N, 3}, means2d.options().dtype(torch::kFloat32));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_fwd_approx_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }
    rasterize_to_pixels_fwd_approx_kernel<CDIM, float>
        <<<blocks, threads, shared_mem, stream>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            
            gt_image.has_value() ? gt_image.value().data_ptr<float>() : nullptr,

            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            max_ids.data_ptr<int32_t>(),
            accumulated_weights_value.data_ptr<float>(),
            accumulated_weights_count.data_ptr<int32_t>(),
            max_weight_depths.data_ptr<float>(),
            accumulated_cur_colors.data_ptr<float>(),
            accumulated_final_colors.data_ptr<float>(),
            accumulated_one_minus_alphas.data_ptr<float>(),
            accumulated_colors.data_ptr<float>(),
            accumulated_gt_colors.data_ptr<float>()
        );

    return std::make_tuple(renders, alphas, last_ids, max_ids, accumulated_weights_value, accumulated_weights_count, max_weight_depths, accumulated_cur_colors, accumulated_final_colors, accumulated_one_minus_alphas, accumulated_colors, accumulated_gt_colors);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_fwd_approx_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]

    // GT image
    const at::optional<torch::Tensor> &gt_image // [C, image_height, image_width, 3]
) {
    GSPLAT_CHECK_INPUT(colors);
    uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim_approx<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                        \
            gt_image                                                           \
        );

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    switch (channels) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(11)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(56)
        __GS__CALL_(59)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}

} // namespace gsplat