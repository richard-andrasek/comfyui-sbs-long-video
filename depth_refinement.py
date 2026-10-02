"""
Depth refinement utilities for ComfyUI LongVideo SBS.

All operations are implemented with PyTorch so that depth refinement can
remain on the same device as the DA3 depth tensor (normally CUDA).

Provided refiners:

    refine_depth_edges()
        Lightweight local edge-aware depth smoothing.

    fast_global_smoother()
        PyTorch implementation of the Fast Global Smoother (FGS)
        weighted-least-squares formulation.

Both functions expect:

    frames_bchw: [B, 3, H, W], RGB, normally in [0, 1]
    depth_b1hw:  [B, 1, H, W], depth, normally in [0, 1]
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Simple edge-aware depth refinement
# ---------------------------------------------------------------------------

def refine_depth_edges(
    frames_bchw: torch.Tensor,
    depth_b1hw: torch.Tensor,
    radius: int = 2,
    strength: float = 8.0,
) -> torch.Tensor:
    """
    Simple local edge-aware depth smoothing.

    This is the original experimental refinement. It estimates local RGB
    variance and uses that to modulate local depth averaging.

    NOTE:
        This is intentionally retained as a baseline/experimental method.
        The Fast Global Smoother below is the preferred refinement.

    Args:
        frames_bchw:
            RGB frames, shape [B, 3, H, W], values normally in [0, 1].

        depth_b1hw:
            Depth map, shape [B, 1, H, W], values normally in [0, 1].

        radius:
            Spatial radius of the local smoothing window.

        strength:
            Strength of the RGB-variance weighting.

    Returns:
        Refined depth with the same shape, dtype and device as depth_b1hw.
    """

    if radius <= 0 or strength <= 0:
        return depth_b1hw

    # Keep guide and depth on exactly the same device/dtype.
    frames_bchw = frames_bchw.to(
        device=depth_b1hw.device,
        dtype=depth_b1hw.dtype,
    )

    kernel_size = radius * 2 + 1

    # RGB -> luminance.
    gray = (
        frames_bchw[:, 0:1] * 0.299
        + frames_bchw[:, 1:2] * 0.587
        + frames_bchw[:, 2:3] * 0.114
    )

    # Local RGB variance.
    local_mean = F.avg_pool2d(
        gray,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    local_sq_mean = F.avg_pool2d(
        gray * gray,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    local_variance = (
        local_sq_mean - local_mean * local_mean
    ).clamp_min(0.0)

    # Original experimental weighting.
    edge_weight = torch.exp(-strength * local_variance)

    # Local depth average.
    depth_mean = F.avg_pool2d(
        depth_b1hw,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    refined = (
        depth_b1hw * edge_weight
        + depth_mean * (1.0 - edge_weight)
    )

    return refined.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Fast Global Smoother
# ---------------------------------------------------------------------------

def _guide_to_luminance(frames_bchw: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB guide to luminance.

    Input:
        [B, 3, H, W]

    Output:
        [B, 1, H, W]
    """

    return (
        frames_bchw[:, 0:1] * 0.299
        + frames_bchw[:, 1:2] * 0.587
        + frames_bchw[:, 2:3] * 0.114
    )


def _solve_tridiagonal(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """
    Batched Thomas-algorithm solver for tridiagonal systems.

    All tensors have the spatial dimension on the final axis.

    Shapes:
        [B, C, N]

    The solve is sequential along N, but every position is processed
    simultaneously across B/C on the GPU.

    This is the main unavoidable sequential component of the classic
    Fast Global Smoother formulation.
    """

    n = rhs.shape[-1]

    # Clone because forward elimination modifies these values.
    c_prime = torch.empty_like(upper)
    d_prime = torch.empty_like(rhs)

    # First element.
    denom = diagonal[..., 0].clamp_min(1e-6)

    c_prime[..., 0] = upper[..., 0] / denom
    d_prime[..., 0] = rhs[..., 0] / denom

    # Forward elimination.
    for i in range(1, n):
        denom = (
            diagonal[..., i]
            - lower[..., i] * c_prime[..., i - 1]
        ).clamp_min(1e-6)

        c_prime[..., i] = (
            upper[..., i] / denom
        )

        d_prime[..., i] = (
            rhs[..., i]
            - lower[..., i] * d_prime[..., i - 1]
        ) / denom

    # Back substitution.
    result = torch.empty_like(rhs)

    result[..., -1] = d_prime[..., -1]

    for i in range(n - 2, -1, -1):
        result[..., i] = (
            d_prime[..., i]
            - c_prime[..., i] * result[..., i + 1]
        )

    return result


def _fgs_horizontal(
    src: torch.Tensor,
    guide: torch.Tensor,
    lambda_value: float,
    sigma_color: float,
) -> torch.Tensor:
    """
    One horizontal FGS solve.

    src/guide:
        [B, 1, H, W]
    """

    # Difference between neighboring guide pixels.
    guide_diff = guide[..., 1:] - guide[..., :-1]

    # Edge affinity.
    #
    # Small RGB difference -> weight near 1.
    # Strong RGB edge     -> weight near 0.
    weights = torch.exp(
        -guide_diff.abs() / max(sigma_color, 1e-6)
    )

    b, c, h, w = src.shape

    # Tridiagonal coefficients.
    lower = torch.zeros_like(src)
    upper = torch.zeros_like(src)
    diagonal = torch.ones_like(src)

    # Edge weight between x-1 and x.
    left_weight = F.pad(
        weights,
        (1, 0),
        mode="constant",
        value=0.0,
    )

    # Edge weight between x and x+1.
    right_weight = F.pad(
        weights,
        (0, 1),
        mode="constant",
        value=0.0,
    )

    lower[..., 1:] = -lambda_value * left_weight[..., 1:]
    upper[..., :-1] = -lambda_value * right_weight[..., :-1]

    diagonal += lambda_value * (
        left_weight + right_weight
    )

    # Flatten rows into independent systems.
    src_rows = src.reshape(b * c * h, w)
    lower_rows = lower.reshape(b * c * h, w)
    diagonal_rows = diagonal.reshape(b * c * h, w)
    upper_rows = upper.reshape(b * c * h, w)

    solved = _solve_tridiagonal(
        lower_rows,
        diagonal_rows,
        upper_rows,
        src_rows,
    )

    return solved.reshape(b, c, h, w)


def _fgs_vertical(
    src: torch.Tensor,
    guide: torch.Tensor,
    lambda_value: float,
    sigma_color: float,
) -> torch.Tensor:
    """
    One vertical FGS solve.

    src/guide:
        [B, 1, H, W]
    """

    # Convert vertical dimension into the final dimension.
    src_t = src.transpose(-1, -2)
    guide_t = guide.transpose(-1, -2)

    result = _fgs_horizontal(
        src_t,
        guide_t,
        lambda_value,
        sigma_color,
    )

    return result.transpose(-1, -2)


def fast_global_smoother(
    frames_bchw: torch.Tensor,
    depth_b1hw: torch.Tensor,
    lambda_value: float = 64.0,
    sigma_color: float = 0.10,
    lambda_attenuation: float = 0.25,
    num_iter: int = 3,
) -> torch.Tensor:
    """
    Fast Global Smoother depth refinement.

    This is a PyTorch implementation of the weighted-least-squares
    formulation used by the Fast Global Smoother.

    The RGB image acts as the guide. Depth is encouraged to become smooth,
    but smoothing across strong RGB edges is strongly suppressed.

    Args:
        frames_bchw:
            RGB frames, [B, 3, H, W], normally [0, 1].

        depth_b1hw:
            DA3 depth, [B, 1, H, W], normally [0, 1].

        lambda_value:
            Regularization/smoothing strength.

            Larger values:
                More aggressive depth smoothing.

            Smaller values:
                Preserve more of the original DA3 depth.

        sigma_color:
            RGB edge sensitivity.

            Smaller values:
                Even relatively small RGB differences block smoothing.

            Larger values:
                Only strong RGB edges block smoothing.

        lambda_attenuation:
            Reduction applied to lambda after each iteration.

            0.25 is the standard FGS default.

        num_iter:
            Number of horizontal/vertical solve iterations.

            3 is the standard FGS default.

    Returns:
        Refined depth with the same shape, dtype and device as depth_b1hw.
    """

    if num_iter <= 0:
        return depth_b1hw

    if lambda_value <= 0:
        return depth_b1hw

    if sigma_color <= 0:
        raise ValueError("sigma_color must be > 0")

    if not (0 < lambda_attenuation <= 1):
        raise ValueError(
            "lambda_attenuation must be > 0 and <= 1"
        )

    # Ensure guide/depth are colocated.
    frames_bchw = frames_bchw.to(
        device=depth_b1hw.device,
        dtype=depth_b1hw.dtype,
    )

    # FGS uses luminance for this depth-refinement implementation.
    guide = _guide_to_luminance(frames_bchw)

    result = depth_b1hw

    current_lambda = float(lambda_value)

    for _ in range(num_iter):
        result = _fgs_horizontal(
            result,
            guide,
            current_lambda,
            sigma_color,
        )

        result = _fgs_vertical(
            result,
            guide,
            current_lambda,
            sigma_color,
        )

        current_lambda *= lambda_attenuation

    return result.clamp(0.0, 1.0)
