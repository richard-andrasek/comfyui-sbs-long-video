import torch
from depth_runner import refine_depth_edges


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1 frame, RGB, 128x128
    rgb = torch.zeros((1, 3, 128, 128), device=device)

    # Left half = black, right half = white.
    # This creates an extremely obvious RGB edge.
    rgb[:, :, :, 64:] = 1.0

    # Depth starts at 0.2 on the left and 0.8 on the right,
    # but deliberately introduce a "halo" around the boundary.
    depth = torch.full(
        (1, 1, 128, 128),
        0.2,
        device=device,
    )
    depth[:, :, :, 64:] = 0.8

    # Add a gradual transition around the RGB edge.
    depth[:, :, :, 62:64] = 0.35
    depth[:, :, :, 64:66] = 0.65

    print(f"device: {device}")
    print(f"RGB:   {rgb.shape} {rgb.dtype} {rgb.device}")
    print(f"Depth: {depth.shape} {depth.dtype} {depth.device}")

    refined = refine_depth_edges(
        frames_bchw=rgb,
        depth_b1hw=depth,
        radius=2,
        strength=8.0,
    )

    print(f"Output: {refined.shape} {refined.dtype} {refined.device}")

    # Basic sanity checks.
    assert refined.shape == depth.shape
    assert refined.device == depth.device
    assert refined.dtype == depth.dtype
    assert torch.isfinite(refined).all()

    print("PASS: refinement completed successfully.")


if __name__ == "__main__":
    main()
