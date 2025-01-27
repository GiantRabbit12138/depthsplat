from ..types import BatchedExample, BatchedViews


def apply_patch_shim_to_views(views: BatchedViews, patch_size: int) -> BatchedViews:
    """
    该函数的主要作用是对批量图像（views["image"]）进行中心裁剪
    使图像的尺寸对齐到给定的 patch_size 的倍数，同时根据裁剪调整相机内参矩阵（views["intrinsics"]）
    """
    _, _, _, h, w = views["image"].shape
    
    # debug
    # print("[apply_patch_shim_to_views] views[\"image\"].shape: {}".format(views["image"].shape))

    # Image size must be even so that naive center-cropping does not cause misalignment.
    assert h % 2 == 0 and w % 2 == 0

    h_new = (h // patch_size) * patch_size
    row = (h - h_new) // 2
    w_new = (w // patch_size) * patch_size
    col = (w - w_new) // 2
    
    # debug
    # print("[apply_patch_shim_to_views] patch_size: {}".format(patch_size))
    # print("[apply_patch_shim_to_views] h_new: {}, row: {}, w_new: {}, col: {}".format(h_new, row, w_new, col))

    # Center-crop the image.
    image = views["image"][:, :, :, row : row + h_new, col : col + w_new]
    depth_image = views["depth_image"][:, :, :, row : row + h_new, col : col + w_new]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = views["intrinsics"].clone()
    intrinsics[:, :, 0, 0] *= w / w_new  # fx
    intrinsics[:, :, 1, 1] *= h / h_new  # fy

    return {
        **views,
        "image": image,
        "depth_image": depth_image,
        "intrinsics": intrinsics,
    }


def apply_patch_shim(batch: BatchedExample, patch_size: int) -> BatchedExample:
    """Crop images in the batch so that their dimensions are cleanly divisible by the
    specified patch size.
    """
    return {
        **batch,
        "context": apply_patch_shim_to_views(batch["context"], patch_size),
        "target": apply_patch_shim_to_views(batch["target"], patch_size),
    }
