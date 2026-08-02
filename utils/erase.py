import numpy as np
from typing import List, Tuple, Union
import random


def random_erase_around_points(
        image: np.ndarray,
        points_fh: List[Tuple[int, int]],
        points_ps: List[Tuple[int, int]],
        p: float = 0.5,
        num_points_to_erase_fh: Union[int, Tuple[int, int]] = (10, 15),
        num_points_to_erase_ps: Union[int, Tuple[int, int]] = (5, 10),
        patch_size_fh: int = 8,
        patch_size_ps: int = 6
) -> np.ndarray:
    """
    以一定概率随机选择几个点，并将这些点周围的patch_size x patch_size区域置零

    Args:
        image: 输入图像，numpy数组，形状为(256, 256, 3)
        points: 包含二维坐标点的列表，每个点格式为(x, y)
        p: 执行数据增强的概率，范围[0, 1]，默认0.5
        fh 10-15
        ps 5-10
        num_points_to_erase: 要擦除的点的数量，可以是固定数字或范围(min, max)
        patch_size: 要擦除的区域大小（边长），默认为4
    Returns:
        增强后的图像，形状保持(256, 256, 3)
    """
    # 首先进行概率判断
    if random.random() >= p:
        return image
    
    #print("Erase Succ")
    image = image.copy()

    # 确定要擦除的点的数量

    num_points_fh = random.randint(num_points_to_erase_fh[0], num_points_to_erase_fh[1])
    num_points_ps = random.randint(num_points_to_erase_ps[0], num_points_to_erase_ps[1])


    # 随机选择点
    selected_points_fh = random.sample(points_fh, num_points_fh)
    selected_points_ps = random.sample(points_ps, num_points_ps)

    half_patch_fh = patch_size_fh // 2
    half_patch_ps = patch_size_ps // 2


    for point in selected_points_fh:
        x, y = point
        #x,y = np.random.randint(half_patch_fh, 256-half_patch_fh),np.random.randint(half_patch_fh, 256-half_patch_fh)
        if not (0 <= x < 256 and 0 <= y < 256):
            continue
        x_min = max(0, x - half_patch_fh)
        x_max = min(256, x + half_patch_fh)
        y_min = max(0, y - half_patch_fh)
        y_max = min(256, y + half_patch_fh)
        image[x_min:x_max, y_min:y_max,:] = 0


    for point in selected_points_ps:
        x, y = point
        #x,y = np.random.randint(half_patch_ps, 256-half_patch_ps),np.random.randint(half_patch_ps, 256-half_patch_ps)
        if not (0 <= x < 256 and 0 <= y < 256):
            continue
        x_min = max(0, x - half_patch_ps)
        x_max = min(256, x + half_patch_ps)
        y_min = max(0, y - half_patch_ps)
        y_max = min(256, y + half_patch_ps)
        image[x_min:x_max, y_min:y_max, :] = 0

    return image