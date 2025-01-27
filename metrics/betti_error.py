from enum import Enum
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
from skimage.measure import euler_number, label

def betti_number(img: np.array) -> List:
    """
    calculates the Betti number B0, B1, and B2 for a 3D img
    from the Euler characteristic number

    code prototyped by
    - Martin Menten (Imperial College)
    - Suprosanna Shit (Technical University Munich)
    - Johannes C. Paetzold (Imperial College)
    """

    # make sure the image is 3D (for connectivity settings)
    assert img.ndim == 3

    # 6 or 26 neighborhoods are defined for 3D images,
    # (connectivity 1 and 3, respectively)
    # If foreground is 26-connected, then background is 6-connected, and conversely
    N6 = 1
    N26 = 3

    # important first step is to
    # pad the image with background (0) around the border!
    padded = np.pad(img, pad_width=1)

    # make sure the image is binary with
    assert set(np.unique(padded)).issubset({0, 1})

    # calculate the Betti numbers B0, B2
    # then use Euler characteristic to get B1

    # get the label connected regions for foreground
    _, b0 = label(
        padded,
        # return the number of assigned labels
        return_num=True,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    euler_char_num = euler_number(
        padded,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    # get the label connected regions for background
    _, b2 = label(
        1 - padded,
        # return the number of assigned labels
        return_num=True,
        # 6 neighborhoods for background
        connectivity=N6,
    )

    # NOTE: need to substract 1 from b2
    b2 -= 1

    b1 = b0 + b2 - euler_char_num  # Euler number = Betti:0 - Bett:1 + Betti:2

    # print(f"Betti number: b0 = {b0}, b1 = {b1}, b2 = {b2}")

    return [b0, b1, b2]


def bett_error(pred, gt):
    assert pred.shape == gt.shape
    assert pred.ndim == 3
    assert gt.ndim == 3

    # make sure the images are binary
    assert set(np.unique(pred)).issubset({0, 1})
    assert set(np.unique(gt)).issubset({0, 1})

    # calculate the Betti numbers for the prediction and ground truth
    betti_pred = betti_number(pred)
    betti_gt = betti_number(gt)

    # calculate the error for each Betti number
    betti_error = np.abs(np.array(betti_pred) - np.array(betti_gt)).sum()

    return betti_error