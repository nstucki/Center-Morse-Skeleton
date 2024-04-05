from __future__ import annotations
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
import typing
import torch
import enum
from contextlib import contextmanager
from timeit import default_timer
if typing.TYPE_CHECKING:
    from typing import Tuple, List, Dict, Union, Literal, Callable, Generator
    from numpy.typing import NDArray
    LossOutputName = str
    from jaxtyping import Float
    from torch import Tensor

# See https://stackoverflow.com/a/30024601
@contextmanager
def elapsed_timer() -> Generator[Callable[[],float], None, None]:
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

class ClDiceMetric(torch.nn.modules.loss._Loss):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def cl_score(v, s):
        """[this function computes the skeleton volume overlap]
        Args:
            v ([bool]): [image]
            s ([bool]): [skeleton]
        Returns:
            [float]: [computed skeleton volume intersection]
        """
        return np.sum(v*s)/np.sum(s)

    @classmethod
    def cl_dice(cls, v_p, v_l):
        """[this function computes the cldice metric]
        Args:
            v_p ([bool]): [predicted image]
            v_l ([bool]): [ground truth image]
        Returns:
            [float]: [cldice metric]
        """
        if len(v_p.shape) == 2:
            tprec = cls.cl_score(v_p, skeletonize(v_l))
            tsens = cls.cl_score(v_l, skeletonize(v_p))
        elif len(v_p.shape) == 3:
            tprec = cls.cl_score(v_p, skeletonize_3d(v_l))
            tsens = cls.cl_score(v_l, skeletonize_3d(v_p))
        else:
            raise RuntimeError("Inputs to cl_dice have wrong shape. Shapes 2 and 3 are supported.")
        return 2 * tprec * tsens / (tprec + tsens)

    def forward(self,
                input: Float[Tensor, "batch channels *spatial_dimensions"],
                target: Float[Tensor, "batch channels *spatial_dimensions"]
                ) -> Tuple[float, Dict[LossOutputName, float | Float[Tensor, ""]]]:           

        with elapsed_timer() as elapsed_cl_dice:
            cl_dice_metrics = [
                self.cl_dice(input_batch_element.squeeze(0), target_batch_element.squeeze(0))
                for input_batch_element, target_batch_element in zip(input.detach().cpu().numpy(), target.detach().cpu().numpy())
            ]

        batch_size = input.shape[0]
        dic: Dict[LossOutputName, float | torch.Tensor] = {**{
            'ClDice Metric': float(np.mean(cl_dice_metrics)),
            'ClDice Metric computation time (s)': elapsed_cl_dice() / batch_size
        }}
        metric: float = dic['ClDice Metric'] # type: ignore
        return metric, dic