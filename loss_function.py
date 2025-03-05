import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from previous_skeleton import MentenSkeletonize, ShitSkeletonize, VitiSkeletonize
from morse_skeleton import DMTSkeletonize
from monai.losses.dice import DiceLoss
import enum
import numpy as np
import typing
import gudhi.wasserstein
import monai
sys.path.append("./metrics/Betti-Matching-3D/build")
from contextlib import contextmanager
import betti_matching
from timeit import default_timer

from typing import Tuple, List, Dict, Union, Literal, Callable, Generator
from numpy.typing import NDArray
LossOutputName = str
from jaxtyping import Float
from torch import Tensor

class FiltrationType(enum.Enum):
    SUPERLEVEL = "superlevel"
    SUBLEVEL = "sublevel"
    BOTHLEVELS = "bothlevels"

class PushUnmatchedTo(enum.Enum):
    DIAGONAL = "diagonal"
    ONE_ZERO = "one_zero"
    DEATH_DEATH = "death_death"

class CubicalComplexConstruction(enum.Enum):
    V_CONSTRUCTION = "V" # Currently not supported
    T_CONSTRUCTION = "T"

class ComparisonOperation(enum.Enum):
    UNION = "union"
    INTERSECTION = "intersection" # Currently not supported

class soft_cldice(nn.Module):
    def __init__(self, sigmoid=True, mode='Shit', iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.sigmoid = sigmoid
        self.iter = iter_
        self.smooth = smooth
        if mode == 'DMT':
            self.skeletonization_module = DMTSkeletonize(reparameterize='fixed')
        if mode == 'Menten': 
            self.skeletonization_module = MentenSkeletonize(num_iter=10)
        if mode == 'Viti':
            self.skeletonization_module = VitiSkeletonize(num_iter=10)
        if mode == 'Shit':
            self.skeletonization_module = ShitSkeletonize(num_iter=10)

    def forward(self, y_pred, y_true):
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        skel_pred = self.skeletonization_module(y_pred)
        skel_true = self.skeletonization_module(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice, {}


class soft_dice_cldice(nn.Module):
    def __init__(self, sigmoid=True, mode='Shit', iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.dice_loss = DiceLoss(sigmoid=sigmoid)
        self.cl_dice_loss = soft_cldice(sigmoid=sigmoid, mode=mode, iter_=iter_, smooth = smooth)
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)  
        cl_dice, _ = self.cl_dice_loss(y_pred, y_true)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice, {'dice_loss':dice, 'cldice_loss': cl_dice}
    

class DiceBettiMatchingLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        alpha: float = 0.5,
        relative=False,
        sigmoid=True,
        include_unmatched_target: bool = False,
        filtration_type: FiltrationType=FiltrationType.SUPERLEVEL,
        cpu_batch_size=16,
        push_unmatched_to: PushUnmatchedTo = PushUnmatchedTo.DIAGONAL
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.relative = relative
        self.sigmoid = sigmoid
        self.filtration_type = filtration_type
        self.cpu_batch_size = cpu_batch_size
        self.push_unmatched_to = push_unmatched_to
        self.include_unmatched_target = include_unmatched_target

    def forward(self,
                input: Float[Tensor, "batch channels *spatial_dimensions"],
                target: Float[Tensor, "batch channels *spatial_dimensions"],
                betti_matching_second_factor: float=1.0 # This is an additional factor, besides alpha, to allow slowly transitioning to using Betti loss
                ) -> Tuple[Float[Tensor, ""], Dict[LossOutputName, Union[float , Float[Tensor, ""]]]]:

        with elapsed_timer() as elapsed_dice:
            dice_loss: torch.Tensor = monai.losses.dice.DiceLoss(sigmoid=self.sigmoid)(input, target)
        with elapsed_timer() as elapsed_betti:
            betti_matching_losses, additional_outputs_dict = compute_betti_matching_loss(
                input, target, cpu_batch_size=self.cpu_batch_size, include_unmatched_target=self.include_unmatched_target, sigmoid=self.sigmoid,
                filtration_type=self.filtration_type, relative=self.relative, push_unmatched_to=self.push_unmatched_to)
        
        batch_size = input.shape[0]
        dic: Dict[LossOutputName, Union[float, torch.Tensor]] = {**{
            'Dice loss': dice_loss,
            'Betti matching loss': torch.mean(torch.cat(betti_matching_losses)),
        }, **additional_outputs_dict}
        loss: torch.Tensor = dic['Dice loss'] + self.alpha*dic['Betti matching loss']*betti_matching_second_factor # type: ignore
        return loss, dic


# See https://stackoverflow.com/a/30024601
@contextmanager
def elapsed_timer() -> Generator[Callable[[],float], None, None]:
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

class DiceWassersteinLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        alpha: float = 0.5,
        relative=False,
        sigmoid=True,
        filtration_type: FiltrationType=FiltrationType.SUPERLEVEL,
        cpu_batch_size=16
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.relative = relative
        self.sigmoid = sigmoid
        self.filtration_type = filtration_type
        self.cpu_batch_size = cpu_batch_size

    def forward(self,
                input: Float[Tensor, "batch channels *spatial_dimensions"],
                target: Float[Tensor, "batch channels *spatial_dimensions"],
                wasserstein_second_factor: float=1.0 # This is an additional factor, besides alpha, to allow slowly transitioning to using Wasserstein loss
                ) -> Tuple[Float[Tensor, ""], Dict[LossOutputName, Union[float, Float[Tensor, ""]]]]:

        with elapsed_timer() as elapsed_dice:
            dice_loss: torch.Tensor = monai.losses.dice.DiceLoss(sigmoid=self.sigmoid)(input, target)
        with elapsed_timer() as elapsed_wasserstein:
            wasserstein_losses, additional_outputs_dict = compute_wasserstein_loss(input, target, cpu_batch_size=self.cpu_batch_size, sigmoid=self.sigmoid, filtration_type=self.filtration_type, relative=self.relative)
        
        batch_size = input.shape[0]
        dic: Dict[LossOutputName, Union[float, torch.Tensor]] = {**{
            'Dice loss': dice_loss,
            'Wasserstein loss': torch.mean(torch.cat(wasserstein_losses)),
        }, **additional_outputs_dict}
        loss: torch.Tensor = dic['Dice loss'] + self.alpha*dic['Wasserstein loss']*wasserstein_second_factor # type: ignore
        return loss, dic
    

def compute_betti_matching_loss(prediction: Float[Tensor, "batch channels *spatial_dimensions"],
                                target: Float[Tensor, "batch channels *spatial_dimensions"],
                                cpu_batch_size: int,
                                include_unmatched_target: bool,
                                sigmoid=False,
                                relative=False,
                                comparison_operation: ComparisonOperation=ComparisonOperation.UNION,
                                filtration_type: FiltrationType=FiltrationType.SUPERLEVEL,
                                complex_construction: CubicalComplexConstruction=CubicalComplexConstruction.T_CONSTRUCTION,
                                push_unmatched_to: PushUnmatchedTo = PushUnmatchedTo.DIAGONAL
                                ) -> Tuple[List[torch.Tensor], Dict[LossOutputName, Tensor]]:
    if complex_construction != CubicalComplexConstruction.T_CONSTRUCTION:
        raise NotImplementedError("Only T construction is currently supported")
    if comparison_operation == ComparisonOperation.INTERSECTION:
        raise NotImplementedError("The intersection comparison operation is currently not supported")

    batch_size = prediction.shape[0]

    if sigmoid:
        prediction = torch.sigmoid(prediction)

    if filtration_type == FiltrationType.SUPERLEVEL:
        # Using (1 - ...) to allow binary sorting optimization on the label, which expects values [0, 1]
        prediction = 1 - prediction
        target = 1 - target
    if filtration_type == FiltrationType.BOTHLEVELS:
        # Just duplicate the number of elements in the batch, once with sublevel, once with superlevel
        prediction = torch.concat([prediction, 1 - prediction])
        target = torch.concat([target, 1 - target])

    if relative:
        pad_value_prediction = prediction.min().item() # make sure to not propagate gradients here!
        pad_value_target = target.min().item()

        prediction = torch.nn.functional.pad(prediction, pad=[1 for _ in range(2 * (len(prediction.shape) - 2))], value=pad_value_prediction)
        target = torch.nn.functional.pad(target, pad=[1 for _ in range(2 * (len(target.shape) - 2))], value=pad_value_target)

    split_indices = np.arange(cpu_batch_size, prediction.shape[0], cpu_batch_size)
    predictions_list_numpy = np.split(prediction.detach().cpu().numpy().astype(np.float64), split_indices)
    targets_list_numpy = np.split(target.detach().cpu().numpy().astype(np.float64), split_indices)

    losses = []
    losses_dicts = []

    num_dimensions = prediction.ndim - 2
    num_matched_by_dim = torch.zeros((num_dimensions,), device=prediction.device)
    num_unmatched_prediction_by_dim = torch.zeros((num_dimensions,), device=prediction.device)
    if include_unmatched_target:
        num_unmatched_target_by_dim = torch.zeros((num_dimensions,), device=prediction.device)

    current_instance_index = 0
    for predictions_cpu_batch, targets_cpu_batch in zip(predictions_list_numpy, targets_list_numpy):
        predictions_cpu_batch, targets_cpu_batch = list(predictions_cpu_batch.squeeze(1)), list(targets_cpu_batch.squeeze(1))
        if not (all(a.data.contiguous for a in predictions_cpu_batch) and all(a.data.contiguous for a in targets_cpu_batch)):
            print("WARNING! Non-contiguous arrays encountered. Shape:", predictions_cpu_batch[0].shape)
            global ENCOUNTERED_NONCONTIGUOUS
            ENCOUNTERED_NONCONTIGUOUS=True
        predictions_cpu_batch = [np.ascontiguousarray(a) for a in predictions_cpu_batch]
        targets_cpu_batch = [np.ascontiguousarray(a) for a in targets_cpu_batch]

        results = betti_matching.compute_matching(
            predictions_cpu_batch, targets_cpu_batch, include_unmatched_target)
        for result_arrays in results:
            loss, losses_dict = _betti_matching_loss(
                prediction[current_instance_index].squeeze(0),
                target[current_instance_index].squeeze(0), result_arrays,
                push_unmatched_to,
                include_unmatched_target=include_unmatched_target)
            losses.append(loss)
            losses_dicts.append(losses_dict)

            num_matched_by_dim += torch.tensor(result_arrays.num_matches_by_dim, device=prediction.device, dtype=torch.long)
            num_unmatched_prediction_by_dim += torch.tensor(result_arrays.num_unmatched_prediction_by_dim, device=prediction.device, dtype=torch.long)
            if include_unmatched_target:
                num_unmatched_target_by_dim += torch.tensor(result_arrays.num_unmatched_target_by_dim, device=prediction.device, dtype=torch.long)

            current_instance_index += 1

    both_level_losses = None
    if filtration_type == FiltrationType.BOTHLEVELS:
        # Add sublevel and superlevel losses to get one loss per input batch element
        both_level_losses = {"sublevel": losses[:len(losses)//2], "superlevel": losses[len(losses)//2:]}
        losses = [(sublevel_loss + superlevel_loss) / 2 for sublevel_loss, superlevel_loss in zip(both_level_losses["sublevel"], both_level_losses["superlevel"])]
        losses_dicts = [{key: (sublevel_dict[key] + superlevel_dict[key])/2 for key in sublevel_dict.keys()}
                        for sublevel_dict, superlevel_dict in zip(losses_dicts[:len(losses_dicts)//2], losses_dicts[len(losses_dicts)//2:])]

    # Gather additional statistics outputs (no gradients!)
    with torch.no_grad():
        additional_outputs_dict = {
            **{f"Number of matches/Dimension {i} (relative={relative}": num_matched / batch_size for i, num_matched in enumerate(num_matched_by_dim)},
            **{f"Number of unmatched pairs in prediction/Dimension {i} (relative={relative}": num_unmatched / batch_size for i, num_unmatched in enumerate(num_unmatched_prediction_by_dim)},
            f"Betti matching loss (relative={relative})": torch.mean(torch.cat(losses)),
            f"Betti matching loss (matched, relative={relative})": torch.mean(torch.cat([losses_dict["Betti matching loss (matched)"] for losses_dict in losses_dicts])),
            f"Betti matching loss (unmatched prediction, relative={relative})": torch.mean(
                torch.cat([losses_dict["Betti matching loss (unmatched prediction)"] for losses_dict in losses_dicts])),
        }
        if include_unmatched_target:
            additional_outputs_dict = {
                **additional_outputs_dict,
                f"Betti matching loss (unmatched target, relative={relative})": torch.mean(
                    torch.cat([losses_dict["Betti matching loss (unmatched target)"] for losses_dict in losses_dicts])),
                **{f"Number of unmatched pairs in target/Dimension {i} (relative={relative}": num_unmatched / batch_size for i, num_unmatched in enumerate(num_unmatched_target_by_dim)},
            }
            

        if filtration_type == FiltrationType.SUBLEVEL:
            additional_outputs_dict["Betti matching loss (sublevel)"] = torch.mean(torch.cat(losses))
        elif filtration_type == FiltrationType.SUPERLEVEL:
            additional_outputs_dict["Betti matching loss (superlevel)"] = torch.mean(torch.cat(losses))
        elif filtration_type == FiltrationType.BOTHLEVELS:
            assert both_level_losses is not None
            additional_outputs_dict["Betti matching loss (sublevel)"] = torch.mean(torch.cat(both_level_losses["sublevel"]))
            additional_outputs_dict["Betti matching loss (superlevel)"] = torch.mean(torch.cat(both_level_losses["superlevel"]))

    return losses, additional_outputs_dict

def _betti_matching_loss_unmatched(
        unmatched_pairs: Float[Tensor, "M 2"],
        push_unmatched_to: PushUnmatchedTo,
        ) -> Float[Tensor, "one_dimension"]:
    # sum over ||(birth_pred_i, death_pred_i), 1/2*(birth_pred_i+death_pred_i, birth_pred_i+death_pred_i)||²
    # reformulated as (birth_pred_i^2 / 4 + death_pred_i^2/4 - birth_pred_i*death_pred_i/2)
    if push_unmatched_to == PushUnmatchedTo.DIAGONAL:
        return ((unmatched_pairs[:, 0] - unmatched_pairs[:, 1])**2).sum()
    elif push_unmatched_to == PushUnmatchedTo.ONE_ZERO:
            return 2 * ((unmatched_pairs[:, 0] - 1) ** 2 + unmatched_pairs[:, 1] ** 2).sum()
    elif push_unmatched_to == PushUnmatchedTo.DEATH_DEATH:
        return 2 * ((unmatched_pairs[:, 0] - unmatched_pairs[:, 1])**2).sum()

def _betti_matching_loss(prediction: Float[Tensor, "*spatial_dimensions"],
                         target: Float[Tensor, "*spatial_dimensions"],
                         betti_matching_result: betti_matching.return_types.BettiMatchingResult,
                         push_unmatched_to: PushUnmatchedTo = PushUnmatchedTo.DIAGONAL,
                         include_unmatched_target: bool = True # the correct loss includes unmatched target pairs; for training, they can be excluded (no gradient), for validation they should be included
                         ) -> Tuple[Float[Tensor, "one_dimension"],  Dict[LossOutputName, Tensor]]:

    (prediction_matches_birth_coordinates, prediction_matches_death_coordinates, target_matches_birth_coordinates,
     target_matches_death_coordinates, prediction_unmatched_birth_coordinates, prediction_unmatched_death_coordinates) = (
         [torch.tensor(array, device=prediction.device, dtype=torch.long) if array.strides[-1] > 0 else torch.zeros(0, 3, device=prediction.device, dtype=torch.long)
          for array in [betti_matching_result.prediction_matches_birth_coordinates, betti_matching_result.prediction_matches_death_coordinates,
                        betti_matching_result.target_matches_birth_coordinates, betti_matching_result.target_matches_death_coordinates,
                        betti_matching_result.prediction_unmatched_birth_coordinates, betti_matching_result.prediction_unmatched_death_coordinates,]])

    # (M, 2) tensor of matched persistence pairs for prediction
    prediction_matched_pairs = torch.stack([
        prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
        for coords in [prediction_matches_birth_coordinates, prediction_matches_death_coordinates]
    ], dim=1)
    # (M, 2) tensor of matched persistence pairs for target
    target_matched_pairs = torch.stack([
        target[tuple(coords[:, i] for i in range(coords.shape[1]))]
        for coords in [target_matches_birth_coordinates, target_matches_death_coordinates]
    ], dim=1)

    # (M, 2) tensor of unmatched persistence pairs for prediction
    prediction_unmatched_pairs = torch.stack([
        prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
        for coords in [prediction_unmatched_birth_coordinates, prediction_unmatched_death_coordinates]
    ], dim=1)

    if include_unmatched_target:
        target_unmatched_pairs = torch.stack([
            target[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [betti_matching_result.target_unmatched_birth_coordinates,
                           betti_matching_result.target_unmatched_death_coordinates]
        ], dim=1)
        

    # sum over ||(birth_pred_i, death_pred_i), (birth_target_i, death_target_i)||²
    loss_matched = 2 * ((prediction_matched_pairs - target_matched_pairs) ** 2).sum()

    # sum over ||(birth_pred_i, death_pred_i), 1/2*(birth_pred_i+death_pred_i, birth_pred_i+death_pred_i)||²
    # reformulated as (birth_pred_i^2 / 4 + death_pred_i^2/4 - birth_pred_i*death_pred_i/2)
    loss_unmatched_prediction = _betti_matching_loss_unmatched(prediction_unmatched_pairs, push_unmatched_to)
    
    # loss = (1 if include_unmatched_target else 0) * loss_matched + loss_unmatched_prediction
    loss = loss_matched + loss_unmatched_prediction
    losses_dict = {
        "Betti matching loss (matched)": loss_matched.reshape(1).detach(),
        "Betti matching loss (unmatched prediction)": loss_unmatched_prediction.reshape(1).detach(),
    }

    if include_unmatched_target:
        loss_unmatched_target = _betti_matching_loss_unmatched(target_unmatched_pairs, push_unmatched_to)
        loss += loss_unmatched_target
        losses_dict["Betti matching loss (unmatched target)"] = loss_unmatched_target.reshape(1).detach()

    return loss.reshape(1), losses_dict


def compute_wasserstein_loss(prediction: Float[Tensor, "batch channels *spatial_dimensions"],
                             target: Float[Tensor, "batch channels *spatial_dimensions"],
                             cpu_batch_size: int,
                             sigmoid=False,
                             relative=False,
                             comparison_operation: ComparisonOperation=ComparisonOperation.UNION,
                             filtration_type: FiltrationType=FiltrationType.SUPERLEVEL,
                             complex_construction: CubicalComplexConstruction=CubicalComplexConstruction.T_CONSTRUCTION
                             ) -> Tuple[List[torch.Tensor], Dict[str, Tensor]]:
    if complex_construction != CubicalComplexConstruction.T_CONSTRUCTION:
        raise NotImplementedError("Only T construction is currently supported")
    if comparison_operation == ComparisonOperation.INTERSECTION:
        raise NotImplementedError("The intersection comparison operation is currently not supported")

    if sigmoid:
        prediction = torch.sigmoid(prediction)
    
    if filtration_type == FiltrationType.SUPERLEVEL:
        # Using (1 - ...) to allow binary sorting optimization on the label, which expects values [0, 1]
        prediction = 1 - prediction
        target = 1 - target
    if filtration_type == FiltrationType.BOTHLEVELS:
        # Just duplicate the number of elements in the batch, once with sublevel, once with superlevel
        prediction = torch.concat([prediction, 1 - prediction])
        target = torch.concat([target, 1 - target])

    if relative:
        pad_value_prediction = prediction.min().item() # make sure to not propagate gradients here!
        pad_value_target = target.min().item()

        prediction = torch.nn.functional.pad(prediction, pad=[1 for _ in range(2 * (len(prediction.shape) - 2))], value=pad_value_prediction)
        target = torch.nn.functional.pad(target, pad=[1 for _ in range(2 * (len(target.shape) - 2))], value=pad_value_target)

    split_indices = np.arange(max(1, cpu_batch_size // 2), prediction.shape[0], max(1, cpu_batch_size // 2))
    predictions_list_numpy = np.split(prediction.detach().cpu().numpy().astype(np.float64), split_indices)
    targets_list_numpy = np.split(target.detach().cpu().numpy().astype(np.float64), split_indices)

    losses = []

    current_instance_index = 0
    for predictions_cpu_batch, targets_cpu_batch in zip(predictions_list_numpy, targets_list_numpy):
        predictions_cpu_batch, targets_cpu_batch = list(predictions_cpu_batch.squeeze(1)), list(targets_cpu_batch.squeeze(1))
        if not (all(a.data.contiguous for a in predictions_cpu_batch) and all(a.data.contiguous for a in targets_cpu_batch)):
            print("WARNING! Non-contiguous arrays encountered. Shape:", predictions_cpu_batch[0].shape)
            global ENCOUNTERED_NONCONTIGUOUS
            ENCOUNTERED_NONCONTIGUOUS=True
        predictions_cpu_batch = [np.ascontiguousarray(a) for a in predictions_cpu_batch]
        targets_cpu_batch = [np.ascontiguousarray(a) for a in targets_cpu_batch]

        barcodes_batch = betti_matching.compute_barcode(
            predictions_cpu_batch + targets_cpu_batch)
        barcodes_predictions, barcodes_targets = barcodes_batch[:len(barcodes_batch)//2], barcodes_batch[len(barcodes_batch)//2:]
        for barcode_prediction, barcode_target in zip(barcodes_predictions, barcodes_targets):
            losses.append(_wasserstein_loss(prediction[current_instance_index].squeeze(0), target[current_instance_index].squeeze(0), barcode_prediction, barcode_target))
            current_instance_index += 1

    additional_outputs_dict = {
        # **{f"Number of matches/Dimension {i}": num_matched for i, num_matched in enumerate(num_matched_by_dim)},
        # **{f"Number of unmatched pairs in prediction/Dimension {i}": num_unmatched for i, num_unmatched in enumerate(num_unmatched_prediction_by_dim)}
    }

    # if filtration_type == FiltrationType.SUBLEVEL:
    #     additional_outputs_dict["Betti matching loss (sublevel)"] = torch.mean(torch.cat(losses))
    # elif filtration_type == FiltrationType.SUPERLEVEL:
    #     additional_outputs_dict["Betti matching loss (superlevel)"] = torch.mean(torch.cat(losses))
    # elif filtration_type == FiltrationType.BOTHLEVELS:
    #     additional_outputs_dict["Betti matching loss (sublevel)"] = torch.mean(torch.cat(losses[:len(losses)//2]))
    #     additional_outputs_dict["Betti matching loss (superlevel)"] = torch.mean(torch.cat(losses[len(losses)//2:]))
    #     # Add sublevel and superlevel losses to get one loss per input batch element
    #     losses = [(sublevel_loss + superlevel_loss) / 2 for sublevel_loss, superlevel_loss in zip(losses[:len(losses)//2], losses[len(losses)//2:])]

    return losses, additional_outputs_dict

def _wasserstein_loss(prediction: Float[Tensor, "*spatial_dimensions"],
                      target: Float[Tensor, "*spatial_dimensions"],
                      barcode_result_prediction: betti_matching.return_types.BarcodeResult,
                      barcode_result_target: betti_matching.return_types.BarcodeResult,
                      ) -> Float[Tensor, "one_dimension"]:

    (prediction_birth_coordinates, prediction_death_coordinates, target_birth_coordinates, target_death_coordinates) = (
        [torch.tensor(array, device=prediction.device, dtype=torch.long) if len(array) > 0 else torch.zeros(0, 3, device=prediction.device, dtype=torch.long)
        for array in [barcode_result_prediction.birth_coordinates, barcode_result_prediction.death_coordinates,
                    barcode_result_target.birth_coordinates, barcode_result_target.death_coordinates]]
    )

    
    # (M, 2) tensor of persistence pairs for prediction
    prediction_pairs = torch.stack([
        prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
        for coords in [prediction_birth_coordinates, prediction_death_coordinates]
    ], dim=1)
    # (M, 2) tensor of persistence pairs for target
    target_pairs = torch.stack([
        target[tuple(coords[:, i] for i in range(coords.shape[1]))]
        for coords in [target_birth_coordinates, target_death_coordinates]
    ], dim=1)

    prediction_pairs = prediction_pairs.as_tensor() if isinstance(prediction_pairs, monai.data.meta_tensor.MetaTensor) else prediction_pairs
    target_pairs = target_pairs.as_tensor() if isinstance(target_pairs, monai.data.meta_tensor.MetaTensor) else target_pairs

    losses_matched_by_dim = []
    losses_unmatched_by_dim = []
    for prediction_pairs_dim, target_pairs_dim in zip(
        torch.split(prediction_pairs, barcode_result_prediction.num_pairs_by_dim.tolist()),
        torch.split(target_pairs, barcode_result_target.num_pairs_by_dim.tolist())
    ):
        _, matching = gudhi.wasserstein.wasserstein_distance(prediction_pairs_dim.detach().cpu(), target_pairs_dim.detach().cpu(),
                                                            matching=True, keep_essential_parts=False) # type: ignore
        matching = torch.tensor(matching.reshape(-1, 2), device=prediction.device, dtype=torch.long)

        matched_pairs = matching[(matching[:,0] >= 0) & (matching[:,1] >= 0)]
        loss_matched = ((prediction_pairs_dim[matched_pairs[:,0]] - target_pairs_dim[matched_pairs[:,1]])**2).sum() # type: ignore
        prediction_pairs_unmatched = prediction_pairs_dim[matching[matching[:,1] == -1][:,0]]
        target_pairs_unmatched = target_pairs_dim[matching[matching[:,0] == -1][:,1]]
        loss_unmatched = 0.5*(((prediction_pairs_unmatched[:,0] - prediction_pairs_unmatched[:,1])**2).sum()
                            + ((target_pairs_unmatched[:,0] - target_pairs_unmatched[:,1])**2).sum()) # type: ignore

        losses_matched_by_dim.append(loss_matched)
        losses_unmatched_by_dim.append(loss_unmatched)

    return (sum(losses_matched_by_dim) + sum(losses_unmatched_by_dim)).reshape(1)