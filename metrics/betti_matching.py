from __future__ import annotations
import torch
import enum
import sys
import betti_matching
import numpy as np

import typing
if typing.TYPE_CHECKING:
    from typing import Tuple, List, Dict, Union, Literal, Callable, Generator
    from numpy.typing import NDArray
    LossOutputName = str
    from jaxtyping import Float
    from torch import Tensor


ENCOUNTERED_NONCONTIGUOUS = False

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


class BettiMatchingLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        relative=False,
        sigmoid=True,
        include_unmatched_target: bool = True,
        filtration_type: FiltrationType=FiltrationType.SUPERLEVEL,
        cpu_batch_size=32,
        push_unmatched_to: PushUnmatchedTo = PushUnmatchedTo.DIAGONAL
    ) -> None:
        super().__init__()
        self.relative = relative
        self.sigmoid = sigmoid
        self.filtration_type = filtration_type
        self.cpu_batch_size = cpu_batch_size
        self.push_unmatched_to = push_unmatched_to
        self.include_unmatched_target = include_unmatched_target

    def forward(self,
                input: Float[Tensor, "batch channels *spatial_dimensions"],
                target: Float[Tensor, "batch channels *spatial_dimensions"]
                ) -> Tuple[Float[Tensor, ""], Dict[LossOutputName, Float[Tensor, ""]]]:
        
        losses, additional_outputs_dict = compute_betti_matching_loss(
            input, target, cpu_batch_size=self.cpu_batch_size, include_unmatched_target=self.include_unmatched_target, sigmoid=self.sigmoid,
            filtration_type=self.filtration_type, relative=self.relative, push_unmatched_to=self.push_unmatched_to)

        dic: Dict[LossOutputName, torch.Tensor] = {**{
            'Betti matching loss': torch.mean(torch.cat(losses)),
        }, **additional_outputs_dict}
        loss = dic['Betti matching loss']
        return loss, dic