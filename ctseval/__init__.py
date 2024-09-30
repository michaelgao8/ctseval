from functools import wraps
from typing import List, Dict, Optional
from ctseval.utils import (validate_trajectories_schema as _validate_trajectories_schema,
                    auprc_score as _auprc_score,
                    precision_recall_curve as _precision_recall_curve,
                    recall_at_fixed_precision as _recall_at_fixed_precision,
                    precision_at_fixed_recall as _precision_at_fixed_recall,
                    roc_curve as _roc_curve,
                    auroc_score as _auroc_score,
                    _extract_attributes,
                    convert_df_to_trajectory_list as _convert_df_to_trajectory_list)
from ctseval._ctseval import compute_metrics_c

def compute_metrics(trajectories: List[Dict], snooze_window: float, detection_window: float, verbosity: Optional[int] = 1) -> List[Dict]:
    """
    Compute metrics for clinical time series predictions.
    
    This function calculates various metrics for evaluating the performance of clinical time series predictions,
    including episode-level and prediction-level metrics.

    Args:
        trajectories (List[Dict]): List of trajectory dictionaries. Each dictionary should contain:
            - predicted_times (List[float]): List of prediction times.
            - predicted_risks (List[float]): List of predicted risk scores.
            - event_occurred (bool): Whether an event occurred or not.
            - event_time (float): Time of the event, if it occurred.
        snooze_window (float): The snooze window duration in units of time.
        detection_window (float): The detection window duration in units of time.
        verbosity (Optional[int], optional): Verbosity level. Defaults to 1.
    
    Returns:
        List[Dict]: List of dictionaries with episode-level and prediction-level metrics.
            Each dictionary contains the following keys:
            - threshold (float): The risk threshold used for this set of metrics.
            - episode_tp (int): True positive episodes.
            - episode_fp (int): False positive episodes.
            - episode_tn (int): True negative episodes.
            - episode_fn (int): False negative episodes.
            - prediction_tp (int): True positive predictions.
            - prediction_fp (int): False positive predictions.

    Raises:
        ValueError: If the trajectories schema is invalid.

    Example:
        >>> trajectories = [
        ...     {
        ...         "predicted_times": [1.0, 2.0, 3.0],
        ...         "predicted_risks": [0.1, 0.2, 0.3],
        ...         "event_occurred": True,
        ...         "event_time": 2.5
        ...     },
        ...     {
        ...         "predicted_times": [1.0, 2.0, 3.0],
        ...         "predicted_risks": [0.05, 0.15, 0.25],
        ...         "event_occurred": False,
        ...         "event_time": 0
        ...     }
        ... ]
        >>> metrics = compute_metrics(trajectories, snooze_window=1.0, detection_window=1.0)
        >>> print(metrics[0])
        {'threshold': 0.3, 'episode_tp': 1, 'episode_fp': 0, 'episode_tn': 1, 'episode_fn': 0, 'prediction_tp': 1, 'prediction_fp': 0}
    """
    validate_trajectories_schema(trajectories)
    results = compute_metrics_c(trajectories, snooze_window, detection_window, verbosity)
    return results

@wraps(_validate_trajectories_schema)
def validate_trajectories_schema(*args, **kwargs):
    return _validate_trajectories_schema(*args, **kwargs)

@wraps(_auprc_score)
def auprc_score(*args, **kwargs):
    return _auprc_score(*args, **kwargs)

@wraps(_precision_recall_curve)
def precision_recall_curve(*args, **kwargs):
    return _precision_recall_curve(*args, **kwargs)

@wraps(_recall_at_fixed_precision)
def recall_at_fixed_precision(*args, **kwargs):
    return _recall_at_fixed_precision(*args, **kwargs)

@wraps(_precision_at_fixed_recall)
def precision_at_fixed_recall(*args, **kwargs):
    return _precision_at_fixed_recall(*args, **kwargs)

@wraps(_roc_curve)
def roc_curve(*args, **kwargs):
    return _roc_curve(*args, **kwargs)

@wraps(_auroc_score)
def auroc_score(*args, **kwargs):
    return _auroc_score(*args, **kwargs)

@wraps(_extract_attributes)
def extract_attributes(*args, **kwargs):
    return _extract_attributes(*args, **kwargs)

@wraps(_convert_df_to_trajectory_list)
def convert_df_to_trajectory_list(*args, **kwargs):
    return _convert_df_to_trajectory_list(*args, **kwargs)
    

__all__ = ['compute_metrics', 'validate_trajectories_schema', 'auprc_score', 'precision_recall_curve',
           'recall_at_fixed_precision', 'precision_at_fixed_recall', 'roc_curve', 'auroc_score',
           'extract_attributes', 'convert_df_to_trajectory_list']