from typing import List, Dict, Tuple
import pandas as pd

def _recall(metric: Dict, total_positive_episodes: float) -> float:
    """
    Calculate recall for a single metric.
    
    Args:
        metric: A dictionary (obtained from compute_metrics) containing 'episode_tp'.
        total_positive_episodes: The total number of positive episodes.
    Returns:
        The recall value.
    """
    return metric['episode_tp'] / total_positive_episodes

def _precision(metric: Dict) -> float:
    """
    Calculate precision for a single metric.
    
    Args:
        metric: A dictionary (obtained from compute_metrics) containing 'prediction_tp' and 'prediction_fp'.
    Returns:
        The precision value or None if division by zero occurs.
    """
    try:
        return metric['prediction_tp'] / (metric['prediction_tp'] + metric['prediction_fp'])
    except ZeroDivisionError:
        return None

def _fpr(metric: Dict, total_negative_episodes: float) -> float:
    """
    Calculate false positive rate for a single metric.

    Args:
        metric: A dictionary (obtained from compute_metrics) containing 'episode_fp' and 'episode_tn'.
        total_negative_episodes: The total number of negative episodes.
    Returns:
        The false positive rate.
    """
    return metric['episode_fp'] / total_negative_episodes

def validate_trajectories_schema(trajectories: List[Dict]):
    """
    Validate the schema of a list of trajectory dictionaries.

    Args:
        trajectories: A list of trajectory dictionaries.
    Raises:
        ValueError: If the schema is invalid.
    """
    required_keys = {'predicted_times', 'predicted_risks', 'event_occurred', 'event_time'}
    
    for traj in trajectories:
        if not isinstance(traj, Dict):
            raise ValueError("Each trajectory must be a dictionary.")
        
        if not required_keys.issubset(traj.keys()):
            raise ValueError(f"Each trajectory must contain the keys: {required_keys}")
        
        if not isinstance(traj['predicted_times'], List) or not all(isinstance(t, (int, float)) for t in traj['predicted_times']):
            raise ValueError("The 'predicted_times' key must be a list of numbers.")
        
        if len(traj['predicted_times']) < 1:
            raise ValueError("The 'predicted_times' list must contain at least one element.")
        
        if not isinstance(traj['predicted_risks'], List) or not all(isinstance(r, (int, float)) for r in traj['predicted_risks']):
            raise ValueError("The 'predicted_risks' key must be a list of numbers.")
        
        if len(traj['predicted_risks']) < 1:
            raise ValueError("The 'predicted_risks' list must contain at least one element.")
        
        if not isinstance(traj['event_occurred'], bool):
            raise ValueError("The 'event_occurred' key must be a boolean.")
        
        if not isinstance(traj['event_time'], (int, float)):
            raise ValueError("The 'event_time' key must be a number.")
        
        # Coerce predicted_times, predicted_risks, and event_time to floats
        traj['predicted_times'] = [float(t) for t in traj['predicted_times']]
        traj['predicted_risks'] = [float(r) for r in traj['predicted_risks']]
        traj['event_time'] = float(traj['event_time'])
        
    return True

def auprc_score(metrics: List[Dict]) -> float:
    """
    Compute the area under the precision-recall curve, using the lower trapezoidal estimator, which is a point-estimator for the area under the precision-recall curve.
    This is chosen due to its performance as an estimator as described in Boyd(2013) and its ability to be computed even when the precision-recall is not necessarily monotonic.

    Args:
        metrics: A list of metric dictionaries.
    Returns:
        The average precision score.
    """
    recall_precision_dict = {}
    total_positive_episodes = metrics[-1]['episode_tp'] + metrics[-1]['episode_fn']
    for m in metrics:
        # Compute recall value
        r = _recall(m, total_positive_episodes)
        
        # Compute precision value
        p = _precision(m)

        if p is not None:
            if r not in recall_precision_dict:
                recall_precision_dict[r] = {'pmin': p, 'pmax': p}
            else:
                recall_precision_dict[r]['pmin'] = min(recall_precision_dict[r]['pmin'], p)
                recall_precision_dict[r]['pmax'] = max(recall_precision_dict[r]['pmax'], p)
    # Sort by recall
    sorted_recalls = sorted(recall_precision_dict.keys())
    precisions_min = [recall_precision_dict[r]['pmin'] for r in sorted_recalls]
    precisions_max = [recall_precision_dict[r]['pmax'] for r in sorted_recalls]

    ap_sum = 0
    for i in range(0, len(sorted_recalls)-1):
        ap_sum += (precisions_min[i] + precisions_max[i+1])/ 2 * (sorted_recalls[i+1] - sorted_recalls[i])

    return ap_sum

def precision_recall_curve(metrics: List[Dict]) -> Tuple[List[float], List[float]]:
    """
    Generate the precision-recall curve data. Note that this may alias sklearn.metrics.precision_recall_curve.

    Args:
        metrics: A list of metric dictionaries.
    Returns:
        A tuple containing two lists: recalls and precisions.
    """
    xs = []
    ys = []
    total_positive_episodes = metrics[-1]['episode_tp'] + metrics[-1]['episode_fn']
    
    for m in metrics:
        curr_prec = _precision(m)
        if curr_prec is not None:
            curr_recall = _recall(m, total_positive_episodes)
            xs.append(curr_recall)
            ys.append(curr_prec)
    return xs, ys

def recall_at_fixed_precision(metrics: List[Dict], target_precision: float) -> float:
    """
    Find the closest recall value at the specified precision.

    Args:
        metrics: List of dictionaries containing 'episode_tp', 'episode_fp', 'episode_fn', and 'prediction_tp'.
        target_precision: The precision value to find the closest recall for.
    Returns:
        The recall value closest to the specified precision, or None if not found.
    """
    total_positive_episodes = metrics[-1]['episode_tp'] + metrics[-1]['episode_fn']

    closest_recall = None
    smallest_diff = float('inf')

    for m in metrics:
        curr_prec = _precision(m)
        if curr_prec is not None:
            diff = abs(curr_prec - target_precision)
            if diff < smallest_diff:
                smallest_diff = diff
                closest_recall = _recall(m, total_positive_episodes)

    return closest_recall

def precision_at_fixed_recall(metrics: List[Dict], target_recall: float) -> float:
    """
    Find the closest precision value at the specified recall.

    Args:
        metrics: List of dictionaries containing 'episode_tp', 'episode_fp', 'episode_fn', and 'prediction_tp'.
        target_recall: The recall value to find the closest precision for.
    Returns:
        The precision value closest to the specified recall, or None if not found.
    """
    total_positive_episodes = metrics[-1]['episode_tp'] + metrics[-1]['episode_fn']

    closest_precision = None
    smallest_diff = float('inf')

    for m in metrics:
        curr_prec = _precision(m)
        if curr_prec is not None:
            curr_recall = _recall(m, total_positive_episodes)
            diff = abs(curr_recall - target_recall)
            if diff < smallest_diff:
                smallest_diff = diff
                closest_precision = curr_prec

    return closest_precision

def roc_curve(metrics: List[Dict]) -> Tuple[List[float], List[float]]:
    """
    Generate the ROC curve data.

    Args:
        metrics: A list of metric dictionaries. Note that this may alias sklearn.metrics.roc_curve.
    Returns:
        A tuple containing two lists: false positive rates and true positive rates.
    """
    fprs = []
    tprs = []
    total_positive_episodes = metrics[-1]['episode_tp'] + metrics[-1]['episode_fn']
    total_negative_episodes = metrics[-1]['episode_fp'] + metrics[-1].get('episode_tn', 0)

    for m in metrics:
        curr_tpr = _recall(m, total_positive_episodes)
        curr_fpr = _fpr(m, total_negative_episodes)
        fprs.append(curr_fpr)
        tprs.append(curr_tpr)
    return fprs, tprs

def _calculate_trapezoid_area(x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Calculate the area of a trapezoid.

    Args:
        x1: x-coordinate of the first point
        x2: x-coordinate of the second point
        y1: y-coordinate of the first point
        y2: y-coordinate of the second point
    Returns:
        Area of the trapezoid
    """
    return 0.5 * (x2 - x1) * (y1 + y2)

def auroc_score(metrics: List[Dict]) -> float:
    """
    Compute the Area Under the ROC Curve (AUROC).

    Args:
        metrics: A list of metric dictionaries.
    Returns:
        The AUROC value.
    """
    fprs, tprs = roc_curve(metrics)
    # Sort the FPR and TPR for proper integration
    sorted_pairs = sorted(zip(fprs, tprs))
    sorted_fprs, sorted_tprs = zip(*sorted_pairs)

    # Compute the Area Under the Curve using the trapezoidal rule
    auroc = 0.0
    for i in range(1, len(sorted_fprs)):
        auroc += _calculate_trapezoid_area(sorted_fprs[i-1], sorted_fprs[i], sorted_tprs[i-1], sorted_tprs[i])

    return auroc

def _extract_attributes(groupbydf: 'pd.core.groupby.generic.DataFrameGroupBy',
                        event_occurred_col: str, 
                        event_time_col: str, 
                        predicted_times_col: str, 
                        predicted_risks_col: str) -> Dict:
    """
    Extract attributes from a grouped dataframe. A helper function used for converting a dataframe to a list of trajectories.

    Args:
        groupbydf: A dataframe grouped by episode_id.
        event_occurred_col: The column name for event occurrence.
        event_time_col: The column name for event time.
        predicted_times_col: The column name for predicted times.
        predicted_risks_col: The column name for predicted risks.
    Returns:
        A dictionary containing the extracted attributes.
    """
    result_dict = {}
    result_dict['event_occurred'] = bool(groupbydf[event_occurred_col].iloc[0])
    result_dict['event_time'] = groupbydf[event_time_col].iloc[0]
    result_dict['predicted_times'] = groupbydf[predicted_times_col].tolist()
    result_dict['predicted_risks'] = groupbydf[predicted_risks_col].tolist()
    return result_dict
    
def convert_df_to_trajectory_list(df: 'pd.DataFrame', 
                                  episode_id_col: str, 
                                  event_occurred_col: str, 
                                  event_time_col: str, 
                                  predicted_times_col: str, 
                                  predicted_risks_col: str) -> List[Dict]:
    """
    Convert a dataframe to a list of trajectory dictionaries.

    Args:
        df: A dataframe.
        episode_id_col: The column name for episode id.
        event_occurred_col: The column name for event occurrence.
        event_time_col: The column name for event time.
        predicted_times_col: The column name for predicted times.
        predicted_risks_col: The column name for predicted risks.
    Returns:
        A list of trajectory dictionaries.
    """
    traj_list = []
    for i, df in df.groupby(episode_id_col):
        traj_list.append(_extract_attributes(df, event_occurred_col, event_time_col, predicted_times_col, predicted_risks_col))
    return traj_list