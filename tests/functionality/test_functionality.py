import pytest 
import ctseval
import json

@pytest.fixture
def single_trajectory():
    return [{
        "predicted_times": [1.0, 2.0, 2.1],
        "predicted_risks": [0.1, 0.2, 0.2],
        "event_occurred": True,
        "event_time": 3.0
    }]

@pytest.fixture
def single_trajectory_w_snooze():
    return [{
        "predicted_times": [1.0, 2.0, 3.0, 4.0, 5.2], # If we snooze for 3 units, we should have another prediction_tp at 5.2 (if detection window is 8+)
        "predicted_risks": [0.1, 0.2, 0.9, 0.9, 0.2],
        "event_occurred": True,
        "event_time": 10.0
    }]

def test_single_trajectory_no_snooze(single_trajectory):
    result = ctseval.compute_metrics(single_trajectory, snooze_window=0, detection_window=1)
    expected_result = [
        {
            'threshold': 0.2,
            'episode_tp': 0,
            'episode_fp': 0,
            'episode_tn': 0, 
            'episode_fn': 1,  # None of the risk scores are above the maximum value, so it should just 1 false negative
            'prediction_tp': 0, # No risk scores above the threshold
            'prediction_fp': 0 
        },
        {
            'threshold': 0.1, 
            'episode_tp': 1, # At least 1 positive prediction within window 
            'episode_fp': 0, 
            'episode_tn': 0, 
            'episode_fn': 0, 
            'prediction_tp': 2, # Both of the risk scores above 0.1 are TPs
            'prediction_fp': 0 # No other risk scores above 0.1
        },
        {
            'threshold': -99999.0,
            'episode_tp': 1,
            'episode_fp': 0,
            'episode_tn': 0,
            'episode_fn': 0,
            'prediction_tp': 2, # All risk scores above the lowest threshold
            'prediction_fp': 1
        }
    ]
    assert result == expected_result

def test_single_trajectory_almost_no_snooze(single_trajectory):
    result_nonzero_snooze = ctseval.compute_metrics(single_trajectory, snooze_window=0.01, detection_window=1)
    result_zero_snooze = ctseval.compute_metrics(single_trajectory, snooze_window=0, detection_window=1)
    assert result_nonzero_snooze == result_zero_snooze

def test_single_trajectory_w_snooze(single_trajectory_w_snooze):
    result = ctseval.compute_metrics(single_trajectory_w_snooze, snooze_window=3, detection_window=8)
    expected_result = [
        {
            'threshold': 0.9, 
            'episode_tp': 0, 
            'episode_fp': 0, 
            'episode_tn': 0, 
            'episode_fn': 1, # There are no risk scores above 0.9, so it should just be 1 false negative
            'prediction_tp': 0, # There are no risk scores above 0.9
            'prediction_fp': 0 # There are no risk scores above 0.9
        }, 
        {
            'threshold': 0.2, 
            'episode_tp': 1, 
            'episode_fp': 0, 
            'episode_tn': 0, 
            'episode_fn': 0, 
            'prediction_tp': 1, # one of them gets snoozed 
            'prediction_fp': 0
        }, 
        {
            'threshold': 0.1, 
            'episode_tp': 1, 
            'episode_fp': 0, 
            'episode_tn': 0, 
            'episode_fn': 0, 
            'prediction_tp': 2, # one from at time 2, and one from at time 5.2 
            'prediction_fp': 0
        },
        {
            'threshold': -99999.0,
            'episode_tp': 1,
            'episode_fp': 0,
            'episode_tn': 0,
            'episode_fn': 0,
            'prediction_tp': 1, 
            'prediction_fp': 1 # 1 FP at time 1, then snooze 3, and then 1 TP at time 5.2
        }
    ]
    assert result == expected_result


