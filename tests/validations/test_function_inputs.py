import pytest 
import ctseval
import json

@pytest.fixture
def benchmark_20():
    with open("./benchmark_data/benchmark_trajectories_200.json", "r") as f:
        benchmark =  json.load(f)
    return benchmark[0:20]

def test_compute_metrics_integer_kwarg_inputs(benchmark_20):
    result = ctseval.compute_metrics(benchmark_20, snooze_window=0, detection_window=12)
    assert result is not None

def test_compute_metrics_int_non_kwarg_inputs(benchmark_20):
    result = ctseval.compute_metrics(benchmark_20, 0, 12)
    assert result is not None

def test_compute_metrics_float_kwarg_inputs(benchmark_20):
    result = ctseval.compute_metrics(benchmark_20, snooze_window=0.0, detection_window=12.0)
    assert result is not None

def test_compute_metrics_float_non_kwarg_inputs(benchmark_20):
    result = ctseval.compute_metrics(benchmark_20, 0.0, 12.0)
    assert result is not None
    
def test_compute_metrics_string_kwarg_inputs(benchmark_20):
    with pytest.raises(TypeError):
        ctseval.compute_metrics(benchmark_20, snooze_window="0", detection_window="12")

def test_compute_metrics_string_non_kwarg_inputs(benchmark_20):
    with pytest.raises(TypeError):
        ctseval.compute_metrics(benchmark_20, "0", "12")

def test_compute_metrics_none_kwarg_inputs(benchmark_20):
    with pytest.raises(TypeError):
        ctseval.compute_metrics(benchmark_20, snooze_window=None, detection_window=None)

def test_validate_trajectories_schema_valid(benchmark_20):
    assert ctseval.validate_trajectories_schema(benchmark_20) == True

def test_validate_trajectories_schema_missing_keys():
    invalid_trajectories = [
        {
            'predicted_times': [1.0, 2.0],
            'predicted_risks': [0.1, 0.2],
            'event_occurred': True
            # 'event_time' key is missing
        }
    ]
    with pytest.raises(ValueError, match="Each trajectory must contain the keys"):
        ctseval.validate_trajectories_schema(invalid_trajectories)

def test_validate_trajectories_schema_invalid_predicted_times():
    invalid_trajectories = [
        {
            'predicted_times': "not a list",
            'predicted_risks': [0.1, 0.2],
            'event_occurred': True,
            'event_time': 1.0
        }
    ]
    with pytest.raises(ValueError, match="The 'predicted_times' key must be a list of numbers"):
        ctseval.validate_trajectories_schema(invalid_trajectories)

def test_validate_trajectories_schema_invalid_predicted_risks():
    invalid_trajectories = [
        {
            'predicted_times': [1.0, 2.0],
            'predicted_risks': "not a list",
            'event_occurred': True,
            'event_time': 1.0
        }
    ]
    with pytest.raises(ValueError, match="The 'predicted_risks' key must be a list of numbers"):
        ctseval.validate_trajectories_schema(invalid_trajectories)

def test_validate_trajectories_schema_invalid_event_occurred():
    invalid_trajectories = [
        {
            'predicted_times': [1.0, 2.0],
            'predicted_risks': [0.1, 0.2],
            'event_occurred': "not a boolean",
            'event_time': 1.0
        }
    ]
    with pytest.raises(ValueError, match="The 'event_occurred' key must be a boolean"):
        ctseval.validate_trajectories_schema(invalid_trajectories)

def test_validate_trajectories_schema_invalid_event_time():
    invalid_trajectories = [
        {
            'predicted_times': [1.0, 2.0],
            'predicted_risks': [0.1, 0.2],
            'event_occurred': True,
            'event_time': "not a number"
        }
    ]
    with pytest.raises(ValueError, match="The 'event_time' key must be a number"):
        ctseval.validate_trajectories_schema(invalid_trajectories)
