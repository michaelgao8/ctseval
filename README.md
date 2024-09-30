# Clinical Time Series Evaluation (CTSEval)

CTSEval is a Python package for evaluating clinical time series predictions. It provides a set of metrics to assess the performance of time series prediction models, with a focus on clinical settings. (TODO: Link DOI)

## Installation

You can install CTSEval using pip:

```bash
pip install ctseval
```

## Usage

To use CTSEval in your project, follow these steps:

1. Import the package:

```python
import ctseval
```

2. The ctseval.compute_metrics function requires a list of trajectories. A trajectory is a dictionary with the following keys:

- `predicted_times`: A list of predicted times.
- `predicted_risks`: A list of predicted risks.
- `event_occurred`: Whether the event occurred.
- `event_time`: The time the event occurred.

The package contains a helper function to convert a `pandas` dataframe to a list of trajectories.
If your dataset has the following structure:

```
| episode_id | event_occurred | event_time | predicted_times | predicted_risks |
| ---------- | -------------- | ---------- | --------------- | --------------- |
| 1          | True           | 10.0       | 1               | 0.1             |
| 1          | True           | 10.0       | 2               | 0.2             |
| 2          | False          | 0.0        | 1               | 0.15            |
| 2          | False          | 0.0        | 2               | 0.25            |
| 2          | False          | 0.0        | 3               | 0.35            |
```

Then you can convert it to a compatible trajectory list:

`trajectories = ctseval.convert_df_to_trajectory_list(df, 'episode_id', 'event_occurred', 'event_time', 'predicted_times', 'predicted_risks')`


2. Use the `compute_metrics` function to evaluate your predictions:

```python
metrics = ctseval.compute_metrics(trajectories, snooze_window, detection_window)
```

Running with a snooze window of 0 will automatically route the function to an efficient implementation which will run fast. However, if snoozing is required, note that the implementation is slow (O(N^2)), as we describe. In the future, intermediate updates will help to estimate the total runtime (TODO).

3. Using the output, you can calculate metrics such as AUROC, AUPRC, etc. and provide the points on the ROC curve, precision-recall curve to plot etc. similar to the `sklearn` api.

```python
# Calculate scores
ctseval.auroc_score(metrics)
ctseval.auprc_score(metrics)
```

```python
# Get points on the curves
ctseval.roc_curve(metrics)
ctseval.precision_recall_curve(metrics)
```

