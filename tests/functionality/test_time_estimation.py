import time
import sys
# Ensure ctseval is discoverable, assuming the script is run from the repo root
# or that ctseval has been installed (e.g., via python setup.py develop)
try:
    import ctseval
except ImportError:
    # Simple attempt to adjust path if running script directly from tests/functionality
    # and ctseval is not yet installed. This is a convenience for development.
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    import ctseval

def generate_test_trajectories():
    """
    Generates sample trajectory data for testing time estimation.
    Features:
    - Multiple trajectories.
    - Events occurring in some trajectories.
    - predicted_risks with blocks of identical scores and then unique scores.
    - predicted_times allowing for true/false positives.
    """
    trajectories = []

    # Trajectory 1: Event occurs, many initial identical low risks, then rising unique risks
    traj1_risks = [0.1] * 50 + [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    traj1_times = [float(i+1) for i in range(len(traj1_risks))]
    trajectories.append({
        "predicted_times": traj1_times,
        "predicted_risks": traj1_risks,
        "event_occurred": True,
        "event_time": float(len(traj1_risks) - 5), # Event towards the end
    })

    # Trajectory 2: No event, mix of risks
    traj2_risks = [0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.2, 0.25, 0.3]
    traj2_times = [float(i+1) for i in range(len(traj2_risks))]
    trajectories.append({
        "predicted_times": traj2_times,
        "predicted_risks": traj2_risks,
        "event_occurred": False,
        "event_time": 0.0,
    })

    # Trajectory 3: Event occurs, risks are generally higher and unique
    traj3_risks = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    traj3_times = [float(i+1) for i in range(len(traj3_risks))]
    trajectories.append({
        "predicted_times": traj3_times,
        "predicted_risks": traj3_risks,
        "event_occurred": True,
        "event_time": float(len(traj3_risks) - 2),
    })

    # Trajectory 4: Event occurs, only a few high risks
    traj4_risks = [0.7, 0.7, 0.8, 0.9, 0.9]
    traj4_times = [float(i+1) for i in range(len(traj4_risks))]
    trajectories.append({
        "predicted_times": traj4_times,
        "predicted_risks": traj4_risks,
        "event_occurred": True,
        "event_time": 3.0,
    })


    # Add more trajectories to make the computation take a noticeable amount of time
    # Let's increase this significantly to ensure non-zero execution time for estimation.
    # Each trajectory adds N_RISK_POINTS * N_TRAJECTORIES_INNER_LOOP work roughly.
    # The outer loop is over unique risk scores.
    # Snooze window processing is roughly N_UNIQUE_RISKS * N_TRAJECTORIES * AVG_TRAJ_LEN

    # Increase number of "filler" trajectories and points per trajectory
    num_filler_trajectories = 200  # Was 20
    points_per_filler_trajectory = 500 # Was 100

    print(f"Generating {num_filler_trajectories} filler trajectories with {points_per_filler_trajectory} points each...")

    for i in range(num_filler_trajectories):
        # Make risks somewhat varied to contribute to unique risk scores
        # but also include some duplicates by stepping coarsely.
        risks = [0.1 + (j*0.001) for j in range(points_per_filler_trajectory)]
        times = [float(k+1) for k in range(len(risks))]
        trajectories.append({
            "predicted_times": times,
            "predicted_risks": risks,
            "event_occurred": (i % 3 == 0), # Some events
            "event_time": float(len(risks) - 10) if (i % 3 == 0) else 0.0,
        })

    return trajectories

if __name__ == "__main__":
    print("Generating test trajectories...")
    trajectories_data = generate_test_trajectories()

    snooze_window = 5.0
    detection_window = 10.0
    # Verbosity is handled by the C code's printf, not a parameter to compute_metrics
    # in the Python wrapper for ctseval. The C function takes verbosity, but
    # py_compute_metrics_c in _ctseval.c sets it to 1 by default if not passed,
    # and the C compute_metrics function uses printf directly.

    print(f"Total trajectories: {len(trajectories_data)}")
    all_risk_scores_count = sum(len(t['predicted_risks']) for t in trajectories_data)
    print(f"Total individual risk scores: {all_risk_scores_count}")

    # Flatten all risk scores and count unique ones to compare with C output
    all_risks_flat = []
    for t_data in trajectories_data:
        all_risks_flat.extend(t_data['predicted_risks'])

    unique_risk_values = sorted(list(set(all_risks_flat)))
    print(f"Total unique risk values in generated data: {len(unique_risk_values)}")


    print(f"\nCalling ctseval.compute_metrics with snooze_window={snooze_window}, detection_window={detection_window}")
    print("Observe the time estimation printouts from the C extension:")
    print("-" * 60)

    start_time = time.time()
    # The verbosity parameter is passed to the C extension, which then passes it to compute_metrics
    # Let's try passing it explicitly, if the wrapper supports it.
    # The wrapper py_compute_metrics_c has: static char *kwlist[] = {"trajectories", "snooze_window", "detection_window", "verbosity", NULL};
    # And it parses: if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|i", kwlist, ... , &verbosity))
    # So, we can pass verbosity. A verbosity of 0 might suppress prints. We want prints.
    # The C code currently prints regardless of a verbosity flag for the time estimation part.
    # The `verbosity` argument in `compute_metrics` C function is not used for the time printf.
    # The `py_compute_metrics_c` takes a verbosity argument, but it's not used to control this specific printf.
    # The printf for time estimation is unconditional in the C code when snooze_window > 0.

    metrics_results = ctseval.compute_metrics(
        trajectories_data,
        snooze_window=snooze_window,
        detection_window=detection_window
        # verbosity=1 # Not strictly needed for time prints, but good practice
    )
    end_time = time.time()

    print("-" * 60)
    print(f"ctseval.compute_metrics call completed in {end_time - start_time:.2f} seconds.")
    print(f"Number of threshold points generated: {len(metrics_results)}")

    # print("\nSample of results (first 3 and last 3):")
    # if len(metrics_results) > 6:
    #     for i in list(range(3)) + list(range(len(metrics_results)-3, len(metrics_results))):
    #         print(metrics_results[i])
    # else:
    #     for result in metrics_results:
    #         print(result)

    print("\nTest script finished. Please check the console output for time estimations.")
