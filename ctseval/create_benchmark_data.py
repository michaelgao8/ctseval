import pandas as pd 
import numpy as np
import json
from .utils import _extract_attributes, convert_df_to_trajectory_list


if __name__ == '__main__':
    NUM_TRAJ = 20000
    event_times = np.random.gamma(shape=200, scale=1, size=NUM_TRAJ)

    # Generate trajectories

    trajectory_dfs = []
    for i in range(NUM_TRAJ):
        # Generate the trajectory length
        traj_len = np.random.randint(low=1, high=3000)
        # Generate the times
        pred_times = np.sort(np.random.uniform(low=0, high=event_times[i], size=traj_len))
        # Generate the risk scores
        pred_risks = np.random.uniform(low=0, high=1, size=traj_len)
        # Generate true or false for whether the event occurred or not
        event_occurred = np.random.rand() > 0.5
        
        traj_df = pd.DataFrame({
            "id": i,
            "event_occurred": event_occurred,
            "event_time": event_times[i],
            "predicted_times": pred_times,
            "predicted_risks": pred_risks
        })
        
        trajectory_dfs.append(traj_df)

    trajectory_df = pd.concat(trajectory_dfs)
    traj_list = convert_df_to_trajectory_list(trajectory_df, 'id', 'event_occurred', 'event_time', 'predicted_times', 'predicted_risks')

    with open("../benchmark_data/benchmark_trajectories_200.json", "w") as f:
        json.dump(traj_list[0:200], f, indent=2)

    with open("../benchmark_data/benchmark_trajectories_2000.json", "w") as f:
        json.dump(traj_list[0:2000], f, indent=2)
        
    with open("../benchmark_data/benchmark_trajectories_20000.json", "w") as f:
        json.dump(traj_list, f, indent=2)