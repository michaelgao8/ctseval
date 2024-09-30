#include <Python.h>
#ifndef CTSEVAL_H
#define CTSEVAL_H

typedef struct {
    int len;
    double *predicted_times;
    double *predicted_risks;
    int event_occurred;
    double event_time;
} Trajectory;

typedef struct {
    double risk;
    int within_window;
    int ep_id;
} RiskScore;

int convert_to_trajectory(PyObject *traj_obj, Trajectory *traj);
int compute_metrics(PyObject *trajectories_obj, double snooze_window, double detection_window, PyObject *result_list, int verbosity);
int compare_risk_scores(const void *a, const void *b);
int convert_to_c_array(PyObject *input, double **output, int *len);

#endif // CTSEVAL_H
