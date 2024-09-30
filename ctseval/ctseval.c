#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Python.h>
#include "ctseval.h"
#include <time.h>


void free_trajectory(Trajectory *traj) {
    if (traj->predicted_times) {
        free(traj->predicted_times);
        traj->predicted_times = NULL;
    }
    if (traj->predicted_risks) {
        free(traj->predicted_risks);
        traj->predicted_risks = NULL;
    }
}

// Function to convert Python trajectory to C structure
int convert_to_trajectory(PyObject *traj_obj, Trajectory *traj) {
    PyObject *predicted_times_obj = PyDict_GetItemString(traj_obj, "predicted_times");
    PyObject *predicted_risks_obj = PyDict_GetItemString(traj_obj, "predicted_risks");
    PyObject *event_occurred_obj = PyDict_GetItemString(traj_obj, "event_occurred");
    PyObject *event_time_obj = PyDict_GetItemString(traj_obj, "event_time");

    if (!predicted_times_obj || !predicted_risks_obj || !event_occurred_obj || !event_time_obj) {
        PyErr_SetString(PyExc_KeyError, "Missing keys in trajectory object");
        return -1;
    }

    if (!PyList_Check(predicted_times_obj) || !PyList_Check(predicted_risks_obj)) {
        PyErr_SetString(PyExc_TypeError, "predicted_times and predicted_risks must be lists");
        return -1;
    }

    traj->len = PyList_Size(predicted_times_obj);
    if (traj->len < 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid size for predicted_times");
        return -1;
    }

    traj->predicted_times = (double *)malloc(traj->len * sizeof(double));
    traj->predicted_risks = (double *)malloc(traj->len * sizeof(double));
    if (traj->predicted_times == NULL || traj->predicted_risks == NULL) {
        free_trajectory(traj);  // Free any allocated memory before returning
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for trajectory arrays");
        return -1;
    }

    for (int j = 0; j < traj->len; j++) {
        PyObject *time_item = PyList_GetItem(predicted_times_obj, j);
        PyObject *risk_item = PyList_GetItem(predicted_risks_obj, j);

        if (time_item == NULL || risk_item == NULL) {
            free_trajectory(traj); // Free allocated memory before returning
            PyErr_SetString(PyExc_IndexError, "Invalid index when accessing lists");
            return -1;
        }

        if (!PyFloat_Check(time_item) || !PyFloat_Check(risk_item)) {
            free_trajectory(traj);  // Free allocated memory before returning
            PyErr_SetString(PyExc_TypeError, "predicted_times and predicted_risks must contain floats");
            return -1;
        }

        traj->predicted_times[j] = PyFloat_AsDouble(time_item);
        traj->predicted_risks[j] = PyFloat_AsDouble(risk_item);
    }

    if (!PyBool_Check(event_occurred_obj)) {
        free_trajectory(traj);  // Free allocated memory before returning
        PyErr_SetString(PyExc_TypeError, "event_occurred must be a boolean");
        return -1;
    }
    traj->event_occurred = PyObject_IsTrue(event_occurred_obj);

    if (!PyFloat_Check(event_time_obj)) {
        free_trajectory(traj);  // Free allocated memory before returning
        PyErr_SetString(PyExc_TypeError, "event_time must be a float");
        return -1;
    }
    traj->event_time = PyFloat_AsDouble(event_time_obj);

    return 0;
}


void get_valid_times(double *predicted_times, double *predicted_risks, int len, double threshold, double snooze_window, double *result, int *result_len) {
    double snooze_boundary = -1.0;
    int count = 0;

    for (int i = 0; i < len; i++) {
        if (predicted_times[i] > snooze_boundary && predicted_risks[i] > threshold) {
            result[count++] = predicted_times[i];
            snooze_boundary = predicted_times[i] + snooze_window;
        }
    }

    *result_len = count;
}

void get_prediction_level_metrics(double *positive_prediction_times, int len, double detection_window, double event_time, int *num_tp, int *num_fp) {
    *num_tp = 0;
    *num_fp = 0;

    for (int i = 0; i < len; i++) {
        if (positive_prediction_times[i] < event_time && positive_prediction_times[i] >= (event_time - detection_window)) {
            (*num_tp)++;
        } else {
            (*num_fp)++;
        }
    }
}

// Comparison function for qsort
int compare_risk_scores(const void *a, const void *b) {
    double diff = ((RiskScore *)b)->risk - ((RiskScore *)a)->risk;
    return (diff > 0) - (diff < 0);
}

void compute_metrics_no_snooze(Trajectory *trajectories, int num_trajectories, double detection_window, PyObject *result_list) {
    if (num_trajectories <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid number of trajectories");
        return;
    }

    // Allocate memory for arrays
    int *event_occurs_episodes = (int *)calloc(num_trajectories, sizeof(int));
    int *all_episodes = (int *)calloc(num_trajectories, sizeof(int));
    if (!event_occurs_episodes || !all_episodes) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for episode arrays");
        free(event_occurs_episodes);
        free(all_episodes);
        return;
    }

    int event_occurs_count = 0;
    int all_count = 0;

    // Count total number of risk scores
    int total_risk_scores = 0;
    for (int i = 0; i < num_trajectories; i++) {
        total_risk_scores += trajectories[i].len;
    }

    RiskScore *risk_scores = (RiskScore *)calloc(total_risk_scores, sizeof(RiskScore));
    if (!risk_scores) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for risk_scores");
        free(event_occurs_episodes);
        free(all_episodes);
        return;
    }
    int risk_scores_count = 0;

    for (int i = 0; i < num_trajectories; i++) {
        Trajectory *traj = &trajectories[i];
        all_episodes[all_count++] = i;

        if (traj->event_occurred) {
            event_occurs_episodes[event_occurs_count++] = i;
            for (int j = 0; j < traj->len; j++) {
                int within_detection_window = (traj->event_time - traj->predicted_times[j]) <= detection_window;
                risk_scores[risk_scores_count++] = (RiskScore){traj->predicted_risks[j], within_detection_window, i};
            }
        } else {
            for (int j = 0; j < traj->len; j++) {
                risk_scores[risk_scores_count++] = (RiskScore){traj->predicted_risks[j], 0, i};
            }
        }
    }

    // Sort risk scores in descending order
    qsort(risk_scores, risk_scores_count, sizeof(RiskScore), compare_risk_scores);

    int *positive_prediction_episodes = (int *)malloc(num_trajectories * sizeof(int));
    if (positive_prediction_episodes == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for positive_prediction_episodes");
        free(event_occurs_episodes);
        free(all_episodes);
        free(risk_scores);
        return;
    }
    int *negative_prediction_episodes = (int *)malloc(num_trajectories * sizeof(int));
    if (negative_prediction_episodes == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for negative_prediction_episodes");
        free(event_occurs_episodes);
        free(all_episodes);
        free(risk_scores);
        free(positive_prediction_episodes);
        return;
    }
    int positive_count = 0;
    int negative_count = 0;

    int episode_tp = 0, episode_fp = 0, episode_fn = 0, episode_tn = 0;
    int prediction_tp = 0, prediction_fp = 0;

    double previous_risk = -1.0;  // Initialize with an impossible risk value

    for (int i = 0; i < risk_scores_count; i++) {
        RiskScore *score = &risk_scores[i];

        // Only process and append results if the risk score is different from the previous one
        if (score->risk != previous_risk) {
            episode_fn = event_occurs_count - episode_tp;
            episode_tn = all_count - event_occurs_count - episode_fp;
            // Push results on R
            PyObject *result_dict = Py_BuildValue("{s:d, s:i, s:i, s:i, s:i, s:i, s:i}",
                                            "threshold", score->risk,
                                            "episode_tp", episode_tp,
                                            "episode_fp", episode_fp,
                                            "episode_tn", episode_tn,
                                            "episode_fn", episode_fn,
                                            "prediction_tp", prediction_tp,
                                            "prediction_fp", prediction_fp);
            PyList_Append(result_list, result_dict);
            Py_DECREF(result_dict);  // Decrease reference count

            // Set the previous risk to the current risk
            previous_risk = score->risk;
        }
    
        if (score->within_window) {
            prediction_tp++;
            int found = 0;
            for (int j = 0; j < positive_count; j++) {
                if (positive_prediction_episodes[j] == score->ep_id) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                positive_prediction_episodes[positive_count++] = score->ep_id;
                for (int j = 0; j < event_occurs_count; j++) {
                    if (event_occurs_episodes[j] == score->ep_id) {
                        episode_tp++;
                        break;
                    }
                }
            }
        } else {
            prediction_fp++;
            int found = 0;
            for (int j = 0; j < negative_count; j++) {
                if (negative_prediction_episodes[j] == score->ep_id) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                negative_prediction_episodes[negative_count++] = score->ep_id;
                int is_event_occurred = 0;
                for (int j = 0; j < event_occurs_count; j++) {
                    if (event_occurs_episodes[j] == score->ep_id) {
                        is_event_occurred = 1;
                        break;
                    }
                }
                if (!is_event_occurred) {
                    episode_fp++;
                }
            }   
        }
    }
    episode_fn = event_occurs_count - episode_tp;
    episode_tn = all_count - event_occurs_count - episode_fp;
    // Push final results with a large negative number as threshold
    PyObject *final_result_dict = Py_BuildValue("{s:d, s:i, s:i, s:i, s:i, s:i, s:i}",
                                    "threshold", -99999.0,
                                    "episode_tp", episode_tp,
                                    "episode_fp", episode_fp,
                                    "episode_tn", episode_tn,
                                    "episode_fn", episode_fn,
                                    "prediction_tp", prediction_tp,
                                    "prediction_fp", prediction_fp);
    PyList_Append(result_list, final_result_dict);
    Py_DECREF(final_result_dict);  // Decrease reference count

    free(event_occurs_episodes);
    free(all_episodes);
    free(risk_scores);
    free(positive_prediction_episodes);
    free(negative_prediction_episodes);
}


int compute_metrics(PyObject *trajectories_obj, double snooze_window, double detection_window, PyObject *result_list, int verbosity) {
    int num_trajectories = PyList_Size(trajectories_obj);
    if (num_trajectories < 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid number of trajectories");
        return -1;
    }
    
    Trajectory *trajectories = (Trajectory *)malloc(num_trajectories * sizeof(Trajectory));
    if (trajectories == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for trajectories");
        return -1;
    }

    // Process all trajectories
    for (int i = 0; i < num_trajectories; i++) {
        PyObject *traj_obj = PyList_GetItem(trajectories_obj, i);
        if (traj_obj == NULL) {
            PyErr_SetString(PyExc_IndexError, "Invalid trajectory object index");
            free(trajectories);
            return -1;
        }
        if (convert_to_trajectory(traj_obj, &trajectories[i]) == -1) {
            free(trajectories);
            return -1;
        }

    }

    if (snooze_window == 0) {
        compute_metrics_no_snooze(trajectories, num_trajectories, detection_window, result_list);
    } else {
        double *risk_scores = NULL;
        int risk_scores_count = 0;

        for (int i = 0; i < num_trajectories; i++) {
            Trajectory *traj = &trajectories[i];
            double *new_risk_scores = (double *)realloc(risk_scores, (risk_scores_count + traj->len) * sizeof(double));
            if (new_risk_scores == NULL) {
                PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for risk scores");
                free(risk_scores);
                for (int j = 0; j < num_trajectories; j++) {
                    free_trajectory(&trajectories[j]);
                }
                free(trajectories);
                return -1;
            }
            risk_scores = new_risk_scores;
            for (int j = 0; j < traj->len; j++) {
                risk_scores[risk_scores_count++] = traj->predicted_risks[j];
            }
        }

        qsort(risk_scores, risk_scores_count, sizeof(double), compare_risk_scores);

        // Add timing variables
        clock_t start, current;
        double cpu_time_used;
        start = clock();

        int print_interval = risk_scores_count / 10;  // Print 10 updates
        double previous_threshold = -1.0;  // Initialize with an impossible threshold value

        for (int t = 0; t < risk_scores_count; t++) {
            double threshold = risk_scores[t];

            // Only process and append results if the threshold is different from the previous one
            if (threshold != previous_threshold) {
                int episode_tp = 0;
                int episode_fp = 0;
                int episode_fn = 0;
                int episode_tn = 0;
                int prediction_tp = 0;
                int prediction_fp = 0;

                for (int i = 0; i < num_trajectories; i++) {
                    Trajectory *traj = &trajectories[i];

                    double positive_predictions[traj->len];
                    int positive_len;
                    get_valid_times(traj->predicted_times, traj->predicted_risks, traj->len, threshold, snooze_window, positive_predictions, &positive_len);

                    if (traj->event_occurred) {
                        int num_tp, num_fp;
                        get_prediction_level_metrics(positive_predictions, positive_len, detection_window, traj->event_time, &num_tp, &num_fp);

                        if (num_tp == 0) {
                            episode_fn++;
                        } else {
                            episode_tp++;
                        }
                        prediction_tp += num_tp;
                        prediction_fp += num_fp;
                    } else {
                        if (positive_len > 0) {
                            prediction_fp += positive_len;
                            episode_fp++;
                        } else {
                            episode_tn++;
                        }
                    }
                }

                PyObject *result_dict = Py_BuildValue("{s:d, s:i, s:i, s:i, s:i, s:i, s:i}",
                                                      "threshold", threshold,
                                                      "episode_tp", episode_tp,
                                                      "episode_fp", episode_fp,
                                                      "episode_tn", episode_tn,
                                                      "episode_fn", episode_fn,
                                                      "prediction_tp", prediction_tp,
                                                      "prediction_fp", prediction_fp);
                PyList_Append(result_list, result_dict);
                Py_DECREF(result_dict);

                previous_threshold = threshold;  // Update the previous threshold
            }

            // Print progress and time estimation
            if (t > 0 && t % print_interval == 0) {
                current = clock();
                cpu_time_used = ((double) (current - start)) / CLOCKS_PER_SEC;
                double progress = (double)(t + 1) / risk_scores_count;
                double estimated_total_time = cpu_time_used / progress;
                double estimated_remaining = estimated_total_time - cpu_time_used;
                printf("Processed %d of %d risk scores (%.1f%%). Estimated time remaining: %.2f seconds\n", 
                       t + 1, risk_scores_count, progress * 100, estimated_remaining);
            }
        }

        // Compute final metrics with the lowest possible threshold
        int episode_tp = 0, episode_fp = 0, episode_fn = 0, episode_tn = 0;
        int prediction_tp = 0, prediction_fp = 0;

        for (int i = 0; i < num_trajectories; i++) {
            Trajectory *traj = &trajectories[i];

            double positive_predictions[traj->len];
            int positive_len;
            get_valid_times(traj->predicted_times, traj->predicted_risks, traj->len, -INFINITY, snooze_window, positive_predictions, &positive_len);

            if (traj->event_occurred) {
                int num_tp, num_fp;
                get_prediction_level_metrics(positive_predictions, positive_len, detection_window, traj->event_time, &num_tp, &num_fp);

                if (num_tp == 0) {
                    episode_fn++;
                } else {
                    episode_tp++;
                }
                prediction_tp += num_tp;
                prediction_fp += num_fp;
            } else {
                if (positive_len > 0) {
                    prediction_fp += positive_len;
                    episode_fp++;
                } else {
                    episode_tn++;
                }
            }
        }

        // Push final results with a large negative number as threshold
        PyObject *final_result_dict = Py_BuildValue("{s:d, s:i, s:i, s:i, s:i, s:i, s:i}",
                                    "threshold", -99999.0,
                                    "episode_tp", episode_tp,
                                    "episode_fp", episode_fp,
                                    "episode_tn", episode_tn,
                                    "episode_fn", episode_fn,
                                    "prediction_tp", prediction_tp,
                                    "prediction_fp", prediction_fp);
        PyList_Append(result_list, final_result_dict);
        Py_DECREF(final_result_dict);  // Decrease reference count

        free(risk_scores);
    }
    
    for (int i = 0; i < num_trajectories; i++) {
        free_trajectory(&trajectories[i]);
    }
    free(trajectories);
    return 0;
}

