#include <Python.h>

// Forward declaration of the process_trajectories function
int compute_metrics(PyObject *trajectories_obj, double snooze_window, double detection_window, PyObject *result_list, int verbosity);

// Helper function to convert Python list or tuple to C array
int convert_to_c_array(PyObject *input, double **output, int *len) {
    // If input is a single Python float, convert it directly
    if (PyFloat_Check(input)) {
        *len = 1;
        *output = (double *)malloc(*len * sizeof(double));
        if (*output == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for array");
            return -1;
        }
        (*output)[0] = PyFloat_AsDouble(input);
    } 
    // If input is a list or tuple
    else if (PyList_Check(input) || PyTuple_Check(input)) {
        *len = PySequence_Size(input);
        *output = (double *)calloc(*len, sizeof(double));  // Use calloc instead of malloc
        if (*output == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for array");
            return -1;
        }
        for (int i = 0; i < *len; i++) {
            PyObject *item = PySequence_GetItem(input, i);
            if (!item) {
                free(*output);
                *output = NULL;
                return -1;
            }
            if (PyFloat_Check(item) || PyLong_Check(item)) {
                (*output)[i] = PyFloat_AsDouble(item);
            } else if (PyDict_Check(item)) {
                PyObject *predicted_risks = PyDict_GetItemString(item, "predicted_risks");
                if (!PyList_Check(predicted_risks)) {
                    PyErr_SetString(PyExc_TypeError, "predicted_risks must be a list");
                    free(*output);
                    *output = NULL;  // Set pointer to NULL after freeing
                    return -1;
                }
                Py_ssize_t predicted_risks_len = PyList_Size(predicted_risks);
                if (predicted_risks_len > 1) {
                    PyErr_SetString(PyExc_ValueError, "predicted_risks list has more than one element");
                    free(*output);
                    *output = NULL;  // Set pointer to NULL after freeing
                    return -1;
                }
                PyObject *risk_item = PyList_GetItem(predicted_risks, 0);
                if (!PyFloat_Check(risk_item)) {
                    PyErr_SetString(PyExc_TypeError, "All predicted_risks elements must be floats");
                    free(*output);
                    *output = NULL;  // Set pointer to NULL after freeing
                    return -1;
                }
                (*output)[i] = PyFloat_AsDouble(risk_item);
            } else {
                PyErr_SetString(PyExc_TypeError, "All elements must be floats or dictionaries");
                free(*output);
                *output = NULL;  // Set pointer to NULL after freeing
                return -1;
            }
            Py_DECREF(item);
        }
    } 
    // If input is not a Python number, list, or tuple
    else {
        PyErr_SetString(PyExc_TypeError, "Input must be a float, list, or tuple");
        return -1;
    }
    return 0;
}

// Rename the Python-callable C function
static PyObject* py_compute_metrics_c(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *trajectories;
    PyObject *snooze_window_obj;
    PyObject *detection_window_obj;
    int verbosity = 1;  // Default verbosity level
    double snooze_window, detection_window;

    static char *kwlist[] = {"trajectories", "snooze_window", "detection_window", "verbosity", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|i", kwlist, &trajectories, &snooze_window_obj, &detection_window_obj, &verbosity)) {
        return NULL;
    }

    // Handle both float and int for snooze_window
    if (PyFloat_Check(snooze_window_obj)) {
        snooze_window = PyFloat_AsDouble(snooze_window_obj);
    } else if (PyLong_Check(snooze_window_obj)) {
        snooze_window = (double)PyLong_AsLong(snooze_window_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "snooze_window must be a float or int");
        return NULL;
    }

    // Handle both float and int for detection_window
    if (PyFloat_Check(detection_window_obj)) {
        detection_window = PyFloat_AsDouble(detection_window_obj);
    } else if (PyLong_Check(detection_window_obj)) {
        detection_window = (double)PyLong_AsLong(detection_window_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "detection_window must be a float or int");
        return NULL;
    }

    if (PyErr_Occurred()) {
        return NULL;
    }

    PyObject *result_list = PyList_New(0);
    if (result_list == NULL) {
        return PyErr_NoMemory();
    }

    if (compute_metrics(trajectories, snooze_window, detection_window, result_list, verbosity) == -1) {
        Py_DECREF(result_list);
        return NULL;
    }

    return result_list;
}

// Method definitions
static PyMethodDef CtsevalMethods[] = {
    {"compute_metrics_c", (PyCFunction)py_compute_metrics_c, METH_VARARGS | METH_KEYWORDS, "Process trajectories and calculate metrics"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef ctseval_module = {
    PyModuleDef_HEAD_INIT,
    "ctseval",
    "Module for computing clinical time series metrics",
    -1,
    CtsevalMethods
};

// Module initialization
PyMODINIT_FUNC PyInit__ctseval(void) {
    return PyModule_Create(&ctseval_module);
}
