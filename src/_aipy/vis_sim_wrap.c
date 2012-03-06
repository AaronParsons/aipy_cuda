#include <Python.h>
#include "vis_sim.h"
#include <cuda_runtime_api.h>
#include "numpy/arrayobject.h"

PyObject *wrap_vis_sim(PyObject *self, PyObject *args){
	PyArrayObject *baseline, *src_dir, *src_int, *src_index, *freqs, *mfreqs;
	PyArrayObject *vis_array;
    PyArrayObject *baseline_cast, *src_dir_cast, *src_int_cast, *src_index_cast, *freqs_cast, *mfreqs_cast;
	npy_intp N_fq, N_src, d0, d1;

	if(!PyArg_ParseTuple(args, "O!O!O!O!O!O!", &PyArray_Type, &baseline, &PyArray_Type, &src_dir, &PyArray_Type, &src_int, &PyArray_Type, &src_index, 						   &PyArray_Type, &freqs, &PyArray_Type, &mfreqs)){
	return NULL;
	}

	N_fq = PyArray_Size((PyObject *)freqs);
    N_src = PyArray_Size((PyObject *)src_int);

    //if the baseline is not 3 numbers, raise an error
    if (PyArray_Size((PyObject *)baseline) != 3){
        PyErr_Format(PyExc_ValueError, "Baseline vector length must be 3");
        return NULL;
    }
    
    //Check that there is one src_index for each source
    if (PyArray_Size((PyObject *)src_index) != N_src){
        PyErr_Format(PyExc_ValueError, "src_index.size != src_int.size");
        return NULL;
    }

    //Check that the length of mfreqs = the number of sources
    if (PyArray_Size((PyObject *)mfreqs) != N_src){
        PyErr_Format(PyExc_ValueError, "mfreqs.size != src_int.size");
        return NULL;
    }

    //Check that src_dir is 2 dimensional, and one dimension is 3 long and the other is equal to N_src
    if (PyArray_NDIM(src_dir) != 2){
        PyErr_Format(PyExc_ValueError, "src_dir must be 2 dimensional");
        return NULL;
        }
    else{
        d0 = PyArray_DIM(src_dir, 0);
        d1= PyArray_DIM(src_dir, 1);
        }

    if (!((d0 == N_src) && (d1 == (npy_intp) 3))){
        PyErr_Format(PyExc_ValueError, "src_dir must have 0th dimension = the length of src_int, and 1st dimension = 3");
        return NULL;
        }

    //XXX Instead of casting, check the type of the arrays
	vis_array     = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(freqs),    PyArray_DIMS(freqs),    NPY_CFLOAT);
	baseline_cast = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(baseline), PyArray_DIMS(baseline), NPY_FLOAT);
    src_dir_cast  = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(src_dir),  PyArray_DIMS(src_dir),  NPY_FLOAT);
    src_int_cast  = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(src_int),  PyArray_DIMS(src_int),  NPY_FLOAT);
    src_index_cast= (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(src_index),PyArray_DIMS(src_int),  NPY_FLOAT);
	freqs_cast    = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(freqs),    PyArray_DIMS(freqs),    NPY_FLOAT);
	mfreqs_cast   = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(mfreqs),    PyArray_DIMS(mfreqs),    NPY_FLOAT);
    
	// check allocation success
	if (vis_array == NULL || baseline_cast == NULL || src_dir_cast == NULL || src_int_cast == NULL || src_index_cast == NULL 
        || freqs_cast == NULL|| mfreqs_cast == NULL) { 
		PyErr_Format(PyExc_MemoryError, "failed to allocate buffer");
		return NULL;
	}
	// cast input arrays to ensure float type, raise exception if either cast fails
	if (PyArray_CastTo(baseline_cast, baseline) || PyArray_CastTo(src_dir_cast, src_dir) || PyArray_CastTo(src_int_cast, src_int) ||PyArray_CastTo(src_index_cast, src_index) || PyArray_CastTo(freqs_cast, freqs) || PyArray_CastTo(mfreqs_cast, mfreqs)){
		PyErr_Format(PyExc_ValueError, "failed to cast inputs to floats");
		return NULL;
	}
	// call third party function on the data inside numpy arrays
	vis_sim(
	(float *)PyArray_DATA(baseline), // access pointer to data buffer, cast as floats
	(float *)PyArray_DATA(src_dir),
    (float *)PyArray_DATA(src_int),
    (float *)PyArray_DATA(src_index),
    (float *)PyArray_DATA(freqs),
    (float *)PyArray_DATA(mfreqs),
	(float *)PyArray_DATA(vis_array), 
    N_fq, N_src // pass pointer to sum buffer to hold result
	);
	Py_DECREF(baseline_cast); // we created new arrays that are no longer needed
	Py_DECREF(src_dir_cast); // must decref them to prevent memory leak
    Py_DECREF(src_int_cast);
    Py_DECREF(src_index_cast);
    Py_DECREF(freqs_cast);
    Py_DECREF(mfreqs_cast);
	return PyArray_Return(vis_array);
	}

static PyMethodDef VissimMethods[] = {
	{"vis_sim", (PyCFunction)wrap_vis_sim, METH_VARARGS,
		"vis_sim(baseline, src_dir, src_int, src_index, freqs, mfreqs) Calculates visibilities."
	},
	{NULL, NULL}
	};
	
PyMODINIT_FUNC init_aipy(void){
	(void) Py_InitModule("_aipy", VissimMethods);
	import_array();
};
