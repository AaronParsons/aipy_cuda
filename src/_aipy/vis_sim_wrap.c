#include <Python.h>
#include "vis_sim.h"
#include <cuda_runtime_api.h>
#include "numpy/arrayobject.h"

PyObject *wrap_vis_sim(PyObject *self, PyObject *args){
	PyArrayObject *baseline, *src_dir, *src_int, *src_index, *freqs, *mfreqs, *beam_arr;
	PyArrayObject *vis_array;
	npy_intp N_fq, N_src, d0, d1, N_beam_fq, l, m;
    npy_float lmin, lmax, mmin, mmax, beamfqmin, beamfqmax;

	if(!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!ffffff",   &PyArray_Type, &baseline, 
                                                         &PyArray_Type, &src_dir, 
                                                         &PyArray_Type, &src_int,
                                                         &PyArray_Type, &src_index,
                             						     &PyArray_Type, &freqs,
                                                         &PyArray_Type, &mfreqs,
                                                         &PyArray_Type, &beam_arr, 
                                                         &lmin, &lmax, &mmin, &mmax, 
                                                         &beamfqmin, &beamfqmax)){
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
        d1 = PyArray_DIM(src_dir, 1);
        }

    if (!((d0 == N_src) && (d1 == (npy_intp) 3))){
        PyErr_Format(PyExc_ValueError, "src_dir must have 0th dimension = the length of src_int, and 1st dimension = 3");
        return NULL;
        }

    //Check that beam_arr is 3 dimensional.  If it is, set l, m, and N_beam_fq
    if (PyArray_NDIM(beam_arr) != 3){
        PyErr_Format(PyExc_ValueError, "beam_arr must be 3 dimensional");
        return NULL;
        }
    else{
        l         = PyArray_DIM(src_dir, 0);
        m         = PyArray_DIM(src_dir, 0);
        N_beam_fq = PyArray_DIM(src_dir, 0);
        }
    //XXX Check the type of the arrays
	vis_array     = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(freqs),    PyArray_DIMS(freqs),    NPY_CFLOAT);

	vis_sim(
	(float *)PyArray_DATA(baseline), // access pointer to data buffer, cast as floats
	(float *)PyArray_DATA(src_dir),
    (float *)PyArray_DATA(src_int),
    (float *)PyArray_DATA(src_index),
    (float *)PyArray_DATA(freqs),
    (float *)PyArray_DATA(mfreqs),
	(float *)PyArray_DATA(vis_array), 
    (float *)PyArray_DATA(beam_arr),
    l, m, N_beam_fq, lmin, lmax, mmin, mmax, beamfqmin, beamfqmax,
    N_fq, N_src // pass pointer to sum buffer to hold result
	);

	return PyArray_Return(vis_array);
	}

static PyMethodDef VissimMethods[] = {
	{"vis_sim", (PyCFunction)wrap_vis_sim, METH_VARARGS,
		"vis_sim(baseline, src_dir, src_int, src_index, freqs, mfreqs)\nCalculates visibilities.  All inputs are numpy float32 arrays"
	},
	{NULL, NULL}
	};
	
PyMODINIT_FUNC init_aipy(void){
	(void) Py_InitModule("_aipy", VissimMethods);
	import_array();
};
