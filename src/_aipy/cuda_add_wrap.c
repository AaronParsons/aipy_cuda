#include <Python.h>
#include "numpy/arrayobject.h"

int cuda_add(int*, int*, int*, int);

PyObject *wrap_cuda_add(PyObject *self, PyObject *args){
	PyArrayObject *a, *b, *c;
	PyArrayObject *a_cast, *b_cast;
	npy_intp N;
	if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b)){
	return NULL;
	}
	N = PyArray_Size(a);
	if (PyArray_Size(b) != N){ //if the 2 arrays are not the same length, raise an error
		PyErr_Format(PyExc_ValueError, "a.size != b.size");
		return NULL;
	}
	c = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(a), PyArray_DIMS(a), NPY_INT);
	a_cast = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(a), PyArray_DIMS(a), NPY_INT);
	b_cast = (PyArrayObject *) PyArray_SimpleNew(PyArray_NDIM(a), PyArray_DIMS(a), NPY_INT);
	if (c == NULL || a_cast == NULL || b_cast == NULL) { // check allocation success
		PyErr_Format(PyExc_MemoryError, "failed to allocate buffer");
		return NULL;
	}
	// cast input arrays to ensure integer type, raise exception if either cast fails
	if (PyArray_CastTo(a_cast, a) || PyArray_CastTo(b_cast,b)) {
		PyErr_Format(PyExc_ValueError, "failed to cast inputs to integers");
		return NULL;
	}
	// call third party function on the data inside numpy arrays
	cuda_add(
	(int *)PyArray_DATA(a_cast), // access pointer to data buffer, cast as integers
	(int *)PyArray_DATA(b_cast),
	(int *)PyArray_DATA(c), N // pass pointer to sum buffer to hold result
	);
	Py_DECREF(a_cast); // we created new arrays that are no longer needed
	Py_DECREF(b_cast); // must decref them to prevent memory leak
	return PyArray_Return(c); // PyArray_Return isn’t required, but good policy
	}

static PyMethodDef NumpyextMethods[] = {
	{"cuda_add",
		(PyCFunction)wrap_cuda_add,
		METH_VARARGS,
		"cuda_add(a,b), return a+b for numpy arrays"
	},
	{NULL, NULL}
	};
	
PyMODINIT_FUNC init_numpyext(void){
	(void) Py_InitModule("_numpyext", NumpyextMethods);
	import_array();
};
