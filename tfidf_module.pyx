# -*- coding: utf-8 -*-
import cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
from libc.math cimport log
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
cpdef np.float64_t get_idf(
        np.ndarray[np.int64_t] counts,
        np.int64_t n_consumers):
    '''Calculate idf(t, D).

    Uses the number of consumers d in D who purchased t.
    '''
    cdef np.float64_t idf
    try:
        idf = <np.float64_t> n_consumers
        idf /= <np.float64_t> np.sum([counts > 0])
        idf = np.log(idf)
    except ZeroDivisionError:
        idf = 0.
    return idf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.float64_t get_tf1(
        np.int64_t count) nogil:
    '''Return tf(t, d) for an item t.

    Uses standard formula.
    '''
    cdef np.float64_t tf
    tf = <np.float64_t> count
    return tf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.float64_t get_tf2(
        np.int64_t count) nogil:
    '''Return tf(t, d) for an item t.

    Uses alternative version.
    '''
    cdef np.float64_t tf
    tf = <np.float64_t> count
    if tf > 0.:
        tf = 1. + log(tf)
    else:
        tf = 0.
    return tf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=2] get_tfidf(
        np.ndarray[np.int64_t] counts,
        np.int64_t n_consumers):
    '''Return both tf-idf versions for all items.'''
    cdef Py_ssize_t i, l = len(counts)
    cdef np.float64_t idf
    idf = get_idf(counts,
                  n_consumers)
    cdef np.ndarray[np.float64_t, ndim=2] res = np.empty(
            (l, 2), dtype=np.float64)
    for i in prange(l,
                    nogil=True,
                    schedule='static'):
        res[i, 0] = get_tf1(counts[i]) * idf
        res[i, 1] = get_tf2(counts[i]) * idf
    return res
