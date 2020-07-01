"""

argmax(a, axis=None, out=None):

    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
"""


import numpy as np
a = np.array( [[1,5,3,30],
               [10,1,2,5],
               [10,30,20,6]],
              dtype='int64')

# along axis=0 , along row
argmax_axis0 = a.argmax(axis=0) # array([1, 2, 2, 0])
argmax_axis1 = a.argmax(axis=1) # array([3, 0, 1])

b = np.array( [1,5,3,30] )
argmax_axis0 = b.argmax() # 3


# sorts along first axis (down)
ind = a.argsort(axis=0)
                     #array([[0, 1, 1, 1],
                           # [1, 0, 0, 2],
                           # [2, 2, 2, 0]])

# sorts along last axis (across)
ind = a.argsort(axis=1)
                # array([[0, 2, 1, 3],
                        # [1, 2, 3, 0],
                        # [3, 0, 2, 1]])
