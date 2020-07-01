import numpy as np


# uniformly distributed variable [0 to 1]
np.random.random((2,2))
# array([[0.70827407, 0.15210029],
#        [0.31615172, 0.6900894 ]])

np.random.randn(2,2)
# array([[-0.066536  , -0.3976586 ],
#        [-1.18245148,  1.49786349]])


np.random.randn(2)
# array([1.25847793, 0.90597546])

np.random.normal((2,6))
# array([1.66985838, 5.11182577])

np.random.rand(3)
# array([0.49289419, 0.19282439, 0.40432695])

np.random.choice([1,2,3,4,5,6])
# 5

N = 10
np.linspace(0,10,N).reshape(N,1) #NxD matrix

Ntrain = 20
np.random.choice(N,Ntrain)
