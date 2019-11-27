import numpy as np
arr = np.arange(16).reshape((2,2,4))
# arr1 = np.arange(16).reshape(())
#donglon, dongnho, cot
arr2 = arr.transpose((1,0,2))
print(arr)
print(arr2)
