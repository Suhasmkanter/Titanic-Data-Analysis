import numpy as np 





array = np.array([[1,2],[4,5],[4,2]],dtype='int32')
# print(array)

# # print(array.shape)
# # print(array.dtype)
# # print(array.astype(str))
# # print(array.std())

# array2 = np.array([[1,2]])
# print(array2)
# print("Shape of array",array.shape,"Shape of Array2 ",array2.shape)
# array3 = array + array2 
# print(array3)
# print(array3[0])
# print(array3[:,:])

import numpy as np

def multiplicationofMatrix(arrayone, arraytwo):
    if arrayone.shape[1] != arraytwo.shape[0]:  
        print("Matrix multiplication not possible: Columns of first must match rows of second!")
        return
    
    # Result matrix of shape (rows of A, columns of B)
    arraythree = np.zeros((arrayone.shape[0], arraytwo.shape[1]))  
    
    # Perform matrix multiplication manually
    for i in range(len(arrayone)):  # Rows of first matrix
        for j in range(len(arraytwo[0])):  # Columns of second matrix
            for k in range(len(arraytwo)):  # Common dimension (Columns of first = Rows of second)
                arraythree[i][j] += arrayone[i][k] * arraytwo[k][j]  # Summation step

    print("Resultant Matrix:\n", arraythree)

# Example Matrices
arrayone = np.array([[2, 3, 4], 
                     [5, 6, 4]])  # Shape (2,3)

arraytwo = np.array([[2, 3], 
                     [3, 2], 
                     [4,5]])  # Shape (3,2)

multiplicationofMatrix(arrayone, arraytwo)


print("Vectorized Operations:")


vector = np.array([1,2,3,4])
vector2 = np.array([1,2,3,4])
vector3 = vector * vector2

print(vector3)



print("Normalize the Datasets:")

arrays = np.arange(10,dtype=int)
print(arrays)
arrayone = (arrays[0:] - min(arrays))/(max(arrays)-min(arrays))
print(arrayone)


