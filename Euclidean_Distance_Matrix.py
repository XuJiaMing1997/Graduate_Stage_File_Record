import numpy as np
import math


Matrix_1 = [[1,2,3,4,5],
            [2,3,4,5,6],
            [3,4,5,6,7],
            [0,1,0,1,0]]

Matrix_2 = [[0,1,0,1,0],
            [1,0,1,0,1],
            [0,0,1,1,0]]


def Original_L2(M1,M2):
    # GT
    Norm_1 = M1/(np.linalg.norm(M1,ord=2,axis=1)[:,np.newaxis])
    Norm_2 = M2/(np.linalg.norm(M2,ord=2,axis=1)[:,np.newaxis])
    Distance = []
    for i in Norm_1:
        for j in Norm_2:
            Distance.append(np.linalg.norm(i-j,ord=2))
    return Distance


# So use this  !!!!!!!
def MatrixVersion_L2(M1,M2):
    Norm_1 = M1/(np.linalg.norm(M1,ord=2,axis=1)[:,np.newaxis])
    Norm_2 = M2/(np.linalg.norm(M2,ord=2,axis=1)[:,np.newaxis])

    # num_test = Norm_1.shape[0]
    # num_train = Norm_2.shape[0]
    # dists = np.zeros((num_test, num_train))
    dists = np.sqrt(-2*np.matmul(Norm_1, np.transpose(Norm_2)) + np.sum(np.square(Norm_2), axis = 1) +
                    np.transpose([np.sum(np.square(Norm_1), axis = 1)]))
    return dists

def MatrixVersion2_L2(M1,M2):
    Norm_1 = M1/(np.linalg.norm(M1,ord=2,axis=1)[:,np.newaxis])
    Norm_2 = M2/(np.linalg.norm(M2,ord=2,axis=1)[:,np.newaxis])
    all_matrix = np.array(np.concatenate((Norm_1,Norm_2),axis=0))
    # all_matrix = np.array(np.concatenate((M1,M2),axis=0))
    dot_product = np.matmul(all_matrix,all_matrix.T) # 2D
    square_norm = np.diagonal(dot_product) # 1D
    # distance: ||X-Y||^2
    distance = square_norm[np.newaxis,:] - 2.0 * dot_product + square_norm[:,np.newaxis]
    distance = np.sqrt(distance)
    # distance = np.maximum(distance,0)
    return distance

def MatrixVersion3_L2(M1,M2):
    Norm_1 = M1/(np.linalg.norm(M1,ord=2,axis=1)[:,np.newaxis])
    Norm_2 = M2/(np.linalg.norm(M2,ord=2,axis=1)[:,np.newaxis])

    # SquareNorm_M1 = np.linalg.norm(Norm_1,ord=2,axis=1) ** 2
    # SquareNorm_M2 = np.linalg.norm(Norm_2,ord=2,axis=1) ** 2
    # ??????  use above will make it get a little difference from respect value

    # SquareNorm_M1 = np.sum(np.square(Norm_1), axis = 1) ** (1./2)
    # SquareNorm_M2 = np.sum(np.square(Norm_2), axis = 1) ** (1./2)
    # SquareNorm_M1 = SquareNorm_M1 ** 2
    # SquareNorm_M2 = SquareNorm_M2 ** 2
    # this type op will get same value as above ---- a little difference from respect value

    # SquareNorm_M1 = np.sqrt(np.sum(np.square(Norm_1), axis = 1))
    # SquareNorm_M2 = np.sqrt(np.sum(np.square(Norm_2), axis = 1))
    # SquareNorm_M1 = SquareNorm_M1 ** 2
    # SquareNorm_M2 = SquareNorm_M2 ** 2
    # this type op get same value as above  ----  SO WE KNOW
    # sqrt or **(1./x) then square will get approximate value than not sqrt

    SquareNorm_M1 = np.sum(np.square(Norm_1), axis = 1)
    SquareNorm_M2 = np.sum(np.square(Norm_2), axis = 1)


    dot_product = np.matmul(Norm_1,Norm_2.T)

    res = np.sqrt(SquareNorm_M1[:,np.newaxis] - 2 * dot_product + SquareNorm_M2[np.newaxis,:])
    return res




def sqrt_check(M1,M2):
    # Norm_1 = M1/(np.linalg.norm(M1,ord=2,axis=1)[:,np.newaxis])
    # Norm_2 = M2/(np.linalg.norm(M2,ord=2,axis=1)[:,np.newaxis])

    Norm_1 = M1
    Norm_2 = M2

    SquareNorm_M1_a = np.linalg.norm(Norm_1,ord=2,axis=1) ** 2
    SquareNorm_M2_a = np.linalg.norm(Norm_2,ord=2,axis=1) ** 2
    print(SquareNorm_M1_a)
    print(SquareNorm_M2_a)

    SquareNorm_M1 = np.sqrt(np.sum(np.square(Norm_1), axis = 1))
    SquareNorm_M2 = np.sqrt(np.sum(np.square(Norm_2), axis = 1))
    SquareNorm_M1 = SquareNorm_M1 ** 2
    SquareNorm_M2 = SquareNorm_M2 ** 2
    print(SquareNorm_M1)
    print(SquareNorm_M2)
    print(np.equal(SquareNorm_M1_a,SquareNorm_M1))
    print(np.equal(SquareNorm_M2_a,SquareNorm_M2))



# Here occur difference !!!!!  Problem   delta ~= 1e-16
res1 = Original_L2(Matrix_1,Matrix_2) # only norm
res2 = MatrixVersion_L2(Matrix_1,Matrix_2).reshape(-1).tolist() # norm + sqrt
print(res1)
print(res2)
print(np.equal(res1,res2))
# ##################################################################
# Here occur difference !!!!!  Problem   delta ~= 1e-16
print('Original_L2 & MatrixVersion_L2 & MatrixVersion2_L2')
res1 = Original_L2(Matrix_1,Matrix_2) # norm
res2 = MatrixVersion_L2(Matrix_1,Matrix_2).reshape(-1).tolist() # norm + sqrt
res3 = MatrixVersion2_L2(Matrix_1,Matrix_2) # norm + sqrt
res3_1 = res3[:len(Matrix_1),-len(Matrix_2):].reshape(-1).tolist()
print(res1)
print(res2)
print(res3_1)
print('Original_L2 & MatrixVersion_L2\n{}'.format(np.equal(res1,res3_1))) # different
print('MatrixVersion_L2 & MatrixVersion2_L2\n{}'.format(np.equal(res2,res3_1))) # Same
# ##################################################################
print('Original_L2 & MatrixVersion_L2 & MatrixVersion2_L2 & MatrixVersion3_L2')
res1 = Original_L2(Matrix_1,Matrix_2) # norm
res2 = MatrixVersion_L2(Matrix_1,Matrix_2).reshape(-1).tolist() # norm + sqrt
res3 = MatrixVersion2_L2(Matrix_1,Matrix_2) # norm + sqrt
res3_1 = res3[:len(Matrix_1),-len(Matrix_2):].reshape(-1).tolist()
res4 = MatrixVersion3_L2(Matrix_1,Matrix_2).reshape(-1).tolist()
print(res1)
print(res2)
print(res3_1)
print(res4)
print('Original_L2 & MatrixVersion3_L2\n{}'.format(np.equal(res1,res4)))
print('Conclusion -> res2 == res3_1 == res4 != res1: {}'.format(res2 == res3_1 == res4 != res1))