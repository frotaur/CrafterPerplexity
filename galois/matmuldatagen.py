"""
    Script to generate sparse F_2 matrices, to be use in gendataset to make LInear Langauges datasets.
    Matrices are saved in galois/matrices
"""

import galois as gal
import numpy  as np
from tqdm import tqdm
import pickle,os

GF = gal.GF(2)

def sparse_id_matrix(size, k):
    """Generates a sparse identity matrix of size `size` with approx `k` ones, and diagonal entries are 1"""
    rando_no_diag = generate_random_matrix(size, k)
    np.fill_diagonal(rando_no_diag,0)
    mat = GF(np.identity(size,dtype=int)+rando_no_diag)

    return mat

def sparse_id_matrix_exact(size, k):
    init = np.array(sparse_id_matrix(size, k)).astype(np.uint32).flatten()
    summed = init.sum()
    while(summed!=k+size):
        nonzeros = np.argwhere(init!=0).flatten()
        zeros = np.argwhere(init==0).flatten()
        if(summed>=k+size):
            init[np.random.choice(nonzeros)] = 0
        elif(summed<k+size):
            init[np.random.choice(zeros)] = 1
        summed=init.sum()
    return GF(init.reshape(size,size))

def rand_sparse_matrix(size,k):
    """Generates a sparse matrix of size `size` with `k` ones. In GF(2)"""
    return GF(generate_random_matrix(size, k))

def generate_random_matrix(size, k):
    """Generates a random matrix of size `size` with `k` ones"""
    # Initialize a matrix of zeros
    matrix = np.zeros((size,size), dtype=int)
    
    flat_matrix = matrix.flatten()
    
    ones_indices = np.random.choice(flat_matrix.size, k, replace=False)
    flat_matrix[ones_indices] = 1
    # Reshape the flat matrix back to its original 2D shape
    return flat_matrix.reshape(size,size)

def generate_random_matrix_more_inv(size, k):
    lines_first = size
    new_k = k - lines_first

    matrix = np.zeros((size,size), dtype=int)

    col_indices = np.random.choice(size, lines_first, replace=False)

    matrix[np.arange(lines_first),col_indices] = 1 # add at least one 1 on each line

    flat_matrix = matrix.flatten()
    extra_ones_indices = np.random.choice(flat_matrix.size, new_k, replace=False)
    flat_matrix[extra_ones_indices] = 1

    return GF(flat_matrix.reshape(size,size))

def search_and_save(size,k,id=False,numtry=300,folder='matrices'):
    numinv = 0
    os.makedirs(folder,exist_ok=True)
    if(id):
        randfunc = sparse_id_matrix_exact
        idname = 'id'
    else:
        randfunc = generate_random_matrix_more_inv
        idname = 'rand'

    for _ in tqdm(range(numtry)):
        M = randfunc(size,k)
        try :
            invM = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
        numinv += 1./numtry

        subsize = int(np.array(M).astype(np.uint32).flatten().sum())
        num_ones_out = int(np.array(invM).astype(np.uint32).flatten().sum())

        print(f'\n In {subsize}, out {num_ones_out} : ')
        # pickle.dump(M,open(f'matrices/Mat{idname}{size}_{subsize}_{num_ones_out}.pkl','wb'))
        pickle.dump(M,open(os.path.join(folder,f'Mat{idname}{size}_{subsize-size}_{num_ones_out-size}.pkl'),'wb'))

        return M
    print(f'num invertible : {numinv*100}%')

if __name__=='__main__':
    #### PARAMETERS ####
    mat_size = 20 # Size of the square matrix to generate
    num_nonzeros = 25 # Number of non-zero entries in the matrix. In this case, it will be the Identity + 5 random ones.
    #### END PARAMETERS ####


    search_and_save(mat_size,num_nonzeros,id=True,numtry=200)