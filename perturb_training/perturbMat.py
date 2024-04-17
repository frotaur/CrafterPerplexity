import galois as gal
import numpy  as np
import pickle as pk
import os,random


def save_perturb_matrix(matrice_name,num_flips, save=True):
    """
        Saves a matrix perturbing matrice_name by num_flips.
        Assumes matrix is located in matrices/ relative to where the script was launched from.

        Args: 
        matrice_name: matrice name in matrice/<matrice_name>
        num_flips: number of flips to apply to the matrix

        Returns:
        path to the saved matrix
    """
    GF = gal.GF(2)
    print('===========================GENERATION==========================')
    matrix = pk.load(open(os.path.join('matrices',f'{matrice_name}.pkl'),'rb'))
    matrix = GF(matrix)

    finished=False
    max_tries =100

    while(not finished and max_tries>0):
        newmatrix = GF(matrix.copy())
        num_changes = 0
        while num_changes<num_flips:
            row = np.random.randint(0,matrix.shape[0])
            col = np.random.randint(0,matrix.shape[1])
            if(row!=col):
                newmatrix[row,col] = GF(1)+matrix[row,col]
                num_changes+=1
        try :
            subsize = int(np.array(newmatrix).astype(np.uint32).flatten().sum())
            invM = np.linalg.inv(newmatrix)
            assert (invM @ newmatrix == GF(np.eye(matrix.shape[0],dtype=int))).all()
            num_ones_in = int(np.array(invM).astype(np.uint32).flatten().sum())
            print(f'perturbed has : ', subsize)
            print(f'Inverse has {num_ones_in} nonzero')
            finished = True
        except np.linalg.LinAlgError:
            finished=False
            max_tries-=1
            print('Failed, retry again ',max_tries, ' times !')
            continue


    if(not finished):
        raise ValueError(f'Could not invert matrix, max_tries is {max_tries}')
    else:
        print('success invertible, saving matrix')
    name = f'{matrice_name}_flipped{num_flips}_{num_ones_in-subsize}plus_{random.randint(0,100)}'
    
    if(save):
        pk.dump(newmatrix,open(os.path.join(f'matrices',name+'.pkl'),'wb'))
        
    print('non-zero diff : ',np.array(newmatrix+matrix).astype(np.uint32).flatten().sum())

    print('=========================END GENERATION========================')
    return name


if __name__=='__main__':
    matrice_name = 'Matid_25'
    num_flips = 11
    
    save_perturb_matrix(matrice_name,num_flips,save=False)