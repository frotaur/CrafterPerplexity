import galois, numpy as np
import pickle as pk
from tqdm import tqdm
import sys,os
from pathlib import Path
sys.path.append('..')

from modules.tok_utils import tok_text_to_h5

def gen_data(matrice_name, num_vectors=None,pertub_rate=0.04,folder='matrices'):
    """
        Generate dataset from matrix.

        Args:
        matrice_name: matrice name in matrice/<matrice_name>
        num_vectors: number of vectors to generate
    """
    curpath = Path(__file__).parent.absolute().as_posix()

    GF = galois.GF(2)

    matrix = pk.load(open(os.path.join(folder,f'{matrice_name}.pkl'),'rb'))
    matrix = GF(matrix)
    print(f'Using : {type(matrix)}, of shape : {matrix.shape}')
    print(f'Pertubation rate : {pertub_rate}')

    if(num_vectors == None):
        num_vectors = 10*matrix.shape[0]
    outpute_file = os.path.join(curpath,f'datasets/{matrice_name}_v{num_vectors//1000}k.txt')
    os.makedirs(os.path.dirname(outpute_file),exist_ok=True)
    f = open(outpute_file,'w')

    for _ in tqdm(range(num_vectors)):
        vector = GF(np.random.randint(0,2,matrix.shape[0]))

        in_pert = GF((np.random.rand(*vector.shape)<pertub_rate).astype(int))
        out_pert = GF((np.random.rand(*vector.shape)<pertub_rate).astype(int))

        result = matrix@vector
        
        result_str = ''.join([str(i) for i in (result+out_pert).tolist()])
        input_str = ''.join([str(i) for i in (vector+in_pert).tolist()])
        f.write(input_str+'_______'+result_str+'\n')
    
    print('made dataset, converting to h5')
    tok_dict = {'$':0, '0':1, '1':2,'_':3}
    tok_text_to_h5(outpute_file,premade_dict=tok_dict,destination_folder=os.path.join(curpath,'..','h5data'),tokenizer_folder=os.path.join(curpath,'..','tokenizers'),tokenizer_name='GaloisMat',output_name=f'{matrice_name}_v{num_vectors//1000}k_pert{pertub_rate}',delete_txt=True)

if __name__=='__main__':

    matrice_name = 'Matid20_6_15' # Name of the matrix to use to generate the Linear Language dataset
    pertub_rate = 0.01 # Probability that a bit will be flipped
    num_vectors = int(5e6) # Number of examples in the linear language dataset

    gen_data(matrice_name,num_vectors=num_vectors,pertub_rate=pertub_rate)

    # matrice_names= [mat.split('.')[0] for mat in os.listdir('sparsemat') if 'Mat' in mat] # List of matrices used for the sparsity learnability experiments
    # for matrice_name in matrice_names:
    #     pertub_rate = 0.01
    #     gen_data(matrice_name,num_vectors=int(5e5),pertub_rate=pertub_rate,folder='sparsemat')