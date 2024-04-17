import galois, numpy as np
import pickle as pk
from tqdm import tqdm
import sys,os
from pathlib import Path
import random, shutil

sys.path.append('../')


from modules import tokenize,make_h5, SimpleTokenizer
import torch


def tok_text_to_h5_fix_tokenizer(text,tokenizername, output_name='last_tokenization'):
    curpath = Path(__file__).parent.absolute().as_posix()
    premade_tok_path = os.path.join(curpath,'tokenizers',tokenizername+'.pt')
    premade_tok_dict = torch.load(premade_tok_path)
    _ = tokenize(text,output_name=output_name,premade_dict=premade_tok_dict,fixed=True)

    # torch.save(token_dict,os.path.join(curpath,'tokenizers',tokenizername+'.pt')) # Change tokenizer name here
    tokenizer = SimpleTokenizer(premade_tok_path)
    make_h5(os.path.join('tokendata',output_name),tokenizer=tokenizer,data_name=output_name,destination_folder=os.path.join(curpath,'datasets'))
    try:
        os.remove(text)
    except Exception as e:
        print(e) # To prevent annoying error on windows
    shutil.rmtree(os.path.join('tokendata',output_name))

def gen_data(matrice_name, num_vectors=None,pertub_rate=0.):
    """
        Generate dataset from matrix. Saves it in h5data, and returns the path to the saved file.

        Args:
        matrice_name: matrice name in matrice/<matrice_name>
        num_vectors: number of vectors to generate
        pertub_rate: perturbation rate %, how much to randomly flip bits

        Returns:
        name of saved dataset in h5data.
    """
    GF = galois.GF(2)
    curpath = Path(__file__).parent.absolute().as_posix()
    print('===========================DATAGENTOKEN==========================')
    print('=============MATRIX : ',matrice_name,'=====================')
    mat_path = f'matrices/{matrice_name}.pkl'
    matrix = pk.load(open(mat_path,'rb'))
    matrix = GF(matrix)
    # print(f'Using : {type(matrix)}, of shape : {matrix.shape}')
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

        input_str = ''.join([str(i) for i in (vector+in_pert).tolist()])
        result_str = ''.join([str(i) for i in (result+out_pert).tolist()])
        f.write(input_str+'_______'+result_str+'\n')
    
    print('made dataset, converting to h5 and deleting matrix')
    os.remove(mat_path)

    out_name = f'forpert{random.randint(0,1000)}'

    tok_text_to_h5_fix_tokenizer(outpute_file,'GaloisMat',output_name=out_name)

    return out_name

if __name__=='__main__':
    curpath = Path(__file__).parent.absolute().as_posix()

    sys.path.append('..')

    from tokenize_to_h5 import tok_text_to_h5

    GF = galois.GF(2)

    matrice_name = 'Matid_25_flipped10_11plus'
    pertub_rate = 0.
    gen_data(matrice_name,num_vectors=int(1e5))
    