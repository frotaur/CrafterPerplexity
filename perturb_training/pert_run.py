from train_perturb import train,prepare_training
import argparse, json, os
from pathlib import Path
import shutil, random, time

def train_both(matrix,blueprint,num_flips,model_f,model_b=None,device='cpu'):
    """
        Trains both forward and backward models. 

        Args :
        matrix : name of the matrix to train on
        blueprint : path to the blueprint file
        num_flips : number of flips to do
        model_f : name of the forward model
        model_b : name of the backward model. If None, does NOT train backward model.
        device : device to train on
    """
    train_b=True
    if(model_b is None):
        model_b = model_f
        train_b=False
    blueprint = json.load(open(blueprint,'r'))
    curpath = Path(__file__).parent.absolute().as_posix()

    perturb_rate = blueprint['training_params']['perturb_rate']
    tokenizer_name = 'GaloisMat'

    (blueprint_f,blueprint_b), (f_load,b_load), dataset_name = prepare_training(blueprint,model_f,model_b,dataset_name=None,matrix=matrix,num_flips=num_flips,pertub_rate=perturb_rate)
    
    random.seed(time.time())
    num_run = random.randint(0,9999)
    run_name_f = matrix+f'_pert{perturb_rate:.2f}'+f'_fl{num_flips}_F{num_run:04d}'
    run_name_b = matrix+f'_pert{perturb_rate:.2f}'+f'_fl{num_flips}_B{num_run:04d}'

    print(f'Will train : {run_name_f} and {run_name_b}')
    ### Train model forward
    train(blueprint_f,f_load,device,run_name=run_name_f,tokenizer_name=tokenizer_name,pickup=False,group=num_flips)
    print('Forward finished, training backward')
    if(train_b):
        train(blueprint_b,b_load,device,run_name=run_name_b,tokenizer_name=tokenizer_name,pickup=False,group=num_flips)
        print('Backward finished, training finished')

    print('deleting dataset')
    os.remove(os.path.join(curpath,'datasets',dataset_name+'.h5'))

    print('deleting saved runs')
    for file in os.listdir(os.path.join(curpath,'runs','PerturbatedPerplexity','state')):
        os.remove(os.path.join(curpath,'runs','PerturbatedPerplexity','state',file))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Starts training of Predictor model given a JSON config file. \
                                     Use 'gen_run.py' to create a JSON config file.")
    parser.add_argument("device", type=str, default='cpu', help="Device string, e.g. 'cuda:0' or 'cpu'")
    args = parser.parse_args()
    curpath = Path(__file__).parent.absolute().as_posix()   

    ## PARAMETERS TO CHANGE ##
    matrix = 'Matid20_6_15' ## Which matrix to use to generate sparse updated linear language dataset
    model_f = 'Matid20_6_0.01' ## Which trained model to use as base for forward model
    model_b = 'Matid20_6_0.01_b' ## Which trained model to use as base for backward model

    num_flips = 0 ## Number of elements to flip in the sparse perturbation of the matrix
    
    ###### DO NOT CHANGE BELOW #####

    blueprint = 'blueprints/gpt2-gpt1.json'
    device = args.device
    
    # To train B and F many times
    for i in range(20):
        print('Run number ',i)
        train_both(matrix,blueprint,num_flips,model_f,model_b,device=device)

    shutil.rmtree(os.path.join(curpath,'runs'))