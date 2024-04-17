"""
    Training script for GPT-like models. Use this along with the output of gen_run !
"""
import torch, torch.optim,os, argparse,json, pathlib,random
from torch.utils.data import Subset
from torch.optim.lr_scheduler import LinearLR
import sys
import copy
sys.path.append('../')
from modules import MinGPT, MinGPT_Trainer, SimpleTokenizer, TokenTextBOS
from perturbMat import save_perturb_matrix
from gendataset import gen_data

project_name = 'PerturbatedPerplexity'
cur_path = pathlib.Path(__file__).parent.absolute().as_posix()

def train(configuration,load_from,device,run_name,tokenizer_name=None,pickup=True,group=None):

    if(tokenizer_name is not None):
        tokenizer_path = os.path.join(cur_path,'tokenizers',f'{tokenizer_name}.pt')
        tokenizer = SimpleTokenizer(tokenizer_path)
    else :
        raise ValueError('No tokenizer name given. Please specify a tokenizer name with the -t flag.')

    configo = configuration
    model_params = configo['model_params']
    training_params = configo['training_params']
    optim_params = configo['optim_params']

    if not os.path.exists(training_params['dataset_folder']):
        raise FileNotFoundError(f"Tried to find dataset folder at \
                                {training_params['dataset_folder']}, but failed. \
                                Make sure there is the folder {training_params['dataset_folder']}\
                                in the same directory.")


    valid_steps =training_params['valid_steps'] # We ask how many validation steps. To get these, we assume 5% of training time allocated for validation.

    valid_every = training_params['valid_every']

    backwards = training_params['backwards']

    # random.seed(42) # For deterministic shuffling of dataset
    dataset_path = training_params['dataset_folder']
    motherDataset = TokenTextBOS(h5_file=dataset_path, backwards=backwards)
    
    assert model_params['vocab_size']==0, 'Set vocab size to 0, determined by tokenizer.'
    assert model_params['attn_length']==0, 'Set attn_length to 0, determined by dataset.'

    model_params['vocab_size'] = tokenizer.vocab_size
    model_params['attn_length'] = motherDataset.attn_length

    # indices = list(range(len(motherDataset)))
    # random.shuffle(indices)
    # motherDataset = Subset(motherDataset, indices) # Shuffled dataset

    # To keep it constant even if switching batch_size, I take batch_size=250
    val_inds = valid_steps*training_params['batch_size']


    val_range = range(len(motherDataset)-val_inds,len(motherDataset)) # Validation, last portion of dataset
    keep_range = range(len(motherDataset)-val_inds) # Training, first portion of dataset
    
    #Whether backwards or forwards, its the individual examples that are flipped, not the dataset. So same thing for both !
    train_dataset = Subset(motherDataset, keep_range)
    val_dataset = Subset(motherDataset, val_range)

    print('DATA SNIPPETS : ')
    for i in range(1,3):
        idx = i
        print(f'{idx} TRAIN :',tokenizer.detokenize(train_dataset[idx][0]))
        print(f'ANSWER : ',tokenizer.detokenize(train_dataset[idx][1]))
        print(f'{idx} VALID :',tokenizer.detokenize(val_dataset[idx][0]))
        print(f'ANSWER : ',tokenizer.detokenize(val_dataset[idx][1]))

    model = MinGPT(**model_params)

    #====================== TRAINING PARAMETERS =======================
    batch_size = training_params['batch_size']
    aggregate = training_params['aggregate']
    totbatches = len(train_dataset)//batch_size
    
    if(training_params['steps_to_train']==None):
        steps_to_train = totbatches # 4 epochs
    else:
        steps_to_train = training_params['steps_to_train']

    print(f'{totbatches=}, {batch_size=}, {len(train_dataset)=}')
    base_lr = optim_params['lr']
    lr_init = optim_params['lr_init']

    print(f'--- Training for ~ {steps_to_train//1000}k minibatches ---')
    #------ Optimizers ------
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr)
    # optim = torch.optim.Adam(model.parameters(), lr=base_lr)
    # optim = torch.optim.SGD(model.parameters(), lr=base_lr)
    scheduler = LinearLR(optim,start_factor=lr_init/base_lr,end_factor=1,total_iters=optim_params['warmup_steps'])
    run_config={'model_params':model_params,'train':training_params,'opti':optim_params}

    run_config['group'] = group

    trainer = MinGPT_Trainer(model=model,optim=optim,scheduler=scheduler,
                            train_dataset=train_dataset,valid_dataset=val_dataset, detokenizer=tokenizer,
                            run_name=run_name, project_name=project_name, state_save_loc=training_params['state_save_loc'], 
                            device=device,  run_config=run_config)

    
    if(os.path.exists(load_from)):
        print('loading pre-trained model')
        trainer.load_model_from_state(load_from)
        # trainer.load_state(os.path.join(training_params['state_save_loc'],project_name,'state',run_name+'.state'))
    else :
        raise ValueError(f'No pretrained model found at {load_from}')
    
    trainer.stepnum =1

    print(f'Will validate every : {valid_every} steps')

    trainer.train_steps(steps=steps_to_train,save_every=2000,aggregate=aggregate,
                        backup_every=training_params['backup_every'],step_log=training_params['step_log'],
                        batch_size=batch_size,valid_every=valid_every,resume_batches=False,pickup=pickup)
    


def prepare_training(blueprint,f_name,b_name,dataset_name=None,pertub_rate=0.,matrix=None,num_flips=None):
    """
        Adjusts blue_print, and creates perturbed matrix dataset to train on if necessary.

        Args:
        blueprint: blueprint dict to adjust
        f_name: name of the forward model to use
        b_name: name of the backward model to use
        dataset_name: name of the dataset to use. If set, will use this dataset. If None, will create it
        matrix: name of the matrix to use. Must be set if dataset_name is None (will default the matid25).
        num_flips: number of flips to apply to the matrix. Must be set if dataset_name is None.
        pertub_rate: perturbation rate %, how much to randomly flip bits.

        Returns: 3-uple
            blueprint: adjusted blueprint dict
            load_from: path to pretrained model
            dataset_name: name of the dataset used
    """
    if(matrix is None):
        matrix = 'Matid_25'

    f_load_from = os.path.join(cur_path,'trained_states',f_name+'.state')
    b_load_from = os.path.join(cur_path,'trained_states',b_name+'.state')

    f_blueprint = copy.deepcopy(blueprint)
    b_blueprint = copy.deepcopy(blueprint)

    if(not os.path.exists(f_load_from) or not os.path.exists(b_load_from)):
        raise ValueError(f'No pretrained model found at {f_load_from} or {b_load_from}')
    
    if(dataset_name is None):
        assert num_flips is not None, 'If dataset_name is None, num_flips must be set !'
        assert matrix is not None, 'If dataset_name is None, matrix must be set !'

        # No dataset name means we create it.
        #### Create perturbed matrix
        pert_mat_name = save_perturb_matrix(matrix,num_flips=num_flips)

        #### Create small perturbed dataset
        dataset_name = gen_data(pert_mat_name,num_vectors=int(5e4),pertub_rate=pertub_rate) ###### IMPORTANT : should match pert_rate of mother matrix

    dataset_path = os.path.join(cur_path,'datasets',f'{dataset_name}.h5')
    assert os.path.exists(dataset_path), f'Dataset {dataset_path} not found. Please run with dataset_name=None to create it !'


    f_blueprint['training_params']['dataset_folder'] = os.path.join(cur_path,'datasets',f'{dataset_name}.h5')
    f_blueprint['training_params']['backwards'] = False


    b_blueprint['training_params']['dataset_folder'] = os.path.join(cur_path,'datasets',f'{dataset_name}.h5')
    b_blueprint['training_params']['backwards'] = True

    return (f_blueprint,b_blueprint),(f_load_from, b_load_from), dataset_name

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Starts training of Predictor model given a JSON config file. \
                                     Use 'gen_run.py' to create a JSON config file.")
    parser.add_argument("orig_mod_name", help="Name of model to copy from")
    parser.add_argument("-m", "--matrix",required=True, type=str, help="Name of the matrix to use. If set, will generate a perturbed dataset from this matrix.")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("-p", "--nopickup", action='store_true', help="If set, will not try to pickup from last checkpoint.")
    parser.add_argument("-l", "--blueprint", type=str, help="Name of the json training blueprint to use. If not set, uses gpt2-medium blueprint.")
    parser.add_argument("-pr", "--perturb_rate", type=float, default=None, help="Perturbation rate of the matrix. If set, will generate a perturbed dataset from this matrix. Otherwise, use blueprint one.")
    num_flips = 10
    
    args = parser.parse_args()
    perturb_rate = args.perturb_rate

    blueprint = args.blueprint
    pickup = not args.nopickup
    device = args.device

    f_name = args.orig_mod_name
    b_name = f_name+'_b'

    if(blueprint is None):
        blueprint = json.load(open('TrainParams/gpt2-medium.json','r'))
    else :
        blueprint = json.load(open(blueprint,'r'))

    if(perturb_rate is None):
        perturb_rate = blueprint['training_params']['perturb_rate']

    tokenizer_name = 'GaloisMat'

    (f_blue,b_blue), (f_load,b_load), dataset_name = prepare_training(blueprint,f_name,b_name,dataset_name=None,matrix=args.matrix,num_flips=num_flips,pertub_rate=perturb_rate)

    print('training forward : ')
    ## Train_forward : 
    run_name = f_load.split('/')[-1].split('.')[0]+f'_pert{perturb_rate:.2f}'+f'_flips{num_flips}'
    train(f_blue,f_load,device,run_name=run_name,tokenizer_name=tokenizer_name,pickup=pickup)

    print('training backward : ')
    ## Train_backward :
    run_name = b_load.split('/')[-1].split('.')[0]+f'_pert{perturb_rate:.2f}'+f'_flips{num_flips}'
    train(b_blue,b_load,device,run_name=run_name,tokenizer_name=tokenizer_name,pickup=pickup)

    print('Removing dataset : ')
    os.remove(os.path.join(cur_path,'datasets',f'{dataset_name}.h5'))