"""
    Training script for GPT models, used for all experiments of section 3 of the paper 'Arrows of Time for Large Language Models'
"""
from modules import MinGPT, MinGPT_Trainer, SimpleTokenizer, TokenTextBOS
import torch, torch.optim,os, argparse,json, pathlib,random
from torch.utils.data import Subset
from torch.optim.lr_scheduler import LinearLR



def train(file_location,tokenizer_name,pickup,project_name = 'ArtificialPerplexity', 
          group=None, load=True,device='cpu'):
    run_name = os.path.splitext(os.path.basename(file_location))[0]

    cur_path = pathlib.Path(__file__).parent.absolute().as_posix()

    if(tokenizer_name is not None):
        ## FOR NOW, THE TOKENIZER IS HARDCODED. IF WE NEED DIFFERENT ONES, WE SHALL SEE WHAT TO DO
        tokenizer_path = os.path.join(cur_path,'tokenizers',f'{tokenizer_name}.pt')

        tokenizer = SimpleTokenizer(tokenizer_path)
    else :
        raise ValueError('No tokenizer name given. Please specify a tokenizer name with the -t flag.')

    with open(file_location,'r') as f :
        configo = json.load(f)
        model_params = configo['model_params']
        training_params = configo['training_params']
        optim_params = configo['optim_params']

    if not os.path.exists(training_params['dataset_folder']):
        raise FileNotFoundError(f"Tried to find dataset folder at \
                                {training_params['dataset_folder']}, but failed. \
                                Make sure there is the folder {training_params['dataset_folder']}\
                                in the same directory.")


    valid_steps =training_params['valid_steps'] # We ask how many validation steps. To get these, we assume 5% of training time allocated for validation.
    valid_percent_time=20 # Time spent validating, in percentage
    valid_percent_time=valid_percent_time/100
    # Note : for now we take the last few % for validation. Preferably, we should shuffle before splitting.
    # However, alot of subtelties with shuffling (partial phrases, consistency across runs, etc), so not implemented yet.
    valid_every = int(valid_steps/valid_percent_time)
    # Backwards training?
    backwards = training_params['backwards']

    random.seed(42) # For deterministic shuffling of dataset

    dataset_path = training_params['dataset_folder']

    motherDataset = TokenTextBOS(h5_file=dataset_path, backwards=backwards)
    
    assert model_params['vocab_size']==0, 'Set vocab size to 0, determined by tokenizer.'
    assert model_params['attn_length']==0, 'Set attn_length to 0, determined by dataset.'

    model_params['vocab_size'] = tokenizer.vocab_size
    model_params['attn_length'] = motherDataset.attn_length

    indices = list(range(len(motherDataset)))
    random.shuffle(indices)
    motherDataset = Subset(motherDataset, indices) # Shuffled dataset

    # To keep it constant even if switching batch_size, I take batch_size=250
    val_inds = valid_steps*training_params['batch_size']
    val_range = range(len(motherDataset)-val_inds,len(motherDataset)) # Validation, last portion of dataset
    keep_range = range(len(motherDataset)-val_inds) # Training, first portion of dataset
    
    #Whether backwards or forwards, its the individual examples that are flipped, not the dataset. So same thing for both !
    train_dataset = Subset(motherDataset, keep_range)
    val_dataset = Subset(motherDataset, val_range)

    print('CHECK THEY ARE THE SAME AS USUAL : ')
    for i in range(1,3):
        idx = i
        print(f'{idx} TRAIN :',tokenizer.detokenize(train_dataset[idx][0]))
        print(f'ANSWER : ',tokenizer.detokenize(train_dataset[idx][1]))
        print(f'{idx} VALID :',tokenizer.detokenize(val_dataset[idx][0]))
        print(f'ANSWER : ',tokenizer.detokenize(val_dataset[idx][1]))



    model = MinGPT(**model_params)

    #====================== TRAINING PARAMETERS =======================
    device = device

    batch_size = training_params['batch_size']
    aggregate = training_params['aggregate']
    totbatches = len(train_dataset)//batch_size
    
    if(training_params['steps_to_train']==None):
        steps_to_train = totbatches*10 # 4 epochs
    else:
        steps_to_train = training_params['steps_to_train']

    print(f'{totbatches=}, {batch_size=}, {len(train_dataset)=}')
    base_lr = optim_params['lr']
    warmup_steps = optim_params['warmup_steps']
    lr_init = optim_params['lr_init']

    print(f'--- Training for ~ {steps_to_train//1000}k minibatches ---')
    #------ Optimizers ------
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = LinearLR(optim,start_factor=lr_init/base_lr,end_factor=1,total_iters=warmup_steps)


    trainer = MinGPT_Trainer(model=model,optim=optim,scheduler=scheduler,
                            train_dataset=train_dataset,valid_dataset=val_dataset, detokenizer=tokenizer,
                            run_name=run_name, project_name=project_name, state_save_loc=training_params['state_save_loc'],
                            device=device, run_config={'group':group,'model_params':model_params,'train':training_params,'opti':optim_params} )

    
    if(os.path.exists(os.path.join(training_params['state_save_loc'],project_name,'state',run_name+'.state')) and (load)):
        trainer.load_state(os.path.join(training_params['state_save_loc'],project_name,'state',run_name+'.state'))
    trainer.stepnum =1

    print(f'Will validate every : {valid_every} steps')
    trainer.train_steps(steps=steps_to_train,save_every=training_params.get('save_every',100),aggregate=aggregate,
                        backup_every=training_params['backup_every'],step_log=training_params['step_log'],
                        batch_size=batch_size,valid_every=valid_every,resume_batches=True,pickup=pickup)






if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Starts training of Predictor model given a JSON config file. \
                                     Use 'gen_run.py' to create a JSON config file.")
    parser.add_argument("file_location", help="Path to the JSON config file. Relative to where you launch the script from.")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("-t", "--tokenizer_name", help="Name of the tokenizer to use (<tok_name>.pt).")
    parser.add_argument("-p", "--nopickup", action='store_true', help="If set, will not try to pickup from last checkpoint.")
    args = parser.parse_args()
    pickup = not args.nopickup
    device = args.device
    tokenizer_name = args.tokenizer_name
    file_location = args.file_location

    train(file_location,tokenizer_name,pickup,device=device)
