"""
    Script that runs in sequence 25*25 natural languages training runs with different sparsities.
    By default, uses the config files in TrainParams/sparsitysizes
"""

from train_script import train
import os, pathlib
curpath = pathlib.Path(__file__).parent.absolute().as_posix()

if __name__=='__main__':
    sparsedir = os.path.join(curpath,'TrainParams','sparsitysizes')
    device = 'cuda:0'

    for train_run in os.listdir(sparsedir):
        print(f'Will train : {train_run}')
        ## BELOW : Modify project_name and device to your needs
        ## The results can be 'grouped' in wandb using the 'group' keyword
        train(os.path.join(sparsedir,train_run),tokenizer_name='GaloisMat',project_name='Sparsities',device=device,group=train_run.split('_')[1],load=False,pickup=False)
