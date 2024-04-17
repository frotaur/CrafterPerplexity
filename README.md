# Artificial Arrow of Time experiments
This code repository contains the necessary code to reproduce the experiments of section 3 of the paper 'Arrows of Time for Large Language Models'.

**NOTE** : The codebase uses wandb for logging purposes. It has NOT been tested without a valid wandb account logged in, so for better results, please login with a wandb account in the command line first.

**NOTE** : Installing through `pip install -r requirements.txt` might install torch without CUDA on WINDOWS. For Windows, prefer installing CUDA separately, followed by the other dependencies.

**NOTE** : Code was only tested on Linux.

## Dataset generation
The dataset used where generated using code. Due to their size, they are not provided, but here are the steps to regenerate them :

### Prime products
To regenerate the prime-products dataset, simply run the script `prime-product_generator.py`. It will save the resulting '.h5' dataset in h5data/primes_1_5.h5.
 It also possible to generate different datasets by altering the parameters at the top of the script :

```
# ------------------- PARAMETERS, CHANGE HERE -------------------
n_pairs = 10**8 # Number of examples to generate
(min_n_digits, max_n_digits) = (1, 5) # Digit range of primes, inclusive
mul_space, eq_space = (' ', ' ' * 3) # space before/after '*', space before/after '=' 
input_file_name = 'list-of-prime-numbers.txt' # List of prime numbers to draw with. Goes up to 19 999 999.
```

The dataset will be saved as `primes_<min-digits>_<max-digits>.h5`

### Linear Languages
To generate a Linear Language dataset, open a terminal inside the `galois` folder. Some pre-generated matrices are located in the 'matrices' folder, and the 'sparsemat' folder. The names are in the format `Matid<matrix-size>_<k1>_<k2>.pkl`, where `k1` is the number of non-zero elements additionally to the diagonal (i.e. `Matid20_0` is the `20*20` identity), and `k2` is the same, but for the inverse matrix.

To generate the datasets, run `gendataset.py`. By default, this will generate the $F_2^{20}$ linear language dataset with a matrix with $k=6$, used in the paper to compare adaptability for sparse update to a learned prior.
To generate the datasets used for the 'loss vs sparsity' graph, look in `gendataset.py`, comment the first part, and uncomment the second one, then run the script :

```
    # matrice_name = 'Matid20_6_15' # Name of the matrix to use to generate the Linear Language dataset
    # pertub_rate = 0.01 # Probability that a bit will be flipped
    # num_vectors = int(5e6) # Number of examples in the linear language dataset

    # gen_data(matrice_name,num_vectors=num_vectors,pertub_rate=pertub_rate)

    matrice_names= [mat.split('.')[0] for mat in os.listdir('sparsemat') if 'Mat' in mat] # List of matrices used for the sparsity learnability experiments
    for matrice_name in matrice_names
        pertub_rate = 0.01
        gen_data(matrice_name,num_vectors=int(5e5),pertub_rate=pertub_rate,folder='sparsemat')
```

To make other, custom dataset, first generate the appropriate matrix using `matmuldatagen.py`. On the bottom, you can specify the matrix size, and the number of non-zero elements desired. Once the matrix is generated and saved in `matrices`, modify `gendataset.py` appropriately to generate the appropriate dataset.

## Training
If the datasets mentioned above have been generated, you can reproduce the experiments of the paper with the following calls to the scripts :
### Sparsity sizes :
Run `python train_sparsities.py` to train all matrices sparsities subsequently. **NOTE** : By default, it trains on 'cuda:0'. To change this, head into train_sparsities, and chance the 'device' parameter  appropriately.

### Perfect learning of sparse linear language
To reproduce the training of Figure 10 in the paper (needed for the sparse updates experiment), run the following script, after having generated the dataset as mentioned above :
`python train_script.py -d <device> -t GaloisMat TrainParams/linear/Matid20_6_0.01.json`, choosing the device appropriately. This will launch the training of the forward model.
`python train_script.py -d <device> -t GaloisMat TrainParams/linear/Matid20_6_0.01_b.json`, choosing the device appropriately. This will launch the training of the backward model.

In case the training crashes, it can be resumed from the last checkpoint simply by rerunning the command. If it is needed to train more after the training run is complete, re-run the command with the `-p` option, to pick up training from the last position, and train further.

### Sparse updates experiments
To re-run the experiments for the sparse updates, some setup is required. First, you need to run the training of the above section. Located in `runs/ArtificialPerplexity/state` are the saved checkpoints for the models, normally named `Matid20_6_0.01.state` and `Matid20_6_0.01_b.state`. Copy them inside `perturb_training/trained_states`. Change the current directory to `perturb_training`, and run the following :

```
python train_perturb.py -d <device>
```
Where <device> denotes the device on which to do the training. With the default configuration, this will perturb the $20*20$ sparse matrix that was learned in the previous experiment, then run for a few gradients steps with the learnt backward and forward priors (repeating the experiment 20 times with different perturbations). Results for the loss are logged in wandb, in the project `PerturbatedPerplexity`. Runs that end in 'B' are backward, 'F' forwards. You can group the loss graph according to the training direction, which reproduces the results reported in the paper.

To change the default parameters, look inside `pert_run.py` :
```
    ## PARAMETERS TO CHANGE ##
    matrix = 'Matid20_6_15' ## Which matrix to use to generate sparse updated linear language dataset
    model_f = 'Matid20_6_0.01' ## Which trained model to use as base for forward model
    model_b = 'Matid20_6_0.01_b' ## Which trained model to use as base for backward model

    num_flips = 4 ## Number of elements to flip in the sparse perturbation of the matrix
    ###### DO NOT CHANGE BELOW #####
```

When changing the matrix around which to perturb, make sure it is present inside `perturb_training/matrices`. Also make sure that the specified `model_f` and `model_b` exist inside `trained_states`. E.g., if `model_b = 'hello_there'`, there should be a file `perturb_training/trained_states/hello_there.state`, which was saved during a training run using train_script.


### Custom models and config JSON files
For all the above experiments, a GPT-1 sized transformer was used. To use a different one, you need to modify the config `.json` files located in `TrainParams`. We repeat here the information for the config files, also present in the other repository :


### JSON config file
To run the training script, we need to provide it with a path to the JSON config file. Their format slightly depends if training a GPT, GRU or LSTM model. In a nutshell, they contain all the necessary hyperparameters for a training run.

Here is a description of each entry : 

```
{
    "model_params": { # Model parameters
        "vocab_size": 0, # Vocabulary size, always set to 0, determined by dataset
        "n_layers": 12, # Number of Transformer Blocks
        "n_heads": 12, # Number of attention heads
        "embed_dim": 768, # Number of hidden/embedding dimensions
        "attn_length": 0, # Attention length, always set to 0, determined by dataset
        "mlp_ratio": 4.0, # MLP ratio
        "dropout": 0.1, # Dropout inside tranformer blocks
        "embd_dropout": null # Dropout for the token embeddings. Defaults to 0.
    },
    "training_params": { 
        "dataset_folder": "h5data/Matid20_6_15_v5000k_pert0.01.h5", # Location of .h5 linear language dataset to train on
        "batch_size": 180, # Batch size
        "aggregate": 1, # Number of times to aggregate gradients before gradient step. (effective batch_size = aggregate*batch_size)
        "backwards": false, # Whether to train in the backwards direction
        "steps_to_train": 30000, # Number of gradient steps to train. Defaults to one epoch of the dataset.
        "save_every": 200, # Number of steps between each save of the training state.
        "backup_every": 5000, # Number of steps between a backup of the training state.
        "step_log": 50, # Number of steps between each log of training loss in wandb
        "valid_steps": 100, # Number of batches seen during one validation.
        "valid_every":1000, # Number of steps between each validation
        "state_save_loc": "runs" # folder in which to save the training state.
    },
    "optim_params": {
        "lr": 0.0001, # Base learning rate
        "warmup_steps": 1000, # Number of batches until learning rate warms up
        "oscil_steps": 300000, # Number of steps between warm restarts (not used in these experiments)
        "lr_shrink": 0.85, # Shrinking factor of lr between warm restarts(not used in these experiments)
        "lr_init": 1e-07, # Initial learning rate, for warmup(not used in these experiments)
        "lr_min": 1e-06 # Minimum learning rate reached during cosine annealing.(not used in these experiments)
    }
}
```