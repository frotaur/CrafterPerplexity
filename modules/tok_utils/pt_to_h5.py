import h5py , os, torch
from tqdm import tqdm
from .. import SimpleTokenizer



def make_h5(pt_data_folder, tokenizer:SimpleTokenizer, data_name=None,destination_folder = None):
    """
        Make h5 dataset from a folder of pt files.

        Args:
            pt_data_folder (str): Folder containing pt files
            tokenizer (SimpleTokenizer): Tokenizer to use to visualize data, not used for the h5 file itself.
            data_name (str, optional): Name of the dataset. Defaults to None, in which case the name of the folder is used.
            destination_folder (str, optional): Folder to save the h5 file. Defaults to 'h5data', in which case the current folder is used.
    """

    if(destination_folder is None):
        destination_folder= 'h5data'

    if(data_name is None):	
        tarname = os.path.join(destination_folder,f'{os.path.basename(pt_data_folder)}.h5')
    else :
        tarname = os.path.join(destination_folder,f'{data_name}.h5')
    os.makedirs(os.path.dirname(tarname),exist_ok=True)


    if(os.path.isdir(pt_data_folder)):
        extract_file = os.listdir(pt_data_folder)[0]
        tensor = torch.load(os.path.join(pt_data_folder,extract_file))
        phrase_length = tensor.shape[1]
        print(f'Detected phrase length of {phrase_length} tokens.')

        with h5py.File(tarname, 'w') as f:
            dset = f.create_dataset("tokens", (0,phrase_length), maxshape=(None,phrase_length), dtype='uint8')  # note the maxshape parameter
            
            current_index = 0
            for file in tqdm(os.listdir(pt_data_folder)):
                if os.path.splitext(file)[1]=='.pt':
                    pt_file = os.path.join(pt_data_folder,file)
                    tensor = torch.load(pt_file,map_location=torch.device('cpu')) # (B,length)
                    assert phrase_length==tensor.shape[1]
                    batch_size = tensor.shape[0]
                    print('snippet', tokenizer.detokenize(tensor[0,:]))
                    # Resize the dataset to accommodate the new data
                    dset.resize((current_index + batch_size,phrase_length))
                    
                    # Add the new data to the dataset
                    dset[current_index:current_index+batch_size,:] = tensor.numpy()
                    
                    # Update the current ind
                    current_index += batch_size
    else :
        raise ValueError(f'{pt_data_folder} not found')


