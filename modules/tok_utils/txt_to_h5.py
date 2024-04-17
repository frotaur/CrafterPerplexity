from .pt_to_h5 import make_h5
from .do_tokenize import tokenize
import torch, os
from .. import SimpleTokenizer
import shutil


def tok_text_to_h5(text,premade_dict=None,destination_folder='h5data',tokenizer_folder='tokenizers', tokenizer_name=None,output_name='last_tokenization',delete_txt=True):
    """
        Tokenizes text to .h5 file, training tokenizer on the go. Saves the tokenizer dictionary in tokenizer_folder. 
        Saves the h5 file in './h5data' folder by default. Deletes the .txt file (by default) and the intermediate .pt files.

        Args :
            text (str) : Location of text to tokenize
            premade_dict (dict, optional): Dictionary to use as tokenizer. Defaults to None, in which case a new one is created.
            destination_folder (str, optional): Folder to save the .h5 file. Defaults to 'h5data'.
            tokenizer_folder (str, optional): Folder to save the tokenizer dictionary. Defaults to 'tokenizers'.
            tokenizer_name (str, optional): Name of the tokenizer dictionary. Defaults to None, in which case it is the same as the text file.
            output_name (str, optional): Name of the tokenization. Defaults to 'last_tokenization'.
            delete_txt (bool, optional): If True, deletes the .txt file.
    """
    if(premade_dict=={}):
        premade_dict=None
    
    token_dict = tokenize(text,output_name='last_tokenization',premade_dict=premade_dict)

    os.makedirs(tokenizer_folder,exist_ok=True)
    if(tokenizer_name is None):
        tokenizername = text.split('.')[0]
    else :
        tokenizername = tokenizer_name
    tok_loc = os.path.join(tokenizer_folder,tokenizername+'.pt')
    torch.save(token_dict,tok_loc) # Change tokenizer name here
    
    toki = SimpleTokenizer(tok_loc)
    make_h5(os.path.join('tokendata','last_tokenization'),tokenizer=toki,destination_folder=destination_folder,data_name=output_name)
    if(delete_txt):
        os.remove(text)
    shutil.rmtree(os.path.join('tokendata','last_tokenization'))