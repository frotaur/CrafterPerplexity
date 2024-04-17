import torch
from tqdm import tqdm
import os, shutil


def tokenize(text_loc, output_name='last_tokenization', premade_dict=None, fixed=False,max_lines= 1e7):
    """
        Tokenizes text to .pt files, training tokenizer on the go. Returns tokenizer dictionary.
        Saves results in tokendata folder, with name 'output_name'. WARNING it will overwrite previous tokenizations, it same name.

        Args:
            text_loc (str): Location of text to tokenize
            output_name (str, optional): Name of the tokenization. Defaults to 'last_tokenization'.
            premade_dict (dict, optional): Dictionary to use as tokenizer. Defaults to None, in which case a new one is created.
            fixed (bool, optional): If True, will raise an error if a character is not in the dictionary. Defaults to False.
            max_lines (int, optional): Number of lines to save in each pt file. Defaults to 1e7.
    """
    line_tensors = []
    tokendir = os.path.join('tokendata',output_name)
    shutil.rmtree(tokendir,ignore_errors=True)
    os.makedirs(tokendir,exist_ok=True)
    if(premade_dict is not None):
        if("$" not in premade_dict):
            print('WARNING, BOS TOKEN NOT IN PREMADE DICT, ADDING IT AT THE END')
            premade_dict["$"] = len(premade_dict)
        token_dict = premade_dict
    else:
        token_dict ={"$":0}

    with open(text_loc,'r') as file:
        numlines = 0
        for line in tqdm(file):
            numlines+=1
            line = line.strip('\n')
            new_phrase = []
            for charac in line:
                if charac not in token_dict:
                    if fixed:
                        raise ValueError(f'Character {charac} not in dictionary, and fixed=True')
                    token_dict[charac] = len(token_dict)
                new_phrase.append(token_dict[charac])
            
            line_tensors.append(torch.tensor(new_phrase,dtype=torch.uint8))
            if(numlines%max_lines==0):
                torch.save(torch.stack(line_tensors,dim=0),os.path.join(tokendir,f'Data{int(numlines//max_lines)}.pt'))
                line_tensors = []
                
    if(len(line_tensors)>0):
        torch.save(torch.stack(line_tensors,dim=0),os.path.join(tokendir,f'Data{int(numlines//max_lines+1)}.pt'))

    return token_dict
