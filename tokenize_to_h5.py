"""
    Given a text dataset, tokenizes it and saves it in h5 format, ready for training.
"""
from modules.tok_utils import tok_text_to_h5
from pathlib import Path

curpath = Path(__file__).parent.absolute().as_posix()


if __name__=='__main__':
    num_dict = {}
    ### PARAMETERS TO CHANGE ####
    text = 'prime-products_1_5.txt' # File to tokenize
    tokenizername = text.split('.')[0]
    output_name= 'primes_1_5'
    premade_dict = {} # Dictionary to use as tokenizer. Defaults to None, in which case a new one is created.
    ##### DO NOT CHANGE BELOW ####

    tok_text_to_h5(text,premade_dict=premade_dict,tokenizer_folder='test_toki',tokenizer_name=tokenizername)