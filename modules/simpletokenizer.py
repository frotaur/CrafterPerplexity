import torch
from pathlib import Path

curpath = Path(__file__).parent.absolute().as_posix()


class SimpleTokenizer:
    """
        Very simple character level tokenizer.

        Args:
            tok_dict_loc (str): Location of the tokenizer dictionary. Should be a torch.save'd dictionary.
    """
    def __init__(self, tok_dict_loc:str):
        self.tok_dict = torch.load(tok_dict_loc)
        self.tok_dict_rev = {v:k for k,v in self.tok_dict.items()}
        self.vocab_size = len(self.tok_dict)	

        print('Got : tok_dict', self.tok_dict)
        print('Got : tok_dict_rev', self.tok_dict_rev)

    def tokenize(self, text:str):
        """
            Tokenizes a string into a torch tensor
        """
        new_phrase = []
        for charac in text:
            if charac not in self.tok_dict:
                raise ValueError(f'Character {charac} not in dictionary')
            new_phrase.append(self.tok_dict[charac])
        return torch.tensor(new_phrase,dtype=torch.uint8)
    
    def detokenize(self, tensor:torch.Tensor):
        """
            Detokenizes a torch tensor into a string
        """
        return ''.join([self.tok_dict_rev.get(int(i),'<UNK>') for i in tensor])