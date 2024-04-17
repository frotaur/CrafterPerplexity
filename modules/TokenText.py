from torch.utils.data import Dataset
import torch,os, h5py

class TokenTextBOS(Dataset):
    """
        Dataset of tokenized phrase of fixed length. Produces tuples of text, and the text shifted by one
        token, to be used as input and target for language modelling. Uses memory mapping, with hdf5.
        The attention_length will be equal to the phrase length, and the stride will be equal to the attention length.
        
        ADDED BOS TOKEN. Returns sequence length which are 'phrase_length+1', and the first token is the BOS token.

        Args:
        text_location : location of the tokenized text tensor
        backwards : if true, the text will be read backwards
        bos_token_id : id of the beginning of sentence token. Default is 0
    """

    def __init__(self,h5_file :str, backwards=False, bos_token_id=0):
        self.h5_file = h5_file
        self.backwards = backwards
        self.bos = bos_token_id


        if(not os.path.isfile(self.h5_file)):
            raise ValueError(f'File/Folder {self.h5_file} not found')
        
        self.h5_file = h5py.File(self.h5_file, 'r')
        self.text_tensor = self.h5_file['tokens']

        self.attn_length = self.text_tensor.shape[1] # because we need to have a target for each input, and we have bos
        
        self.num_tokens = self.text_tensor.shape[0]*self.text_tensor.shape[1]
        self.length = self.text_tensor.shape[0]# Number of phrases

        print(f'Dataset contains {self.num_tokens/1e6:.2f}M tokens, resulting in {self.length//1000}k examples.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
            Returns a tuple of (input, target) tensors, each of shape (data_length+1)

            For now, when backwards, we still give the examples in the 'forward' way, but
            we flip them. Maybe there is some reason why this is no bueno, but I don't think so.
        """

        if(self.backwards):
            return self.add_BOS(torch.tensor(self.text_tensor[idx,1:],dtype=torch.long).flip(dims=(0,))), \
                torch.tensor(self.text_tensor[idx,:],dtype=torch.long).flip(dims=(0,))
        else :
            return self.add_BOS(torch.tensor(self.text_tensor[idx,:-1],dtype=torch.long)), \
            torch.tensor(self.text_tensor[idx,:],dtype=torch.long)

    def add_BOS(self,tens):
        """
            Adds a BOS token at the beginning of the tensor, and returns it.

            Args:
            tens : tensor of shape (attn_length)
        """
        return torch.cat([torch.tensor([self.bos],dtype=torch.long),tens],dim=0) # (attn_length+1)

class TokenTextBOSAttention(Dataset):
    """
        Dataset of tokenized phrase of fixed length. Produces tuples of text, and the text shifted by one
        token, to be used as input and target for language modelling. Uses memory mapping, with hdf5.
        The attention lenght can be lower than the size of the phrase, each time the location of the attended message is random.
        
        Args:
        text_location : location of the tokenized text tensor
        backwards : if true, the text will be read backwards
        bos_token_id : id of the beginning of sentence token. Default is 0
        attention_size : number of tokens returned. Note it include one BOS token.
    """

    def __init__(self,h5_file :str, backwards=False, bos_token_id=0, attention_size=5):
        self.h5_file = h5_file
        self.backwards = backwards
        self.bos = bos_token_id


        if(not os.path.isfile(self.h5_file)):
            raise ValueError(f'File/Folder {self.h5_file} not found')
        
        self.h5_file = h5py.File(self.h5_file, 'r')
        self.text_tensor = self.h5_file['tokens']
        assert attention_size<=self.text_tensor.shape[1], 'Attention size cannot be larger than phrase length'
        phrase_size = self.text_tensor.shape[1]
        self.attn_length = attention_size # because we need to have a target for each input, and we have bos
        
        self.num_tokens = self.text_tensor.shape[0]*self.text_tensor.shape[1]
        self.length = self.text_tensor.shape[0]# Number of phrases

        print(f'Dataset contains {self.num_tokens/1e6:.2f}M tokens, resulting in {self.length//1000}k examples.')

        self.attn_start_sequence = torch.randperm(phrase_size-self.attn_length-1)
        self.len_start_seq = len(self.attn_start_sequence)
        self.current_attn_start = 0
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
            Returns a tuple of (input, target) tensors, each of shape (data_length+1)

            For now, when backwards, we still give the examples in the 'forward' way, but
            we flip them. Maybe there is some reason why this is no bueno, but I don't think so.
        """
        start = self.attn_start_sequence[self.current_attn_start]
        self.current_attn_start = (self.current_attn_start+1)%self.len_start_seq
        full = torch.tensor(self.text_tensor[idx,:],dtype=torch.long)
        phrase = torch.tensor(self.text_tensor[idx,start:start+self.attn_length],dtype=torch.long)
        print('phrase :', phrase)
        if(self.backwards):
            return self.add_BOS(phrase[1:].flip(dims=(0,))), \
                phrase.flip(dims=(0,)),full
        else :
            return self.add_BOS(phrase[:-1]), \
            phrase,full

    def add_BOS(self,tens):
        """
            Adds a BOS token at the beginning of the tensor, and returns it.

            Args:
            tens : tensor of shape (attn_length)
        """
        return torch.cat([torch.tensor([self.bos],dtype=torch.long),tens],dim=0) # (attn_length+1)
