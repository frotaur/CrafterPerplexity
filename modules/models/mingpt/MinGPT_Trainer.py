import torch, torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb, random
from torchenhanced import Trainer
from .MinGPT import MinGPT
from ...TokenText import TokenTextBOS
from ...simpletokenizer import SimpleTokenizer


class MinGPT_Trainer(Trainer):
    def __init__(self, model: MinGPT, train_dataset: TokenTextBOS, valid_dataset : TokenTextBOS,
                 detokenizer :SimpleTokenizer=None, optim: Optimizer = None, scheduler: _LRScheduler = None, 
                 state_save_loc=None, device: str = 'cpu', run_name: str = None, project_name: str = None,
                 run_config: dict ={}):
        super().__init__(model, optim, scheduler, state_save_loc=state_save_loc, device=device, run_name=run_name, project_name=project_name, run_config=run_config)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # For logging :
        self.batch_loss = []

        self.detokenizer= detokenizer

        self.mean_loss_val = []
        # Print number of parameters
        print(f"Number of parameters : {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

    def get_loaders(self,batch_size,num_workers=0):
        t_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        v_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # self.text_table = wandb.Table(columns=["batches","text"])
        # self.token_perptable = wandb.Table(columns=["batches","token_perplexity"])
        return t_dataloader, v_dataloader

    def process_batch(self, batch_data):
        # Check if correct, but should be :
        loss = self.compute_loss(batch_data) #(B,T) loss

        if(self.do_batch_log) :
            wandb.log({'lr' : self.scheduler.get_last_lr()[0]},commit=False)

        return loss.mean()

    def compute_loss(self, batch_data):
        token_in, token_truth = batch_data
        token_in = token_in.to(self.device)
        token_truth = token_truth.to(self.device) # (B, T)

        token_in = self.model.forward(token_in).transpose(1,2) # (B, vsize, T)

        # Check if correct, but should be :
        loss = F.cross_entropy(token_in,token_truth,reduction='none') # (B, T)

        return loss

    def process_batch_valid(self, batch_data):
        # Check if correct, but should be :
        loss = self.compute_loss(batch_data).mean(dim=0)
        self.mean_loss_val.append(loss)


        return loss.mean()
        

    def valid_log(self):
        mean_loss_tok = sum(self.mean_loss_val)/(len(self.mean_loss_val)) # (n_tok) normally
        print('mean_loss_tok shape : ', mean_loss_tok.shape)
        data = [[f'{i:03}', mean_loss_tok[i].item()] for i in range(mean_loss_tok.shape[0]) ]
        table = wandb.Table(data=data, columns=["token", "loss"])
        wandb.log(
            {
                f"token_loss:{self.batches : 04d}": wandb.plot.bar(
                    table, "token", "loss", title=f"Batch : {self.batches : 04d}"
                )
            }
        )

        self.mean_loss_val= []