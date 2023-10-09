import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import Adam
from pathlib import Path
import os
from tqdm import tqdm

def exists(x):
    return x is not None

class Trainer(object):
    """
    Basic Trainer
    
    model (NeRFRender): nerf renderer
    sampler (RaySampler): ray sampler
    batch_size (int): Batch size per GPU
    batch_split (bool): Split batch with gradient accumulation. 
    training_lr (float): training leraning rate
    training_num_steps (int): training steps
    learning_rate (float): Learning rate.
    result_folder (str): Output directory.
    amp: if use amp.
    fp16: if use fp16.

    >>> renderer = create_renderer(...)
    >>> ray_sampler = create_ray_sampler(...)
    >>> trainer = Trainer(model=model, ray_sampler=ray_sampler, ....)
    >>> trainer.run()
    """
    def __init__(
        self,
        model,
        *, 
        sampler=None,
        results_folder='./result', 
        train_lr=1e-2,
        train_batch_size=4096,
        train_num_steps=25000,
        gradient_accumulate_every=1,
        adam_betas=(0.9,0.99),
        i_print=100,
        i_image=1000,
        i_save=50000,
        split_batches=False,
        amp=False,
        fp16=False,
        with_tracking=False,
        **kwargs,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            project_dir=results_folder if with_tracking else None,
            log_with="all",
        )

        self.accelerator.native_amp = amp

        self.model = model
        self.sampler = sampler

        self.train_num_steps = train_num_steps
        self.i_save = i_save
        self.i_print = i_print
        self.i_image = i_image
        self.train_batch_size = train_batch_size

        self.results_folder = results_folder
        self.gradient_accumulate_every = gradient_accumulate_every
        self.with_tracking = with_tracking
        self.step = 0

        self.opt = Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)
        
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # accelerator tracking
        if self.with_tracking:
            run = os.path.split(__file__)[-1].split(".")[0]
            self.accelerator.init_trackers(run, config={
                'train_lr':train_lr,
                'train_batch_size':train_batch_size,
                'gradient_accumulate_every':gradient_accumulate_every,
                'train_num_steps':train_num_steps,
            })



    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
            
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def on_train_step(self):
        raise NotImplementedError('not implemeted')

    def on_evaluate_step(self):
        raise NotImplementedError('not implemeted')

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):

                    with self.accelerator.autocast():
                        loss, log_dict = self.on_train_step()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss

                    self.accelerator.backward(loss)

                # all reduce to get the total loss
                total_loss = accelerator.reduce(total_loss)
                total_loss = total_loss.item()
                log_str = f'loss: {total_loss:.3f}'
                
                for k in log_dict.keys():
                    log_str += " {}: {:.3f}".format(k, log_dict[k])
                
                pbar.set_description(log_str)

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                if accelerator.is_main_process:
                    
                    if (self.step % self.i_image == 0):
                        self.on_evaluate_step()

                    if self.step !=0 and (self.step % self.i_save == 0):
                        milestone = self.step // self.i_save
                        self.save(milestone)
                
                pbar.update(1)

        if self.with_tracking:
            accelerator.end_training()