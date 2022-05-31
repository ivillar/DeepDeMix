import nussl
import torch
import json
import glob
import numpy as np
from nussl.datasets import transforms as nussl_tfm
from models import CNNAutoEncoder, CNNEncoder
from common import utils, data, viz
from pathlib import Path
from train_utils import pretrain_step, pretrain_val_step
from ignite.engine.events import Events
from sox import transform
from models import CNNEncoder

musdb18_formatted_root = './data/formatted/'
saved_model_directory = '/models/'
pretrained_encoder_model_path = "/models/pretraned_encoder.pth"


# Prepare MUSDB
data.prepare_musdb(musdb18_formatted_root)

utils.logger()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_MIXTURES = int(1e8) # We'll set this to some impossibly high number for on the fly mixing.

# parameters for short-time fourier transform
stft_params = nussl.STFTParams(window_length=512, hop_length=128)

# Data Transform
tfm = nussl_tfm.Compose([
    nussl_tfm.SumSources([['bass', 'drums', 'other']]), #sum histograms of these channels
    nussl_tfm.MagnitudeSpectrumApproximation(),
    nussl_tfm.IndexSources('source_magnitudes', 1),
    nussl_tfm.ToSeparationModel(), # convert histograms to tensors
])

# location of training data
train_folder = musdb18_formatted_root + "train"
val_folder = musdb18_formatted_root + "valid"

template_event_parameters = {
    'label': ('const', 'vocals'),
    'source_file': ('choose', []),
    'source_time': ('uniform', 0, 7),
    'event_time': ('const', 0),
    'event_duration': ('const', 5.0),
    'snr': ('uniform', -5, 5),
    'pitch_shift': ('uniform', -2, 2),
    'time_stretch': ('uniform', 0.8, 1.2)
}

fg_folder_train = bg_folder_train = train_folder
fg_folder_val = bg_folder_val = val_folder
    
# Initialize our mixing function with our specific source material and event template
mix_func_train = MixClosure(fg_folder_train, bg_folder_train, template_event_parameters)

# Create a nussl OnTheFly data generator
train_data = nussl.datasets.OnTheFly(stft_params = stft_params,
    transform = tfm,               
    num_mixtures = 6000,
    mix_closure = mix_func_train
)
train_dataloader = torch.utils.data.DataLoader(
    train_data, num_workers=2, batch_size=1)

mix_func_val = MixClosure(fg_folder_val, bg_folder_val, template_event_parameters)

# create data generator for validation data
val_data = nussl.datasets.OnTheFly(stft_params = stft_params, 
    transform = tfm, 
    num_mixtures = 10,
    mix_closure = mix_func_val)

val_dataloader = torch.utils.data.DataLoader(
    val_data, num_workers=1, batch_size=1)



###################

n_t, n_f, n_c = 1724, 257, 1
channels = [16, 32, 64, 128]
conv_filter_shapes = [(8, 16), (8, 16), (4, 8), (4, 8)]
maxpool_kernel_sizes = maxpool_strides = [(16, 4), (8, 4)]
n_t_f = n_t // 16
n_f_f = n_f // 4
n_t_f //= 8
n_f_f //= 4

linear_dims = [n_t_f * n_f_f * channels[-1], 500, 250, 125]
discard_probs = [0.5, 0.5, 0.5, 0.5]

model = CNNEncoder.build(n_t, n_f, n_c, channels, conv_filter_shapes, 
                 maxpool_kernel_sizes, maxpool_strides, linear_dims, discard_probs)
model.to(DEVICE)

# define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#loss_fn = nussl.ml.train.loss.L1Loss()
loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)

def train_step_wrapped(engine, batch):
    return pretrain_step(model, loss_fn, optimizer, engine, batch)

def val_step_wrapped(engine, batch):
    return pretrain_val_step(model, loss_fn, engine, batch)

# Create the engines
trainer, validator = nussl.ml.train.create_train_and_validation_engines(
    train_step_wrapped, val_step_wrapped, device=DEVICE
)

# We'll save the output relative to this notebook.
output_folder = Path('.').absolute()

# Adding handlers from nussl that print out details about model training
# run the validation step, and save the models.
nussl.ml.train.add_stdout_handler(trainer, validator)
nussl.ml.train.add_validate_and_checkpoint(output_folder, model, 
    optimizer, train_data, trainer, val_dataloader, validator)

print("Currently training...")

@trainer.on(Events.ITERATION_COMPLETED(every=10))
def log_training(engine):
    batch_loss = engine.state.output
    lr = optimizer.param_groups[0]['lr']
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print(f"Epoch {e}/{n} : {i} - batch loss: {batch_loss}, lr: {lr}")

trainer.run(
    train_dataloader,
    max_epochs=2
)

separator = nussl.separation.deep.DeepMaskEstimation(
    nussl.AudioSignal(), model_path='checkpoints/best.model.pth',
    device=DEVICE,
)

separator.model.save(pretrained_encoder_model_path)