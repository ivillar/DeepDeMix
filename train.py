import nussl
import torch
import json
import glob
import numpy as np
from nussl.datasets import transforms as nussl_tfm
from models import BaselineBiLSTM
from common import utils, data, viz
from pathlib import Path
from train_utils import val_step, train_step

musdb18_formatted_root = './data/formatted/'

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

# create data generator for training data
train_data = data.on_the_fly(stft_params, transform=tfm, 
    fg_path=train_folder, num_mixtures=6000, coherent_prob=1.0)
train_dataloader = torch.utils.data.DataLoader(
    train_data, num_workers=1, batch_size=16)

# create data generator for validation data
val_data = data.on_the_fly(stft_params, transform=tfm, 
    fg_path=val_folder, num_mixtures=300, coherent_prob=1.0)
val_dataloader = torch.utils.data.DataLoader(
    val_data, num_workers=1, batch_size=16)

# define the number of features
nf = stft_params.window_length // 2 + 1
model = BaselineBiLSTM.build(nf, 1, 50, 1, True, 0.0, 1, 'sigmoid')
model.to(DEVICE)

# define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nussl.ml.train.loss.L1Loss()

def train_step_wrapped(engine, batch):
    return train_step(model, loss_fn, optimizer, engine, batch)

def val_step_wrapped(engine, batch):
    return val_step(model, loss_fn, engine, batch)

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
trainer.run(
    train_dataloader, 
    epoch_length=10, 
    max_epochs=30
)
print("Finished training.")
print("Now evaluating.")

separator = nussl.separation.deep.DeepMaskEstimation(
    nussl.AudioSignal(), model_path='checkpoints/best.model.pth',
    device=DEVICE,
)

test_folder = musdb18_formatted_root + "test"
test_data = data.mixer(stft_params, transform=None, 
    fg_path=test_folder, num_mixtures=MAX_MIXTURES, coherent_prob=1.0)

item = test_data[0]

separator.audio_signal = item['mix']
estimates = separator()
# Since our model only returns one source, let's tack on the
# residual (which should be accompaniment)
estimates.append(item['mix'] - estimates[0])

viz.show_sources(estimates)

tfm = nussl_tfm.Compose([
    nussl_tfm.SumSources([['bass', 'drums', 'other']]),
])
test_dataset = nussl.datasets.MUSDB18(subsets=['test'], transform=tfm)

# Just do 5 items for speed. Change to 50 for actual experiment.
for i in range(50):
    item = test_dataset[i]
    separator.audio_signal = item['mix']
    estimates = separator()

    source_keys = list(item['sources'].keys())
    estimates = {
        'vocals': estimates[0],
        'bass+drums+other': item['mix'] - estimates[0]
    }

    sources = [item['sources'][k] for k in source_keys]
    estimates = [estimates[k] for k in source_keys]

    evaluator = nussl.evaluation.BSSEvalScale(
        sources, estimates, source_labels=source_keys
    )
    scores = evaluator.evaluate()
    output_folder = Path(output_folder).absolute()
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / sources[0].file_name.replace('wav', 'json')
    with open(output_file, 'w') as f:
        json.dump(scores, f, indent=4)


json_files = glob.glob(f"*.json")
df = nussl.evaluation.aggregate_score_files(
    json_files, aggregator=np.nanmedian)
nussl.evaluation.associate_metrics(separator.model, df, test_dataset)
report_card = nussl.evaluation.report_card(
    df, report_each_source=True)
print(report_card)

print("Saving model...")
separator.model.save('checkpoints/best.model.pth')