import nussl
import torch
import json
import glob
import numpy as np
from nussl.datasets import transforms as nussl_tfm
from models import CNNAutoEncoder, CNNEncoder
from common import utils, data,
from pathlib import Path
from train_utils import val_step, train_step
from ignite.engine.events import Events

musdb18_formatted_root = './data/formatted/'
saved_model_directory = '/models/'

# location of training data
train_folder = musdb18_formatted_root + "train"
val_folder = musdb18_formatted_root + "valid"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# define features and inputs to encoder model
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
n_s_i = 1
n_s_o = 4

pretrained_encoder_model_path = "/models/pretrained_encoder.pth"

if __name__ == "__main__":

    # Prepare MUSDB
    data.prepare_musdb(musdb18_formatted_root)

    utils.logger()
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

    for mode in ['no_pretrain', 'pretrain']:
        # define encoder model (will is an argument to the autoencoder model)
        encoder = CNNEncoder.build(n_t, n_f, n_c, channels, conv_filter_shapes, 
                    maxpool_kernel_sizes, maxpool_strides, linear_dims, discard_probs)

        if mode == 'pretrain':
            model_checkpoint = torch.load(pretrained_encoder_model_path)
            encoder.load_state_dict(model_checkpoint)

        # define autoencoder model using pretrained parameters
        autoencoder = CNNAutoEncoder.build(encoder, linear_dims[-1], [144], n_t, n_f, n_c, n_s_i, n_s_o)
        autoencoder.to(DEVICE)

        # define loss function and optimizer
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        loss_fn = nussl.ml.train.loss.L1Loss()

        def train_step_wrapped(engine, batch):
            return train_step(autoencoder, loss_fn, optimizer, engine, batch)

        def val_step_wrapped(engine, batch):
            return val_step(autoencoder, loss_fn, engine, batch)

        # Create the engines
        trainer, validator = nussl.ml.train.create_train_and_validation_engines(
            train_step_wrapped, val_step_wrapped, device=DEVICE
        )

        # We'll save the output relative to this notebook.
        output_folder = Path('.').absolute()

        # Adding handlers from nussl that print out details about model training
        # run the validation step, and save the models.
        nussl.ml.train.add_stdout_handler(trainer, validator)
        nussl.ml.train.add_validate_and_checkpoint(output_folder, autoencoder, 
            optimizer, train_data, trainer, val_dataloader, validator)

        @trainer.on(Events.ITERATION_COMPLETED(every=10))
        def log_training(engine):
            batch_loss = engine.state.output
            lr = optimizer.param_groups[0]['lr']
            e = engine.state.epoch
            n = engine.state.max_epochs
            i = engine.state.iteration

        print(f"Currently training {mode} model...")

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

        tfm = nussl_tfm.Compose([
            nussl_tfm.SumSources([['bass', 'drums', 'other']]),
        ])
        test_dataset = nussl.datasets.MUSDB18(subsets=['test'], transform=tfm)

        # Test model on test data.
        for i in range(len(test_dataset)):
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

        print(f"Saving {mode} model...")
        separator.model.save(f'/models/{mode}_autoencoder.pth')