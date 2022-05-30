import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from torch import nn
import torch
from common import argbind
from torch.nn import LayerNorm, Linear, Conv2d, ReLU, MaxPool2d, Dropout

class Model(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        super().__init__()
        
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size, 
            num_layers, bool(bidirectional), dropout
        )
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size, 
                                   num_sources, activation, 
                                   num_audio_channels)
        
    def forward(self, data):
        mix_magnitude = data # save for masking
        
        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        estimates = mix_magnitude.unsqueeze(-1) * mask
        
        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output

class BaselineBiLSTM(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        super().__init__()
        
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size, 
            num_layers, bool(bidirectional), dropout
        )
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size, 
                                   num_sources, activation, 
                                   num_audio_channels)
        
    def forward(self, data):
        mix_magnitude = data # save for masking
        
        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        estimates = mix_magnitude.unsqueeze(-1) * mask
        
        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output
    
    # Added function
    @staticmethod
    @argbind.bind_to_parser()
    def build(num_features, num_audio_channels, hidden_size, 
              num_layers, bidirectional, dropout, num_sources, 
              activation='sigmoid'):
        # Step 1. Register our model with nussl
        nussl.ml.register_module(BaselineBiLSTM)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'BaselineBiLSTM',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'activation': activation
                }
            }
        }
        
        # Step 2b: Define the connections between input and output.
        # Here, the mix_magnitude key is the only input to the model.
        connections = [
            ['model', ['mix_magnitude']]
        ]
        
        # Step 2c. The model outputs a dictionary, which SeparationModel will
        # change the keys to model:mask, model:estimates. The lines below 
        # alias model:mask to just mask, and model:estimates to estimates.
        # This will be important later when we actually deploy our model.
        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['estimates', 'mask',]
        config = {
            'name': 'BaselineBiLSTM',
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)

class ConvBlock(nn.Module):
    def __init__(self, channels, filter_shapes):
        super().__init__()
        self.conv2d_1 = Conv2d(channels[0], channels[1], filter_shapes[0], padding = 'same')
        self.relu_1 = ReLU()
        self.conv2d_2 = Conv2d(channels[1], channels[2], filter_shapes[1], padding = 'same')
        self.relu_2 = ReLU()

    def forward(self, data):
        data = self.conv2d_1(data)
        data = self.relu_1(data)
        data = self.conv2d_2(data)
        output = self.relu_2(data)

        return output     

class MaxPoolBlock(nn.Module):
    def __init__(self, kernel_size, stride, discard_prob):
        super().__init__()
        self.maxpool2d = MaxPool2d(kernel_size = kernel_size, stride = stride)
        self.dropout = Dropout(discard_prob)

    def forward(self, data):
        data = self.maxpool2d(data)
        output = self.dropout(data)
        return output

class LinearBlock(nn.Module):
    def __init__(self, linear_in, linear_out, discard_prob):
        super().__init__()
        self.linear = Linear(linear_in, linear_out)
        self.relu = ReLU()
        self.dropout = Dropout(discard_prob)

    def forward(self, data):
        data = self.linear(data)
        data = self.relu(data)
        output = self.dropout(data)
        return output

class CNNEncoder(nn.Module):
    def __init__(self, n_t, n_f, n_c, channels, conv_filter_shapes, 
                 maxpool_kernel_sizes, maxpool_strides, linear_dims, discard_probs):
        super().__init__()
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = LayerNorm((n_c, n_t, n_f))

        self.convblock_1 = ConvBlock([n_c] + channels[:2], conv_filter_shapes[:2])
        self.maxpoolblock_1 = MaxPoolBlock(maxpool_kernel_sizes[0], maxpool_strides[0], discard_probs[0])
        self.convblock_2 = ConvBlock(channels[1:], conv_filter_shapes[2:])
        self.maxpoolblock_2 = MaxPoolBlock(maxpool_kernel_sizes[1], maxpool_strides[1], discard_probs[0])

        self.linearblock_1 = LinearBlock(linear_dims[0], linear_dims[1], discard_probs[0])
        self.linearblock_2 = LinearBlock(linear_dims[1], linear_dims[2], discard_probs[1])
        self.final_linear = Linear(linear_dims[2], linear_dims[3])
        
    def forward(self, data):
        """
        input will be of shape (m, n_t, n_f, n_c, n_s), where:
          m = batch size,
          n_t = number of time steps
          n_f = number of frequency bins,
          n_c = number of audio channels,
          n_s = numbre of sources
        """
        data = self.amplitude_to_db(data)
        # change to (m, n_s, n_c, n_t, n_f)
        data = self.input_normalization(
            torch.permute(
                data, (0, 4, 3, 1, 2)
                )
            )
        
        m, n_s, n_c, n_t, n_f = data.size()

        # change to (m * n_s, n_c, n_t, n_f)
        data = torch.reshape(data, (-1, n_c, n_t, n_f))

        data = self.convblock_1(data)
        data = self.maxpoolblock_1(data)
        data = self.convblock_2(data)
        data = self.maxpoolblock_2(data)

        data = torch.reshape(data, (m * n_s, -1))
        data = torch.reshape(data, (m, n_s, -1))


        data = self.linearblock_1(data)
        data = self.linearblock_2(data)
        data = self.final_linear(data)

        output = {
            'mask': data,
            'estimates': data
        }
        return output
"""
    # Added function
    @staticmethod
    @argbind.bind_to_parser()
    def build(n_t, n_f, n_c, channels, conv_filter_shapes, 
                 maxpool_kernel_sizes, maxpool_strides, linear_dims, discard_probs):
        # Step 1. Register our model with nussl
        nussl.ml.register_module(CNNEncoder)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'CNNEncoder',
                'args': {
                    'n_t': n_t,
                    'n_f': n_f,
                    'n_c': n_c,
                    'channels': channels,
                    'conv_filter_shapes': conv_filter_shapes,
                    'maxpool_kernel_sizes': maxpool_kernel_sizes,
                    'maxpool_strides': maxpool_strides,
                    'linear_dims': linear_dims,
                    'discard_probs': discard_probs
                }
            }
        }
        
        # Step 2b: Define the connections between input and output.
        # Here, the mix_magnitude key is the only input to the model.
        connections = [
            ['model', ['source_magnitudes']]
        ]
        
        # Step 2c. The model outputs a dictionary, which SeparationModel will
        # change the keys to model:mask, model:estimates. The lines below 
        # alias model:mask to just mask, and model:estimates to estimates.
        # This will be important later when we actually deploy our model.
        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['estimates', 'mask',]
        config = {
            'name': 'CNNEncoder',
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)
"""