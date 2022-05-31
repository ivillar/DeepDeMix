import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from torch import nn
import torch
from common import argbind
from torch.nn import LayerNorm, Linear, Conv2d, ReLU, MaxPool2d, Dropout, Upsample, Sigmoid

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
        data = data['source_magnitudes']
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

class UpsampleConvBlock(nn.Module):
    def __init__(self, decoder_channels, decoder_filter_shape, last_block, upsample_size = None):
        """
        decoder_channels      a list/tuple of 2, where the first element is the input
                              channel and the 2nd element is the output channel
        
        decoder_filter_shape: a tuple of the filter shapes

        last_block:           bool indicating whether this is the last block

        upsample_size:        None if this is the last block, if not then it is
                              a tuple of the filter shape of shape
                              (d_c, d_t, d_f)
        """
        super().__init__()
        # (n_c_i, intermediate, (3, 3), padding = 'same')
        self.conv2d = Conv2d(decoder_channels[0], decoder_channels[1], decoder_filter_shape, padding = 'same')
        self.last_block = last_block
        if not self.last_block:
          self.relu = ReLU()
          self.upsample = Upsample(upsample_size)
        else:
          self.sigmoid = Sigmoid()

    def forward(self, data):
        #data.size() = (m, n_s_o, n_c, hidden_dims[0]**1/2, hidden_dims[0]**1/2)
        pdb.set_trace()
        data = self.conv2d(data)
        # data.size() = 
        if not self.last_block:
          
          data = self.relu(data)
          data = self.upsample(data)
        else:
          data = self.sigmoid(data)

        return data 

class CNNDecoder(nn.Module):
    def __init__(self, encoded_dims, hidden_dims, n_t, n_f, n_c, n_s_o, 
                 decoder_channels, decoder_filter_shapes, last_blocks):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.output_dims = (n_t, n_f, n_c, n_s_o)
        self.linear_1 = Linear(encoded_dims, hidden_dims[0])
        self.relu_1 = ReLU()
        self.linear_2 = Linear(hidden_dims[0], hidden_dims[0] * n_s_o * n_c)
        self.relu_2 = ReLU()
        self.upsample_conv_block_1 = UpsampleConvBlock(decoder_channels[:2], decoder_filter_shapes[0], last_blocks[0], upsample_size = (n_t, n_f))
        self.upsample_conv_block_2 = UpsampleConvBlock(decoder_channels[1:3], decoder_filter_shapes[1], last_blocks[1], upsample_size = (n_t, n_f))
        self.upsample_conv_block_3 = UpsampleConvBlock(decoder_channels[2:], decoder_filter_shapes[2], last_blocks)

    def forward(self, data):
        # data.size() = (m, encoded_dims)
        data = self.linear_1(data)
        data = self.relu_1(data)
        # data.size() = (m, hidden_dims[0])
        data = self.linear_2(data)
        data = self.relu_2(data)
        # data.size() = (m, hidden_dims[0] * n_s_o)
        n_t, n_f, n_c, n_s_o = self.output_dims
        hidden_dims = self.hidden_dims

        data = torch.reshape(data, (data.size()[0], n_s_o,
                                    int(hidden_dims[0]**(1/2)), 
                                    int(hidden_dims[0]**(1/2))))
        
        # data.size() = (m, n_s_o, hidden_dims[0]**1/2, hidden_dims[0]**1/2)
        data = self.upsample_conv_block_1(data)
        data = self.upsample_conv_block_2(data)
        data = self.upsample_conv_block_3(data)
        # data.size() = (m, n_s_o, n_t, n_f)

        data = torch.unsqueeze(data, 2)

        # data.size() = (m, n_s_o, n_c, n_t, n_f)
        output = torch.permute(data, (0, 3, 4, 2, 1))
        # data.size() = (m, n_t, n_f, n_c, n_s_o)
        return output

class CNNAutoEncoder(nn.Module):
    def __init__(self, encoder_model, encoded_dims, hidden_dims,
                 n_t, n_f, n_c, n_s_i, n_s_o):
        super().__init__()
        self.n_s_i = n_s_i
        self.encoder = encoder_model
        self.decoder = CNNDecoder(encoded_dims, hidden_dims, n_t, n_f, n_c, n_s_o)
        
    def forward(self, data):

        mix_magnitude = data

        # data.size = (m, n_t, n_f, n_c)
        data = torch.unsqueeze(data, 4)
        # data.size = (m, n_t, n_f, n_c, 1)
        data = torch.repeat_interleave(data, 1, self.n_s_i)
        # data.size = (m, n_t, n_f, n_c, n_s_i)
        data = self.encoder({"source_magnitudes": data})


        data = data["estimates"]
        # data.size = (m, n_s_i, encoded_dims)
        data = torch.reshape(data, (-1, self.n_s_i * data.size()[-1]))
        # data.size = (m, encoded_dims)
        mask = self.decoder(data)

        estimates = mix_magnitude.unsqueeze(-1) * mask
        
        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output

        output = {
            'mask': data,
            'estimates': data
        }
        return output
        
    @staticmethod
    @argbind.bind_to_parser()
    def build(encoder_model, encoded_dims, hidden_dims,
                 n_t, n_f, n_c, n_s_i, n_s_o):
        # Step 1. Register our model with nussl
        nussl.ml.register_module(CNNAutoEncoder)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'CNNAutoEncoder',
                'args': {
                    'encoder_model': encoder_model,
                    'encoded_dims': encoded_dims,
                    'hidden_dims': hidden_dims,
                    'n_t': n_t,
                    'n_f': n_f,
                    'n_c': n_c,
                    'n_s_i': n_s_i,
                    'n_s_o': n_s_o
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
            'name': 'CNNAutoEncoder',
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)