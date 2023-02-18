from argparse import Namespace
import torch
import math
from typing import Union

from .Layer import Conv1d, LayerNorm, LinearAttention
from .Diffusion import Diffusion

class DiffSinger(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.encoder = Encoder(self.hp)
        self.diffusion = Diffusion(self.hp)

    def forward(
        self,
        tokens: torch.LongTensor,
        notes: torch.LongTensor,
        durations: torch.LongTensor,
        lengths: torch.LongTensor,
        genres: torch.LongTensor,
        singers: torch.LongTensor,
        features: Union[torch.FloatTensor, None]= None,
        ddim_steps: Union[int, None]= None
        ):
        encodings, linear_predictions = self.encoder(
            tokens= tokens,
            notes= notes,
            durations= durations,
            lengths= lengths,
            genres= genres,
            singers= singers
            )    # [Batch, Enc_d, Feature_t]

        encodings = torch.cat([encodings, linear_predictions], dim= 1)  # [Batch, Enc_d + Feature_d, Feature_t]

        if not features is None or ddim_steps is None or ddim_steps == self.hp.Diffusion.Max_Step:
            diffusion_predictions, noises, epsilons = self.diffusion(
                encodings= encodings,
                features= features,
                )
        else:
            noises, epsilons = None, None
            diffusion_predictions = self.diffusion.DDIM(
                encodings= encodings,
                ddim_steps= ddim_steps
                )

        return linear_predictions, diffusion_predictions, noises, epsilons


class Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        if self.hp.Feature_Type == 'Mel':
            self.feature_size = self.hp.Sound.Mel_Dim
        elif self.hp.Feature_Type == 'Spectrogram':
            self.feature_size = self.hp.Sound.N_FFT // 2 + 1

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size
            )
        self.note_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Notes,
            embedding_dim= self.hp.Encoder.Size
            )
        self.duration_embedding = Duration_Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size
            )
        self.genre_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Genres,
            embedding_dim= self.hp.Encoder.Size,
            )
        self.singer_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Singers,
            embedding_dim= self.hp.Encoder.Size,
            )
        torch.nn.init.xavier_uniform_(self.token_embedding.weight)
        torch.nn.init.xavier_uniform_(self.note_embedding.weight)
        torch.nn.init.xavier_uniform_(self.genre_embedding.weight)
        torch.nn.init.xavier_uniform_(self.singer_embedding.weight)

        self.fft_blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.ConvFFT.Head,
                ffn_kernel_size= self.hp.Encoder.ConvFFT.FFN.Kernel_Size,
                dropout_rate= self.hp.Encoder.ConvFFT.Dropout_Rate
                )
            for _ in range(self.hp.Encoder.ConvFFT.Stack)    
            ])

        self.linear_projection = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.feature_size,
            kernel_size= 1,
            bias= True,
            w_init_gain= 'linear'     
            )

    def forward(
        self,
        tokens: torch.Tensor,
        notes: torch.Tensor,
        durations: torch.Tensor,
        lengths: torch.Tensor,
        genres: torch.Tensor,
        singers: torch.Tensor
        ):
        x = \
            self.token_embedding(tokens) + \
            self.note_embedding(notes) + \
            self.duration_embedding(durations) + \
            self.genre_embedding(genres).unsqueeze(1) + \
            self.singer_embedding(singers).unsqueeze(1)
        x = x.permute(0, 2, 1)  # [Batch, Enc_d, Enc_t]

        for block in self.fft_blocks:
            x = block(x, lengths)   # [Batch, Enc_d, Enc_t]

        linear_predictions = self.linear_projection(x)  # [Batch, Feature_d, Enc_t]

        return x, linear_predictions

class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,
        ffn_kernel_size: int,
        dropout_rate: float= 0.1,
        ) -> None:
        super().__init__()

        self.attention = LinearAttention(
            channels= channels,
            calc_channels= channels,
            num_heads= num_head,
            dropout_rate= dropout_rate
            )
        
        self.ffn = FFN(
            channels= channels,
            kernel_size= ffn_kernel_size,
            dropout_rate= dropout_rate
            )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())).unsqueeze(1).float()   # float mask

        # Attention + Dropout + LayerNorm
        x = self.attention(x)
        
        # FFN + Dropout + LayerNorm
        x = self.ffn(x, masks)

        return x * masks

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout_rate: float= 0.1,
        ) -> None:
        super().__init__()
        self.conv_0 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'relu'
            )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        self.norm = LayerNorm(
            num_features= channels,
            )
        
    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        residuals = x

        x = self.conv_0(x * masks)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_1(x * masks)
        x = self.dropout(x)
        x = self.norm(x + residuals)

        return x * masks

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/soobinseo/Transformer-TTS/blob/master/network.py
class Duration_Positional_Encoding(torch.nn.Embedding):
    def __init__(
        self,        
        num_embeddings: int,
        embedding_dim: int,
        ):        
        positional_embedding = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        super().__init__(
            num_embeddings= num_embeddings,
            embedding_dim= embedding_dim,
            _weight= positional_embedding
            )
        self.weight.requires_grad = False

        self.alpha = torch.nn.Parameter(
            data= torch.ones(1) * 0.01,
            requires_grad= True
            )

    def forward(self, durations):
        '''
        durations: [Batch, Length]
        '''
        return self.alpha * super().forward(durations)  # [Batch, Dim, Length]

    @torch.jit.script
    def get_pe(x: torch.Tensor, pe: torch.Tensor):
        pe = pe.repeat(1, 1, math.ceil(x.size(2) / pe.size(2))) 
        return pe[:, :, :x.size(2)]

def Mask_Generate(lengths: torch.Tensor, max_length: Union[torch.Tensor, int, None]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]