import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from hubert.modeling_hubert import HubertModel
from torch import Tensor
from wav2vec import Wav2Vec2Model
from utils import PeriodicPositionalEncoding, create_alignment_mask


def adjust_input_representation(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    """
    Brings audio embeddings and visual frames to the same frame rate.

    Args:
        audio_embedding_matrix: The audio embeddings extracted by the audio encoder
        vertex_matrix: The animation sequence represented as a series of vertex positions (or blendshape controls)
        ifps: The input frame rate (it is 50 for the HuBERT encoder)
        ofps: The output frame rate
    """
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    elif ifps > ofps:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
    else:
        factor = 1
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num


# dropout mask for speech conditioning
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class FaceDiffDamm(nn.Module):
    def __init__(self, args):
        super(FaceDiffDamm, self).__init__()
        self.num_blendshapes = args.vertice_dim
        self.latent_dim = args.feature_dim
        self.num_layers = 4
        self.dropout = 0.1

        self.i_fps = 50
        self.o_fps = 30

        # audio encoder
        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()

        frozen_layers = [0,1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # rnn based
        self.rnn = nn.GRU(self.audio_dim * 2 + self.num_blendshapes + self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # final output
        self.out = nn.Linear(self.latent_dim, self.num_blendshapes)
        nn.init.normal_(self.out.weight)
        nn.init.normal_(self.out.bias)

    def forward(self, noised_anim, timesteps, cond_embed, one_hot=None, template=None):
        _, n_frames, _ = noised_anim.shape
        emb = self.embed_timestep(timesteps)
        emb = emb.repeat(n_frames, 1, 1)
        emb = emb.permute(1, 0, 2)

        hidden_states = cond_embed
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        hidden_states, noised_anim, frame_num = adjust_input_representation(
            hidden_states, noised_anim, self.i_fps, self.o_fps
        )
        hidden_states = hidden_states[:, :frame_num]

        input_emb = torch.cat([hidden_states, emb, noised_anim], -1)
        output, _ = self.rnn(input_emb)
        y_pred = self.out(output)
        return y_pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class FaceDiffBeat(nn.Module):
    def __init__(
            self,
            args,
            vertice_dim: int,
            latent_dim: int = 256,
            cond_feature_dim: int = 1536,
            diffusion_steps: int = 1000,
            gru_latent_dim: int = 256,
            num_layers: int = 2,

    ) -> None:

        super().__init__()
        self.i_fps = args.input_fps # audio fps (input to the network)
        self.o_fps = args.output_fps # 4D Scan fps (output or target)
        self.one_hot_timesteps = np.eye(args.diff_steps)

        # Audio Encoder
        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.device = args.device

        frozen_layers = [0,1]

        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        # timestep projection
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )

        self.norm_cond = nn.LayerNorm(cond_feature_dim + vertice_dim + latent_dim)

        # facial decoder
        self.gru = nn.GRU(
            cond_feature_dim + vertice_dim + latent_dim,
            gru_latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)self_attn

    def forward(
            self, x: Tensor,  times: Tensor, cond_embed: Tensor, one_hot=None, template=None
    ):
        batch_size, device = x.shape[0], x.device
        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)

        hidden_states = cond_embed
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        hidden_states, x, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        cond_embed = hidden_states[:, :frame_num]
        x = x[:, :frame_num]

        cond_tokens = cond_embed

        # create the diffusion timestep embedding
        t_tokens = self.time_mlp(times)
        t_tokens = t_tokens.repeat(frame_num, 1, 1)
        t_tokens = t_tokens.permute(1, 0, 2)

        # full conditioning tokens
        full_cond_tokens = torch.cat([cond_tokens, x, t_tokens], dim=-1)
        full_cond_tokens = self.norm_cond(full_cond_tokens)

        output, _ = self.gru(full_cond_tokens)
        output = self.final_layer(output)

        return output


class FaceDiff(nn.Module):
    def __init__(
            self,
            args,
            vertice_dim: int,
            latent_dim: int = 512,
            cond_feature_dim: int = 1536,
            diffusion_steps: int = 500,
            gru_latent_dim: int = 512,
            num_layers: int = 2,


    ) -> None:

        super().__init__()
        self.i_fps = args.input_fps # audio fps (input to the network)
        self.o_fps = args.output_fps # 4D Scan fps (output or target)
        self.one_hot_timesteps = np.eye(args.diff_steps)

        # Audio Encoder
        self.audio_encoder = HubertModel.from_pretrained("huggingface/facebook/hubert-base-ls960")
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.device = args.device

        frozen_layers = [0,1]

        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)

        # noised animation projection
        self.input_projection = nn.Sequential(
            nn.Linear(vertice_dim, latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        # timestep projection
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )
        self.norm_cond = nn.LayerNorm(latent_dim * 4)

        # facial decoder
        self.gru = nn.GRU(latent_dim * 4, gru_latent_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

        # Subject embedding, S
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), latent_dim, bias=False)

        self.emo_embedding = nn.Embedding(num_embeddings=8, embedding_dim=latent_dim)

    def forward(
            self, x: Tensor,  times: Tensor, cond_embed: Tensor, template, one_hot, emo_id
    ):
        batch_size, device = x.shape[0], x.device
        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)

        template = template.unsqueeze(1)
        obj_embedding = self.obj_vector(one_hot)

        # project to latent space
        x = x.permute(1, 0, 2)
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)

        hidden_states = cond_embed
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        hidden_states, x, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        cond_embed = hidden_states[:, :frame_num]
        x = x[:, :frame_num]

        cond_tokens = self.cond_projection(cond_embed)

        # create the diffusion timestep embedding
        t_tokens = self.time_mlp(times)
        t_tokens = t_tokens.repeat(frame_num, 1, 1)
        t_tokens = t_tokens.permute(1, 0, 2)

        emo_embed = self.emo_embedding(emo_id)
        emo_embed = emo_embed.unsqueeze(1).expand(-1, cond_tokens.shape[1], -1)

        # full conditioning tokens
        full_cond_tokens = torch.cat([cond_tokens, x, t_tokens, emo_embed], dim=-1)
        full_cond_tokens = self.norm_cond(full_cond_tokens)

        output, _ = self.gru(full_cond_tokens)
        output = output * obj_embedding
        output = self.final_layer(output)
        output = output + template

        return output

class MyFaceDiff(nn.Module):
    def __init__(self, args, vertice_dim: int, latent_dim: int = 512, cond_feature_dim: int = 1536, diffusion_steps: int = 500, period: int = 25):
        super().__init__()
        self.i_fps = args.input_fps # audio fps (input to the network)
        self.o_fps = args.output_fps # 4D Scan fps (output or target)

        self.audio_encoder = HubertModel.from_pretrained("huggingface/facebook/hubert-base-ls960")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.PPE = PeriodicPositionalEncoding(latent_dim, period = period, max_seq_len=5000)
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(vertice_dim, latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.time_embedding = nn.Embedding(num_embeddings=diffusion_steps, embedding_dim=latent_dim)
        self.emo_embedding = nn.Embedding(num_embeddings=8, embedding_dim=latent_dim)
        # self.feature_map = nn.Linear(latent_dim * 4, latent_dim)
        # self.norm_cond = nn.LayerNorm(latent_dim)
        self.feature_map_tgt = nn.Linear(latent_dim * 4, latent_dim)
        self.norm_cond_tgt = nn.LayerNorm(latent_dim)
        self.feature_map_mem = nn.Linear(latent_dim * 3, latent_dim)
        self.norm_cond_mem = nn.LayerNorm(latent_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=2 * latent_dim,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.obj_vector = nn.Linear(len(args.train_subjects.split()), latent_dim, bias=False)

        self.final_layer = nn.Linear(latent_dim, vertice_dim)
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x: Tensor,  times: Tensor, cond_embed: Tensor, template, one_hot, emo_id):
        batch_size, frame_num, _ = x.shape
        # hidden_states = self.audio_encoder(cond_embed, frame_num=frame_num).last_hidden_state
        hidden_states = self.audio_encoder(cond_embed).last_hidden_state
        hidden_states, x, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        hidden_states = hidden_states[:, :frame_num]

        cond_tokens = self.cond_projection(hidden_states) # bs, len, vert

        # project to latent space
        x = x.permute(1, 0, 2)
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.PPE(x)

        t_tokens = self.time_embedding(times)
        t_tokens = t_tokens.unsqueeze(1).expand(-1, cond_tokens.shape[1], -1)

        emo_embed = self.emo_embedding(emo_id)
        emo_embed = emo_embed.unsqueeze(1).expand(-1, cond_tokens.shape[1], -1)

        # full conditioning tokens
        # full_cond_tokens = torch.cat([x, cond_tokens, t_tokens, emo_embed], dim=-1)
        # full_cond_tokens = self.feature_map(full_cond_tokens)
        # full_cond_tokens = self.norm_cond(full_cond_tokens)
        full_cond_tokens_tgt = torch.cat([x, cond_tokens, t_tokens, emo_embed], dim=-1)
        full_cond_tokens_mem = torch.cat([cond_tokens, t_tokens, emo_embed], dim=-1)
        full_cond_tokens_tgt = self.feature_map_tgt(full_cond_tokens_tgt)
        full_cond_tokens_mem = self.feature_map_mem(full_cond_tokens_mem)
        full_cond_tokens_tgt = self.norm_cond_tgt(full_cond_tokens_tgt)
        full_cond_tokens_mem = self.norm_cond_mem(full_cond_tokens_mem)

        output = self.transformer_decoder(full_cond_tokens_tgt, full_cond_tokens_mem)
        obj_embedding = self.obj_vector(one_hot)
        output = output * obj_embedding
        output = self.final_layer(output)
        template = template.unsqueeze(1)
        output = output + template

        return output

class MyFaceDiff_conv(nn.Module):
    def __init__(self, args, vertice_dim: int = 15069, latent_dim: int = 512, cond_feature_dim: int = 1536, diffusion_steps: int = 500, period: int = 25):
        super().__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.i_fps = args.input_fps # audio fps (input to the network)
        self.o_fps = args.output_fps # 4D Scan fps (output or target)
        self.audio_encoder = HubertModel.from_pretrained("huggingface/facebook/hubert-base-ls960")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)

        self.motion_encoder = nn.Sequential(
            nn.Linear(vertice_dim, latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.one_hot_timesteps = np.eye(args.diff_steps)
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )

        if not args.emo:
            self.hidden_conv = nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, stride=2, padding=1)
            self.conv_1 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding='same')
            self.cond_conv_1 = nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=3, padding='same')
            self.d_conv = nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, stride=2, padding=1)
            self.cond_conv_2 = nn.Conv1d(latent_dim * 4, latent_dim * 2, kernel_size=3, padding='same')
            self.u_conv = nn.ConvTranspose1d(latent_dim * 2, latent_dim, kernel_size=3, stride=2, padding = 1, output_padding=1)
            self.cond_conv_3 = nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=5, padding='same')
            self.conv_2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding='same')
        else:
            # self.emo_embedding = nn.Embedding(num_embeddings=8, embedding_dim=latent_dim)
            # self.hidden_conv = nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, stride=2, padding=1)
            # self.emo_conv = nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, stride=2, padding=1)
            # self.conv_1 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding='same')
            # self.cond_conv_1 = nn.Conv1d(latent_dim * 3, latent_dim, kernel_size=3, padding='same')
            # self.d_conv = nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, stride=2, padding=1)
            # self.cond_conv_2 = nn.Conv1d(latent_dim * 6, latent_dim * 3, kernel_size=3, padding='same')
            # self.u_conv = nn.ConvTranspose1d(latent_dim * 3, latent_dim, kernel_size=3, stride=2, padding = 1, output_padding=1)
            # self.cond_conv_3 = nn.Conv1d(latent_dim * 3, latent_dim, kernel_size=5, padding='same')
            # self.conv_2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding='same')

            self.emo_embedding = nn.Embedding(num_embeddings=8, embedding_dim=latent_dim)
            self.hidden_conv = nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, stride=2, padding=1)
            self.conv_1 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding='same')
            self.cond_conv_1 = nn.Conv1d(latent_dim * 3, latent_dim, kernel_size=3, padding='same')
            self.d_conv = nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, stride=2, padding=1)
            self.cond_conv_2 = nn.Conv1d(latent_dim * 4, latent_dim * 2, kernel_size=3, padding='same')
            self.u_conv = nn.ConvTranspose1d(latent_dim * 2, latent_dim, kernel_size=3, stride=2, padding = 1, output_padding=1)
            self.cond_conv_3 = nn.Conv1d(latent_dim * 3, latent_dim, kernel_size=5, padding='same')
            self.conv_2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding='same')

        self.obj_vector = nn.Linear(len(args.train_subjects.split()), latent_dim, bias=False)

        self.motion_decoder = nn.Linear(latent_dim, vertice_dim)
        nn.init.constant_(self.motion_decoder.weight, 0)
        nn.init.constant_(self.motion_decoder.bias, 0)        

    def forward(self, x: Tensor,  times: Tensor, cond_embed: Tensor, template, one_hot, emo_id):
        batch_size, device = x.shape[0], x.device
        hidden_states = self.audio_encoder(cond_embed).last_hidden_state
        hidden_states, x, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        hidden_states = hidden_states[:, :frame_num]
        hidden_states = self.cond_projection(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).contiguous() # bs, latent_dim, seq_len

        x = x.permute(1, 0, 2)
        x = self.motion_encoder(x)
        x = x.permute(1, 0, 2) # bs, seq_len, latent_dim

        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)
        t_tokens = self.time_mlp(times)
        t_tokens = t_tokens.repeat(frame_num, 1, 1)
        t_tokens = t_tokens.permute(1, 0, 2) # bs, seq_len, latent_dim
        
        if not self.args.emo:
            output = (x + t_tokens).permute(0, 2, 1)
            output = self.conv_1(output)
            output = torch.cat((output, hidden_states), dim=1)
            output = self.cond_conv_1(output)
            output = self.d_conv(output)
            hidden_states_res = self.hidden_conv(hidden_states)
            output = torch.cat((output, hidden_states_res), dim=1)
            output = self.cond_conv_2(output)
            output = self.u_conv(output)
            if output.shape[2] > hidden_states.shape[2]:
                output = output[:, :, :-1]
            output = torch.cat((output, hidden_states), dim=1)
            output = self.cond_conv_3(output)
            output = self.conv_1(output)
            output = output.permute(0, 2, 1)
        else:
            emo_embed = self.emo_embedding(emo_id)
            emo_embed = emo_embed.unsqueeze(1).expand(-1, x.shape[1], -1).permute(0, 2, 1).contiguous() # bs, latent_dim, seq_len

            output = (x + t_tokens).permute(0, 2, 1)
            output = self.conv_1(output)
            output = torch.cat((output, hidden_states, emo_embed), dim=1)
            output = self.cond_conv_1(output)
            output = self.d_conv(output)
            hidden_states_res = self.hidden_conv(hidden_states)
            output = torch.cat((output, hidden_states_res), dim=1)
            output = self.cond_conv_2(output)
            output = self.u_conv(output)
            if output.shape[2] > hidden_states.shape[2]:
                output = output[:, :, :-1]
            output = torch.cat((output, hidden_states, emo_embed), dim=1)
            output = self.cond_conv_3(output)
            output = self.conv_1(output)
            output = output.permute(0, 2, 1)
            
            # output = (x + t_tokens).permute(0, 2, 1)
            # output = self.conv_1(output)
            # output = torch.cat((output, hidden_states, emo_embed), dim=1)
            # output = self.cond_conv_1(output)
            # output = self.d_conv(output)
            # hidden_states_res = self.hidden_conv(hidden_states)
            # emo_res = self.emo_conv(hidden_states)
            # output = torch.cat((output, hidden_states_res, emo_res), dim=1)
            # output = self.cond_conv_2(output)
            # output = self.u_conv(output)
            # if output.shape[2] > hidden_states.shape[2]:
            #     output = output[:, :, :-1]
            # output = torch.cat((output, hidden_states, emo_embed), dim=1)
            # output = self.cond_conv_3(output)
            # output = self.conv_1(output)
            # output = output.permute(0, 2, 1)
        
        obj_embedding = self.obj_vector(one_hot)
        output = self.motion_decoder(output + obj_embedding)

        template = template.unsqueeze(1)
        output = output + template

        return output

class MyFaceDiff_trans_v3(nn.Module):
    def __init__(self, args, vertice_dim: int = 15069, latent_dim: int = 512, cond_feature_dim: int = 1536, diffusion_steps: int = 500, period: int = 30):
        super().__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.i_fps = args.input_fps # audio fps (input to the network)
        self.o_fps = args.output_fps # 4D Scan fps (output or target)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("/mnt-quick/PixelAI/Public/Datasets/Voice/pretrained_models/audio_feature_extractor/facebook_wav2vec2-xlsr-53-espeak-cv-ft")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_proj = nn.Linear(cond_feature_dim, latent_dim)

        self.motion_encoder = nn.Sequential(
            nn.Linear(vertice_dim, latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.PPE = PeriodicPositionalEncoding(latent_dim, period=period, max_seq_len=5000)

        self.one_hot_timesteps = np.eye(args.diff_steps)
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )

        if args.emo:
            self.emo_embedding = nn.Embedding(num_embeddings=8, embedding_dim=latent_dim)
            self.cond_proj = nn.Linear(latent_dim * 4, latent_dim)
        else:
            self.cond_proj = nn.Linear(latent_dim * 2, latent_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=2 * latent_dim,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.obj_vector = nn.Linear(len(args.train_subjects.split()), latent_dim, bias=False)

        self.motion_decoder = nn.Linear(latent_dim, vertice_dim)
        nn.init.constant_(self.motion_decoder.weight, 0)
        nn.init.constant_(self.motion_decoder.bias, 0)        

    def forward(self, x: Tensor,  times: Tensor, cond_embed: Tensor, template, one_hot, emo_id):
        batch_size, frame_num, device = x.shape[0], x.shape[1], x.device
        # hidden_states = self.audio_encoder(cond_embed).last_hidden_state
        # hidden_states, x, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        # hidden_states = hidden_states[:, :frame_num]
        hidden_states = self.audio_encoder(cond_embed, frame_num=frame_num).last_hidden_state
        hidden_states = self.audio_proj(hidden_states) # bs, seq_len, latent_dim

        x = x.permute(1, 0, 2)
        x = self.motion_encoder(x)
        x = x.permute(1, 0, 2) # bs, seq_len, latent_dim
        x = self.PPE(x)

        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)
        t_tokens = self.time_mlp(times)
        # t_tokens = t_tokens.unsqueeze(0).unsqueeze(0) # bs, 1, latent_dim
        t_tokens = t_tokens.repeat(frame_num, 1, 1) 
        t_tokens = t_tokens.permute(1, 0, 2) # bs, seq_len, latent_dim
        
        cond_mask = torch.ones((frame_num, 1)).bool().to(device)
        memory_mask = create_alignment_mask(frame_num, k=3).to(device)
        if self.args.emo:
            emo_embed = self.emo_embedding(emo_id) # bs, latent_dim
            # emo_embed = emo_embed.unsqueeze(0) # bs, 1, latent_dim
            emo_embed = emo_embed.unsqueeze(1).expand(-1, frame_num, -1) # bs, seq_len, latent_dim
            cond = self.cond_proj(torch.cat([x, hidden_states, t_tokens, emo_embed], dim=-1)) 
            # cond = torch.cat([t_tokens, emo_embed, hidden_states], dim=1)
            # memory_mask = torch.cat([cond_mask, cond_mask, memory_mask], dim=1)
        else:
            cond = self.cond_proj(torch.cat([t_tokens, hidden_states], dim=-1))
            # cond = torch.cat([t_tokens, hidden_states], dim=1)
            # memory_mask = torch.cat([cond_mask, memory_mask], dim=1)
        
        output = self.transformer_decoder(tgt=cond, memory=cond)

        obj_embedding = self.obj_vector(one_hot)
        output = self.motion_decoder(output + obj_embedding)

        template = template.unsqueeze(1)
        output = output + template

        return output
