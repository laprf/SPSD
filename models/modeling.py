import copy
import logging
import math
import os

import ml_collections
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from fightingcv_attention.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention as SSA
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from models.head import Decoder
from models.modeling_resnet import ResNetV2
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
}


class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.channel = channel
        self.conv_1 = nn.Conv2d(self.channel, self.channel, 1, padding=0)
        self.conv_2 = nn.Conv2d(self.channel, self.channel, 1, padding=0)
        self.bn = nn.BatchNorm2d(self.channel)

    def forward(self, feat_1, feat_2):
        feat_1 = self.bn(self.conv_1(feat_1))
        feat_2 = self.bn(self.conv_1(feat_2))
        return feat_1 + feat_2


class RefineMechanism(nn.Module):
    def __init__(self, config):
        super(RefineMechanism, self).__init__()
        n_feat = int(config.hidden_size)

        self.spatial_wise = SSA(d_model=n_feat, h=4)
        self.out_conv = nn.Conv2d(n_feat, 1, 1, 1, 0)

    def forward(self, refine):
        refine_hw = rearrange(refine, 'b c h w -> b (h w) c')
        spat = self.spatial_wise(refine_hw, refine_hw, refine_hw)
        spat = rearrange(spat, 'b (h w) c -> b c h w', h=22)
        refine_feat = spat + refine
        return refine_feat, torch.sigmoid(self.out_conv(refine_feat))


class EFGM(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, size=88):
        super(EFGM, self).__init__()
        self.size = size
        self.edge_conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.edge_out = nn.Conv2d(out_dim, 1, 3, 1, 1)
        self.edge_mask = nn.Sequential(
            nn.AdaptiveAvgPool2d(size),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid(),
        )
        self.conv_edge_enhance = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, stride=1, padding=1
        )
        self.syncbn_edge = torch.nn.BatchNorm2d(out_dim)

    def forward(self, res_feat):
        res_feat = F.interpolate(res_feat, size=self.size, mode="bilinear", align_corners=False)
        edge_feat = self.edge_conv(res_feat)
        edge_feat_mask = self.edge_mask(edge_feat)
        edge_feat = torch.mul(
            edge_feat, edge_feat_mask
        ) + self.conv_edge_enhance(
            edge_feat
        )
        edge_feat = self.syncbn_edge(edge_feat)
        edge_feat = F.relu(edge_feat, inplace=False)

        edge_out = self.edge_out(edge_feat)
        edge_out = F.interpolate(
            edge_out, size=352, mode="bilinear", align_corners=False
        )
        return edge_feat, edge_out


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)  #

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, refine=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        if refine is not None:
            mixed_value_layer = self.value(hidden_states) * rearrange(refine, 'b c h w -> b (h w) c')
        else:
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):  # positon encoding and position encoding
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.patch_dim = 16
        self.flatten_dim = 768

        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                in_channels=in_channels,
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )

        self.dropout = Dropout(p=0.1)

    def forward(self, x):
        B = x.shape[0]
        if self.hybrid:
            res_feat, x = self.hybrid_model(x)

        x = self.patch_embeddings(x)  # [b, h, w, c]
        x = x.flatten(2)  # [b, h*w, c]
        x = x.transpose(-1, -2)  # [b, c, h*w]

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, res_feat


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x, refine):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x, refine)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):  # num_layers = 12
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, refine):
        for i, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states, refine)

        encoded = self.encoder_norm(hidden_states)
        return encoded


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=352, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.HSI_embed = Embeddings(config, img_size, in_channels=50)
        self.spec_embed = Embeddings(config, img_size, in_channels=3)
        self.HSI_edge = EFGM(in_dim=256, out_dim=256, size=88)
        self.spec_edge = EFGM(in_dim=256, out_dim=256, size=88)
        self.fuse = FFM(channel=256)
        self.edge_fuse = FFM(channel=256)
        self.guide = RefineMechanism(config)
        self.gate = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.attn = Encoder(config, vis)
        self.decoder = Decoder(embed_dim=256, hidden_size=256)

    def forward(self, HSI_input, spec_sal_input):
        vis = []

        HSI_embed, HSI_res_feat = self.HSI_embed(HSI_input)
        spec_embed, spec_res_feat = self.spec_embed(spec_sal_input)

        edge_feat, edge_out = self.HSI_edge(HSI_res_feat[0])

        HSI_embed, spec_embed = map(lambda t: rearrange(t, 'b (h w) n -> b n h w', h=22), (HSI_embed, spec_embed))
        refine_feat, refine_map = self.guide(spec_embed)
        gate = self.gate(refine_feat)
        vis.append(refine_feat)
        vis.append(gate)

        decoder_input = self.attn(rearrange(HSI_embed, 'b n h w -> b (h w) n'), gate)
        decoder_output = self.decoder(decoder_input, edge_feat)
        return decoder_output, edge_out, refine_map, vis

    def load_from(self, weights):
        with torch.no_grad():
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.HSI_embed.position_embeddings
            ntok_new = posemb_new.size(1)

            posemb_grid = posemb[0, 1:, 0:769:3]

            gs_old = int(np.sqrt(len(posemb_grid)))
            gs_new = int(np.sqrt(ntok_new))
            posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

            zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
            posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            self.HSI_embed.position_embeddings.copy_(np2th(posemb_grid))
            for bname, block in self.HSI_embed.hybrid_model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)

            self.spec_embed.hybrid_model.root.conv.weight.copy_(
                np2th(weights["conv_root/kernel"], conv=True)
            )
            gn_weight = np2th(weights["gn_root/scale"]).view(-1)
            gn_bias = np2th(weights["gn_root/bias"]).view(-1)
            self.spec_embed.hybrid_model.root.gn.weight.copy_(
                gn_weight
            )
            self.spec_embed.hybrid_model.root.gn.bias.copy_(
                gn_bias
            )
            self.spec_embed.position_embeddings.copy_(np2th(posemb_grid))
            for bname, block in self.spec_embed.hybrid_model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)


def get_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"grid": (14, 14)})
    config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = config.hidden_size * 4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.img_size = 352
    config.pretrained_dir = os.environ['HOME'] + "/my_PSOD_large/imagenet21k2012_R50+ViT-B_16.npz"
    return config
