# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from utils.atom_feature import AtomFeatureEncoder
from utils.relative_features import compute_relative_features
from utils.rbf_encoding import RBFEncoding
from e3nn.o3 import spherical_harmonics 

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, kernel_size=3, padding=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 添加卷积层
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,  # 修改为更小的卷积核大小
                    padding=padding
                )
            )

            # 非最后一层添加GELU和Dropout
            if i < num_layers - 1:
                self.layers.append(nn.ReLU())    # 替换ReLU
                #self.layers.append(nn.Dropout(0.1))  # 新增Dropout

    def forward(self, x):
        # 输入形状: [B, D, L]
        for layer in self.layers:
            x = layer(x)
        return x  # 输出形状: [B, output_dim, L]


class Transformer(nn.Module):

    def __init__(self, token_num=100, d_model=512, nhead=8, dos_num=40, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False):
        super().__init__()
        # Atom type embedding
        self.tok_emb = nn.Embedding(token_num, d_model)
        # Numeric atomic feature embedding
        self.num_emb_encoder = AtomFeatureEncoder(input_dim=3, out_dim=d_model)
        # LayerNorms for matching distributions
        self.atom_norm = nn.LayerNorm(d_model)
        self.num_norm  = nn.LayerNorm(d_model)
        # Fusion projection
        self.fuse_proj = nn.Linear(d_model * 2, d_model)

        # RBF & SH encoders remain
        self.rbf_encoder = RBFEncoding(num_centers=64, cutoff=10.0)
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before,
            rbf_encoder=self.rbf_encoder
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.query_embed = nn.Parameter(torch.zeros(dos_num, d_model))
        self.tgt = nn.Parameter(torch.zeros(dos_num, d_model))
        self.out_embed = CNN(d_model, d_model*3, output_dim=1, num_layers=6)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, src, mask, pos):
        # src: [B, L] atom indices; pos carries lattice+coords
        B, Lp, _ = pos.shape
        atom_len = Lp - 2
        mask = mask[:, :atom_len]

        # Extract atom indices and numeric features
        atom_idx = src[:, 2:]  # [B, L]
        atom_emb = self.tok_emb(atom_idx)         # [B, L, d_model]
        num_emb  = self.num_emb_encoder(atom_idx) # [B, L, d_model]

        # Normalize each stream
        atom_emb = self.atom_norm(atom_emb)
        num_emb  = self.num_norm(num_emb)

        # Fuse into unified embedding
        fused = torch.cat([atom_emb, num_emb], dim=-1)  # [B, L, 2*d_model]
        atom_src = self.fuse_proj(fused)                # [B, L, d_model]

        # Compute relative geometry features
        distances, unit_dirs = compute_relative_features(pos)
        rel_feat = self.rbf_encoder(distances)

        # Encoder
        memory = self.encoder(
            src=atom_src,
            src_key_padding_mask=mask,
            pos=pos,
            rel_features=rel_feat,
            rel_dirs=unit_dirs
        )

        # Decoder
        query_embed = self.query_embed.unsqueeze(0).repeat(B, 1, 1)
        tgt = self.tgt.unsqueeze(0).repeat(B, 1, 1)
        hs, attn = self.decoder(
            tgt, memory,
            memory_key_padding_mask=mask,
            pos=pos,
            query_pos=query_embed
        )

        # Output
        hs = hs.permute(0, 2, 1)
        res = self.out_embed(hs)
        res = res.permute(0, 2, 1)
        return res, attn

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            rel_features=None,
            rel_dirs=None):
        
        output = src
    
        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos,
                           rel_features=rel_features,
                           rel_dirs=rel_dirs)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output, attention = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="leaky_relu", normalize_before=False, rbf_encoder=None):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        self.dim = d_model
        self.nhead = nhead
        
        # 标准多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 用于前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        
        # 保存 RBF 编码器，用于获取相对特征的维度信息
        self.rbf_encoder = rbf_encoder
        
        # 新增：将相对特征映射到与查询向量相同的维度（或者一个合理的中间维度），
        # 然后再根据头数分解，用于修正注意力打分
        self.rel_proj = nn.Linear(self.rbf_encoder.num_centers, d_model)  
        
        self.max_ell = 3  # 球谐函数最大阶数，自己调整
        dim_sph = sum([2 * l + 1 for l in range(self.max_ell + 1)])  # 球谐展开维度
        
        self.dir_proj = nn.Linear(dim_sph, d_model)
        
    def forward(self, src, src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None,
                     rel_features=None,
                     rel_dirs=None):
        """
        参数:
            src: [B, L, d_model]
            rel_features: [B, L, L, num_centers]，预计算的 RBF 编码
        """
        B, L, _ = src.size()
        
        # 计算查询、键、值
        q, k, v = src, src, src
        
        # 标准多头注意力的点积部分
        attn_output, attn_weights = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

        # 将相对特征映射到 d_model 维度，再 reshape 为 [B, L, L, nhead, d_model_head]
        # 其中 d_model_head = d_model // nhead
        rel_emb = self.rel_proj(rel_features)  # [B, L, L, d_model]
        d_model_head = self.dim // self.nhead
        rel_emb = rel_emb.view(B, L, L, self.nhead, d_model_head)
        
        # 计算球面谐波，l从0到max_ell
        sph_emb = spherical_harmonics(list(range(self.max_ell + 1)), rel_dirs, normalize=True)
        # sph_emb dtype是float，实部球谐，形状跟 rel_dirs 对应
        
        # 投影到 d_model 维度
        dir_emb = self.dir_proj(sph_emb)  # [B, L, L, d_model]

        dir_emb = dir_emb.view(B, L, L, self.nhead, d_model_head)
        
        # 将查询拆分为多个头：[B, L, nhead, d_model_head]
        q_heads = q.view(B, L, self.nhead, d_model_head)
        
        # 计算额外的相对打分：点积 q_heads 和 rel_emb 的最后一维
        # 得到 [B, L, L, nhead]
        rel_scores = (q_heads.unsqueeze(2) * rel_emb).sum(-1)
        dir_scores = (q_heads.unsqueeze(2) * dir_emb).sum(-1)        # [B,L,L,nhead]
        
        # 将 rel_scores 转换为形状 [B * nhead, L, L]，与标准注意力分数相加
        rel_scores = rel_scores.permute(0, 3, 1, 2).reshape(B * self.nhead, L, L)
        dir_scores = dir_scores.permute(0, 3, 1, 2).reshape(B*self.nhead, L, L)

        # 重新计算标准注意力打分（这里假设已经进行了缩放处理），并加上 rel_scores
        # 这里我们直接模拟一个加法操作，实际中可能需要调整比例因子
        # 获取标准自注意力的打分（未经过 softmax 的部分）：
        q_scaled = q / (d_model_head ** 0.5)
        q_heads2 = q_scaled.view(B, L, self.nhead, d_model_head).permute(0, 2, 1, 3).reshape(B * self.nhead, L, d_model_head)
        k_heads = k.view(B, L, self.nhead, d_model_head).permute(0, 2, 1, 3).reshape(B * self.nhead, L, d_model_head)
        base_scores = torch.bmm(q_heads2, k_heads.transpose(1, 2))
        
        # 新的总打分
        total_scores = base_scores + rel_scores + dir_scores
        # 计算新的注意力权重并输出
        attn_weights_new = F.softmax(total_scores, dim=-1)
        
        # 重新计算注意力输出：将新的权重作用于 v（同样需要拆分成头）
        v_heads = v.view(B, L, self.nhead, d_model_head).permute(0, 2, 1, 3).reshape(B * self.nhead, L, d_model_head)
        attn_output_new = torch.bmm(attn_weights_new, v_heads)
        attn_output_new = attn_output_new.view(B, self.nhead, L, d_model_head).permute(0, 2, 1, 3).reshape(B, L, self.dim)
        
        # 使用新的注意力输出
        src2 = attn_output_new

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = tgt
        k = tgt
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        q = tgt
        k = memory
        tgt2, attention_v = self.multihead_attn(query=q, key=memory, value=memory,
                                                attn_mask=memory_mask,
                                                key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention_v

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)  # 添加 LeakyReLU 支持
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
