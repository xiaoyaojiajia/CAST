import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer

class ClimateAwareDeformableAligner(nn.Module):
    def __init__(self, c_in, seq_len, n_groups=8, offset_range_factor=10.0, use_weather=True, weather_dim=4):
        super().__init__()
        self.n_groups = n_groups
        self.offset_range_factor = offset_range_factor
        self.seq_len = seq_len
        self.use_weather = use_weather
        
        # ==========================================================
        # [Fix] 修正分组维度的计算逻辑
        # 必须考虑 forward 过程中的 Padding，确保维度对齐
        # ==========================================================
        if c_in % n_groups == 0:
            self.group_dim = c_in // n_groups
        else:
            # 如果不能整除，forward 会补齐到最近的 n_groups 倍数
            # 计算补齐后的总维度
            pad_c = n_groups - (c_in % n_groups)
            self.group_dim = (c_in + pad_c) // n_groups

        # 偏移预测器输入维度
        # Group特征 + 天气/时间特征
        predictor_in_dim = self.group_dim + (weather_dim if use_weather else 0)
        
        self.offset_conv = nn.Sequential(
            nn.Conv1d(predictor_in_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1, bias=False) 
        )

        # 门控残差连接参数
        self.gate = nn.Parameter(torch.zeros(1)) 

    def _get_ref_points(self, L, B, dtype, device):
        ref = torch.linspace(0.5, L - 0.5, L, dtype=dtype, device=device)
        ref = ref.div(L - 1.0).mul(2.0).sub(1.0)
        return ref.view(1, 1, L, 1).expand(B * self.n_groups, -1, -1, -1)

    def forward(self, x, x_ext=None):
        B, L, C = x.size()
        x = x.permute(0, 2, 1) # [B, C, L]
        
        # Padding 处理
        pad_c = 0
        if C % self.n_groups != 0:
            pad_c = self.n_groups - (C % self.n_groups)
            x = F.pad(x, (0, 0, 0, pad_c))
        
        C_padded = x.shape[1]
        x_grouped = x.reshape(B * self.n_groups, -1, L)
        
        # 构建预测器输入
        if self.use_weather and x_ext is not None:
            # x_ext: [B, L, D_w] -> [B, D_w, L]
            x_ext = x_ext.permute(0, 2, 1)
            x_ext_expanded = x_ext.repeat(self.n_groups, 1, 1)
            predictor_input = torch.cat([x_grouped, x_ext_expanded], dim=1)
        else:
            predictor_input = x_grouped

        # 预测偏移量
        offset = self.offset_conv(predictor_input) 
        
        # 稳定性约束
        offset = torch.tanh(offset) * self.offset_range_factor 
        offset_norm = offset * (2.0 / (L - 1.0))
        
        # 变形重采样
        ref_points = self._get_ref_points(L, B, x.dtype, x.device)
        
        # 构造采样 Grid: x = time + offset, y = 0
        grid_x = (ref_points + offset_norm.unsqueeze(-1)).clamp(-1, 1)
        grid_y = torch.zeros_like(grid_x)
        
        # [Fix] 之前的 permute 错误也需要保留修正
        grid = torch.cat([grid_x, grid_y], dim=-1).permute(0, 2, 1, 3) 
        
        x_grouped_expanded = x_grouped.unsqueeze(-1)
        x_sampled = F.grid_sample(
            x_grouped_expanded, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        ).squeeze(-1)
        
        # 门控残差融合
        x_out = x_grouped + torch.tanh(self.gate) * x_sampled
        
        # 还原形状
        x_out = x_out.reshape(B, C_padded, L)
        if pad_c > 0:
            x_out = x_out[:, :C, :] 
            
        x_out = x_out.permute(0, 2, 1) # [B, L, C]
        
        return x_out, offset

# 下面的 STAR 和 Model 类保持不变
class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)
        
        weight = F.softmax(combined_mean, dim=1)
        combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)
        
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        output = self.gen4(combined_mean_cat)
        return output, None

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        
        # 动态设置 Group 数量
        num_groups = min(configs.enc_in, 8) 
        weather_dim = 4 # 假设时间特征维度为4
        
        self.aligner = ClimateAwareDeformableAligner(
            c_in=configs.enc_in, 
            seq_len=configs.seq_len,
            n_groups=num_groups,
            offset_range_factor=10.0,
            use_weather=True,
            weather_dim=weather_dim
        )
        
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Align-then-Aggregate
        x_enc, offsets = self.aligner(x_enc, x_ext=x_mark_enc)

        _, _, N = x_enc.shape 
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
        return dec_out, offsets

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, offsets = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], offsets