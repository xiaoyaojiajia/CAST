import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer

# =============================================================================
# 1. 气候感知变形对齐器 (稳健修复版)
# =============================================================================
class ClimateAwareDeformableAligner(nn.Module):
    def __init__(self, c_in, seq_len, n_groups=8, offset_range_factor=10.0, use_weather=True, weather_dim=4):
        super().__init__()
        self.c_in = c_in        # 保存标准节点数 (307)
        self.seq_len = seq_len  # 保存标准序列长 (96)
        self.n_groups = n_groups
        self.offset_range_factor = offset_range_factor
        self.use_weather = use_weather
        
        # 1. 计算分组
        if c_in % n_groups == 0:
            self.group_dim = c_in // n_groups
        else:
            pad_c = n_groups - (c_in % n_groups)
            self.group_dim = (c_in + pad_c) // n_groups

        # 2. 偏移预测器
        predictor_in_dim = self.group_dim + (weather_dim if use_weather else 0)
        
        self.offset_conv = nn.Sequential(
            nn.Conv1d(predictor_in_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1, bias=False) 
        )

        # 3. 天气数值注入器
        if self.use_weather:
            self.weather_injector = nn.Sequential(
                nn.Conv1d(weather_dim, self.group_dim * n_groups, kernel_size=1), # 注意：这里映射到 Padding 后的维度
                nn.Sigmoid() 
            )
            self.weather_val_proj = nn.Conv1d(weather_dim, self.group_dim * n_groups, kernel_size=1)

        self.gate = nn.Parameter(torch.zeros(1)) 
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.gate, 0)
        if self.use_weather:
            nn.init.constant_(self.weather_injector[0].weight, 0)
            nn.init.constant_(self.weather_injector[0].bias, -5.0) 
            nn.init.constant_(self.weather_val_proj.weight, 0)
            nn.init.constant_(self.weather_val_proj.bias, 0)
            
    def _get_ref_points(self, L, B, dtype, device):
        ref = torch.linspace(0.5, L - 0.5, L, dtype=dtype, device=device)
        ref = ref.div(L - 1.0).mul(2.0).sub(1.0)
        return ref.view(1, 1, L, 1).expand(B * self.n_groups, -1, -1, -1)

    def forward(self, x, x_ext=None):
        # [核心修复]: 强制维度检查，确保转换为 [B, C, L]
        B, D1, D2 = x.shape
        
        # 判断逻辑: 如果最后一个维度等于 c_in (307), 说明是 [B, L, C] -> 需要 Permute
        if D2 == self.c_in: 
            x = x.permute(0, 2, 1) # -> [B, C, L]
        # 否则假设已经是 [B, C, L]
        
        # 获取当前的 C 和 L
        C_curr = x.shape[1]
        L_curr = x.shape[2]
        
        # --- Padding 逻辑 ---
        pad_c = 0
        target_c = self.n_groups * self.group_dim
        if C_curr < target_c:
            pad_c = target_c - C_curr
            x_pad = F.pad(x, (0, 0, 0, pad_c))
        else:
            x_pad = x
            
        # Reshape: [B, Groups, Group_Dim, L]
        x_grouped = x_pad.reshape(B, self.n_groups, self.group_dim, L_curr)
        # Flatten for Conv1d: [B * Groups, Group_Dim, L]
        x_grouped_flat = x_grouped.reshape(B * self.n_groups, self.group_dim, L_curr)
        
        # --- 准备天气输入 ---
        if self.use_weather and x_ext is not None:
            # x_ext: [B, L, D] -> [B, D, L]
            if x_ext.shape[-1] == self.seq_len: # 如果已经是 [B, D, L]
                 pass
            else:
                 x_ext = x_ext.permute(0, 2, 1) # [B, L, D] -> [B, D, L]

            # 扩展天气特征: [B, D, L] -> [B * Groups, D, L]
            x_ext_expanded = x_ext.repeat_interleave(self.n_groups, dim=0)
            
            predictor_input = torch.cat([x_grouped_flat, x_ext_expanded], dim=1)
        else:
            predictor_input = x_grouped_flat
            x_ext = None

        # --- A. Offset ---
        offset = self.offset_conv(predictor_input) # [B*Groups, 1, L]
        offset = torch.tanh(offset) * self.offset_range_factor 
        
        # (采样逻辑)
        offset_norm = offset * (2.0 / (L_curr - 1.0))
        ref_points = self._get_ref_points(L_curr, B, x.dtype, x.device)
        grid_x = (ref_points + offset_norm.unsqueeze(-1)).clamp(-1, 1)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.cat([grid_x, grid_y], dim=-1).permute(0, 2, 1, 3) 
        
        x_grouped_expanded = x_grouped_flat.unsqueeze(-1)
        x_sampled = F.grid_sample(x_grouped_expanded, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(-1)
        
        # 还原
        x_out = x_grouped_flat + torch.tanh(self.gate) * x_sampled
        
        # --- B. 数值注入 (Injection) ---
        if self.use_weather and x_ext is not None:
            # Injector 输出的是 [B, C_padded, L] -> 需要 reshape 成 [B*G, G_dim, L]
            weather_gate = self.weather_injector(x_ext) # [B, C_padded, L]
            weather_value = self.weather_val_proj(x_ext)
            
            # 变形以匹配 x_out
            w_gate_flat = weather_gate.reshape(B * self.n_groups, self.group_dim, L_curr)
            w_val_flat = weather_value.reshape(B * self.n_groups, self.group_dim, L_curr)
            
            x_out = x_out + w_gate_flat * w_val_flat

        # 还原回 [B, C, L]
        x_out = x_out.reshape(B, target_c, L_curr)
        if pad_c > 0:
            x_out = x_out[:, :C_curr, :] 
            
        # 最终输出转回 [B, L, C] (因为后面 Embedding 需要 L 在中间)
        x_out = x_out.permute(0, 2, 1) 
        
        return x_out, offset.reshape(B, self.n_groups, L_curr)

# =============================================================================
# 2. SOFTS 核心模块: STAR
# =============================================================================
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

# =============================================================================
# 3. 主模型: CAST
# =============================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        
        num_groups = min(configs.enc_in, 8) 
        weather_dim = getattr(configs, 'weather_dim', 4)
            
        self.aligner = ClimateAwareDeformableAligner(
            c_in=configs.enc_in, 
            seq_len=configs.seq_len,
            n_groups=num_groups,
            offset_range_factor=10.0,
            use_weather=True,
            weather_dim=weather_dim 
        )
        
        # [修复] DataEmbedding_inverted 只传3个参数
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, 
            configs.d_model, 
            configs.dropout
        )
        
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

        # 调用 Align 对齐模块
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