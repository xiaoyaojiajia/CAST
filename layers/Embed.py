import torch
import torch.nn as nn

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch, Channels, Time] -> [Batch, Channels, D_model]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # SOFTS 的实现通常是将时间维度 flatten 后映射，或者直接线性映射
            # 这里对应 SOFTS 源码中的 inverted embedding 逻辑
            x = self.value_embedding(x)
        return self.dropout(x)