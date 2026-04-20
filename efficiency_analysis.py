import torch
import time
import argparse
import numpy as np

# 尝试导入 thop，如果未安装则提示
try:
    from thop import profile
except ImportError:
    print("错误: 缺少 thop 库。请先在终端运行 'pip install thop'")
    exit()

# 导入您的 CAST 模型
from models import CAST

def get_args():
    parser = argparse.ArgumentParser(description='CAST Efficiency Analysis')
    
    # 核心参数 (默认值已根据您最近的 PEMS04 实验配置填写)
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    parser.add_argument('--enc_in', type=int, default=307, help='encoder input size (num nodes)')
    parser.add_argument('--dec_in', type=int, default=307, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=307, help='output size')
    parser.add_argument('--weather_dim', type=int, default=11, help='dimension of weather features')
    
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_core', type=int, default=64, help='dimension of core')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='use norm and denorm')
    
    # 测试时的 Batch Size
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for inference test')
    
    return parser.parse_args()

def measure_efficiency():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Efficiency Analysis on [{device}] ===")
    
    # 1. 实例化模型
    model = CAST.Model(args).float().to(device)
    model.eval()

    # 2. 构造虚拟输入数据 (Dummy Data)
    # 注意：Decoder 的输入长度在您的代码逻辑中是 label_len + pred_len
    dec_seq_len = args.label_len + args.pred_len
    
    x_enc = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(device)
    x_mark_enc = torch.randn(args.batch_size, args.seq_len, args.weather_dim).to(device)
    x_dec = torch.randn(args.batch_size, dec_seq_len, args.dec_in).to(device)
    x_mark_dec = torch.randn(args.batch_size, dec_seq_len, args.weather_dim).to(device)

    print(f"\n[Configuration]")
    print(f"- Nodes (enc_in): {args.enc_in}")
    # print(f"- Input Shape: {x_enc.shape}")
    print(f"- d_model: {args.d_model}, e_layers: {args.e_layers}")
    
    # ==========================================
    # 阶段 A：计算参数量 (Params) 和计算量 (FLOPs)
    # ==========================================
    # thop 会进行一次前向传播来统计操作数
    flops, params = profile(model, inputs=(x_enc, x_mark_enc, x_dec, x_mark_dec), verbose=False)
    
    print(f"\n[Complexity Metrics]")
    print(f"- Parameters: {params / 1e6:.4f} M (百万)")
    print(f"- FLOPs:      {flops / 1e9:.4f} G (十亿次浮点运算)")

    # ==========================================
    # 阶段 B：计算推理时延 (Inference Time)
    # ==========================================
    # GPU 预热 (Warm-up)：GPU 刚启动时较慢，需预热以获得真实速度
    print(f"\n[Inference Speed] Warming up GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # 正式测量
    iterations = 300
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / iterations * 1000
    
    print(f"- Batch Size: {args.batch_size}")
    print(f"- Average Inference Time: {avg_time_ms:.2f} ms / batch")
    print(f"- FPS (Batches per sec):  {1000 / avg_time_ms:.2f}")
    
    print("\n=== Analysis Complete ===")

if __name__ == '__main__':
    measure_efficiency()