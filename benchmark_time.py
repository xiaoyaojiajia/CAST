import torch
import time
import argparse
import numpy as np
from models import CAST

def get_args():
    parser = argparse.ArgumentParser(description='CAST Time Efficiency Benchmark')
    
    # 核心参数 (默认匹配 PeMS04 的配置)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=12)
    
    parser.add_argument('--enc_in', type=int, default=307)
    parser.add_argument('--dec_in', type=int, default=307)
    parser.add_argument('--c_out', type=int, default=307)
    parser.add_argument('--weather_dim', type=int, default=11)
    
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_core', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=256)
    
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--use_norm', type=int, default=1)
    
    # 模拟真实数据集的规模 (PeMS04 大约 1万条训练集，3千条测试集)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_steps_per_epoch', type=int, default=300) # 10000/32 ≈ 300 步
    parser.add_argument('--test_steps', type=int, default=100)            # 3000/32 ≈ 100 步
    
    return parser.parse_args()

def benchmark():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Efficiency Benchmark on [{device}] ===")
    
    model = CAST.Model(args).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # 预先生成在 GPU 上的 Dummy Data (消除硬盘 I/O 干扰)
    dec_seq_len = args.label_len + args.pred_len
    x_enc = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(device)
    x_mark_enc = torch.randn(args.batch_size, args.seq_len, args.weather_dim).to(device)
    x_dec = torch.randn(args.batch_size, dec_seq_len, args.dec_in).to(device)
    x_mark_dec = torch.randn(args.batch_size, dec_seq_len, args.weather_dim).to(device)
    
    # 真实的 Label 张量
    batch_y = torch.randn(args.batch_size, args.pred_len, args.c_out).to(device)

    print("\n[Warming up GPU...]")
    # GPU 预热，防止初始测速不准
    model.train()
    for _ in range(10):
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(outputs, tuple): outputs = outputs[0]
        loss = criterion(outputs[:, -args.pred_len:, :], batch_y)
        loss.backward()

    # ==========================================
    # 1. 测量训练时间 (Training Time: s/epoch)
    # ==========================================
    print("\n[Measuring Training Time (s/epoch)...]")
    model.train()
    
    if device.type == 'cuda': torch.cuda.synchronize()
    train_start_time = time.time()
    
    for _ in range(args.train_steps_per_epoch):
        optimizer.zero_grad()
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(outputs, tuple): outputs = outputs[0]
        loss = criterion(outputs[:, -args.pred_len:, :], batch_y)
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda': torch.cuda.synchronize()
    train_time = time.time() - train_start_time

    # ==========================================
    # 2. 测量测试时间 (Testing Time: s/pass)
    # ==========================================
    print("[Measuring Testing Time (s/pass)...]")
    model.eval()
    
    if device.type == 'cuda': torch.cuda.synchronize()
    test_start_time = time.time()
    
    with torch.no_grad():
        for _ in range(args.test_steps):
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
    if device.type == 'cuda': torch.cuda.synchronize()
    test_time = time.time() - test_start_time

    # ==========================================
    # 输出学术报告标准格式
    # ==========================================
    print("\n" + "="*40)
    print("🎯 EFFICIENCY ANALYSIS RESULTS")
    print("="*40)
    print(f"Dataset Size Simulated: PeMS04 (Batch Size: {args.batch_size})")
    print(f"1 Epoch = {args.train_steps_per_epoch} batches")
    print(f"1 Test Pass = {args.test_steps} batches")
    print("-" * 40)
    print(f"✅ Training Time : {train_time:.2f} s/epoch")
    print(f"✅ Testing Time  : {test_time:.2f} s/pass")
    print("="*40)

if __name__ == '__main__':
    benchmark()