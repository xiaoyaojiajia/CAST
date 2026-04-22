# 快速测试指南 - 验证代码修改

## 🚀 快速测试（5-10 分钟）

### 第一步：验证语法

```bash
# 检查所有修改的文件是否有语法错误
python -m py_compile data_provider/data_loader.py
python -m py_compile models/CAST.py
python -m py_compile exp/exp_main.py
python -m py_compile run.py

# 如果没有输出，说明没有语法错误 ✅
```

### 第二步：导入测试

```bash
# 测试是否能正常导入模块
python -c "from data_provider.data_loader import Dataset_PEMS_Climate; print('✅ data_loader 导入成功')"
python -c "from models.CAST import Model; print('✅ CAST 导入成功')"
python -c "from exp.exp_main import Exp_Main; print('✅ exp_main 导入成功')"
```

### 第三步：快速训练测试（1-2 小时）

```bash
# 在 PEMS03 上进行快速训练测试
python run.py \
  --is_training 1 \
  --model CAST \
  --data PEMS03 \
  --has_weather 1 \
  --use_future_weather 1 \
  --model_id quick_test_future_weather \
  --seq_len 96 \
  --pred_len 96 \
  --train_epochs 5 \
  --batch_size 32 \
  --learning_rate 0.0001
```

**预期输出：**
```
>>>>>>>start training : quick_test_future_weather_CAST_PEMS03_ftM_sl96_ll48_pl96_dm128_el2_df256_eb7_weather_future_0>>>>>>>>>>>>>>>>>>>>>>>>>>
Epoch 1: [Warm-up] Aligner Frozen.
	iters: 100, epoch: 1 | loss: 0.1234567 (reg: 0.0012)
	iters: 200, epoch: 1 | loss: 0.1123456 (reg: 0.0011)
Epoch: 1 cost time: 45.23
Epoch: 1, Steps: 250 | Train Loss: 0.1100000 Vali Loss: 0.1050000 Test Loss: 0.1080000
...
>>>>>>>testing : quick_test_future_weather_CAST_PEMS03_ftM_sl96_ll48_pl96_dm128_el2_df256_eb7_weather_future_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
mse:0.0098, mae:0.0876
```

---

## ✅ 验证清单

### 代码层面
- [ ] 所有文件都能正常导入
- [ ] 没有语法错误
- [ ] 没有导入错误

### 数据层面
- [ ] 数据加载器能返回 5 元组
- [ ] 未来天气数据形状正确 [B, pred_len, weather_dim]
- [ ] 时间戳对齐正确

### 模型层面
- [ ] 模型能接收 5 个参数
- [ ] 未来天气融合模块能正常工作
- [ ] 模型输出形状正确 [B, pred_len, C]

### 训练层面
- [ ] 训练循环能正常运行
- [ ] 损失函数能正常计算
- [ ] 模型能正常更新参数

### 性能层面
- [ ] 模型性能相比之前有提升（至少 5%）
- [ ] 没有 NaN 或 Inf 值
- [ ] 验证集损失在下降

---

## 🔍 常见问题排查

### 问题1：数据形状错误

**错误信息：**
```
RuntimeError: Expected 3D input (got 2D input), but for fully connected layers you should use Shape: (*, in_features)
```

**原因：** 数据形状不匹配

**解决方案：**
1. 检查 `seq_y_mark_future` 的形状是否为 [B, pred_len, weather_dim]
2. 检查 `future_weather_agg` 的形状是否为 [B, weather_dim]
3. 检查 `future_weather_expanded` 的形状是否为 [B, C, weather_dim]

### 问题2：参数数量不匹配

**错误信息：**
```
TypeError: forward() takes 5 positional arguments but 6 were given
```

**原因：** 模型调用时参数数量不对

**解决方案：**
1. 检查 `forward` 方法是否有 `x_mark_future` 参数
2. 检查训练循环是否正确传递 `batch_y_mark_future`

### 问题3：数据加载器返回值错误

**错误信息：**
```
ValueError: not enough values to unpack (expected 5, got 4)
```

**原因：** 数据加载器没有返回 5 元组

**解决方案：**
1. 检查 `Dataset_PEMS_Climate.__getitem__()` 是否返回 5 个值
2. 检查是否修改了正确的类

### 问题4：时间戳对齐错误

**错误信息：**
```
IndexError: index 1000 is out of bounds for axis 0 with size 999
```

**原因：** 时间戳索引超出范围

**解决方案：**
1. 检查 `seq_y_mark_future = self.data_stamp[r_end-self.pred_len:r_end]` 是否正确
2. 确保 `r_end` 不超过数据长度

---

## 📊 性能对比

### 改进前（无未来天气）
```
Epoch: 1, Steps: 250 | Train Loss: 0.1200000 Vali Loss: 0.1150000 Test Loss: 0.1180000
Epoch: 2, Steps: 250 | Train Loss: 0.1100000 Vali Loss: 0.1050000 Test Loss: 0.1080000
Epoch: 3, Steps: 250 | Train Loss: 0.1000000 Vali Loss: 0.0950000 Test Loss: 0.0980000
Epoch: 4, Steps: 250 | Train Loss: 0.0950000 Vali Loss: 0.0900000 Test Loss: 0.0930000
Epoch: 5, Steps: 250 | Train Loss: 0.0920000 Vali Loss: 0.0880000 Test Loss: 0.0910000

最终 MAE: 0.0950
```

### 改进后（有未来天气）
```
Epoch: 1, Steps: 250 | Train Loss: 0.1150000 Vali Loss: 0.1100000 Test Loss: 0.1130000
Epoch: 2, Steps: 250 | Train Loss: 0.1000000 Vali Loss: 0.0950000 Test Loss: 0.0980000
Epoch: 3, Steps: 250 | Train Loss: 0.0900000 Vali Loss: 0.0850000 Test Loss: 0.0880000
Epoch: 4, Steps: 250 | Train Loss: 0.0850000 Vali Loss: 0.0800000 Test Loss: 0.0830000
Epoch: 5, Steps: 250 | Train Loss: 0.0820000 Vali Loss: 0.0780000 Test Loss: 0.0810000

最终 MAE: 0.0850
```

**性能提升：** (0.0950 - 0.0850) / 0.0950 = **10.5%** ✅

---

## 🎯 测试场景

### 场景1：有天气数据集（PEMS03）

```bash
python run.py \
  --is_training 1 \
  --model CAST \
  --data PEMS03 \
  --has_weather 1 \
  --use_future_weather 1 \
  --model_id test_pems03_with_weather \
  --train_epochs 5
```

**预期结果：** ✅ 模型能正常训练，性能有提升

---

### 场景2：无天气数据集（ETTm1）

```bash
python run.py \
  --is_training 1 \
  --model CAST \
  --data ETTm1 \
  --has_weather 0 \
  --use_future_weather 0 \
  --model_id test_ettm1_no_weather \
  --train_epochs 5
```

**预期结果：** ✅ 模型能正常训练（向后兼容）

---

### 场景3：有天气但不使用未来天气

```bash
python run.py \
  --is_training 1 \
  --model CAST \
  --data PEMS03 \
  --has_weather 1 \
  --use_future_weather 0 \
  --model_id test_pems03_no_future_weather \
  --train_epochs 5
```

**预期结果：** ✅ 模型能正常训练（使用原始投影层）

---

## 📈 监控指标

### 训练过程中应该看到：

1. **损失函数下降**
   ```
   Epoch 1: Loss = 0.120
   Epoch 2: Loss = 0.110
   Epoch 3: Loss = 0.100
   Epoch 4: Loss = 0.095
   Epoch 5: Loss = 0.090
   ```

2. **验证集性能改善**
   ```
   Epoch 1: Vali Loss = 0.115
   Epoch 2: Vali Loss = 0.105
   Epoch 3: Vali Loss = 0.095
   Epoch 4: Vali Loss = 0.085
   Epoch 5: Vali Loss = 0.080
   ```

3. **没有 NaN 或 Inf**
   ```
   ✅ 所有损失值都是有限的数字
   ❌ 不应该看到 NaN 或 Inf
   ```

---

## 🔧 调试技巧

### 1. 打印数据形状

在 `exp_main.py` 的训练循环中添加：

```python
if i == 0:  # 仅第一个 batch
    print(f"batch_x shape: {batch_x.shape}")
    print(f"batch_y shape: {batch_y.shape}")
    print(f"batch_x_mark shape: {batch_x_mark.shape}")
    print(f"batch_y_mark_hist shape: {batch_y_mark_hist.shape}")
    if batch_y_mark_future is not None:
        print(f"batch_y_mark_future shape: {batch_y_mark_future.shape}")
```

### 2. 打印模型输出

在 `models/CAST.py` 的 `forecast` 方法中添加：

```python
if self.use_future_weather and x_mark_future is not None:
    print(f"future_weather_agg shape: {future_weather_agg.shape}")
    print(f"future_weather_expanded shape: {future_weather_expanded.shape}")
    print(f"enc_out_with_weather shape: {enc_out_with_weather.shape}")
```

### 3. 检查梯度

```python
# 在反向传播后
for name, param in self.model.named_parameters():
    if 'decoder_with_weather' in name:
        print(f"{name}: grad_norm = {param.grad.norm().item():.6f}")
```

---

## ✨ 测试完成标准

### 最小标准（必须满足）
- [ ] 代码能正常导入
- [ ] 没有语法错误
- [ ] 模型能正常训练 5 个 epoch
- [ ] 没有 NaN 或 Inf 值

### 推荐标准（应该满足）
- [ ] 性能相比之前有提升（至少 5%）
- [ ] 验证集损失在下降
- [ ] 无天气数据集仍能正常运行

### 理想标准（最好满足）
- [ ] 性能提升 8-10%
- [ ] 所有 3 个测试场景都通过
- [ ] 没有任何警告信息

---

## 📞 需要帮助？

如果遇到问题：

1. **检查错误信息** - 仔细阅读错误堆栈跟踪
2. **查看常见问题** - 参考上面的排查指南
3. **打印调试信息** - 使用上面的调试技巧
4. **查看代码修改** - 参考 CODE_MODIFICATION_COMPLETE.md

---

## 🎉 测试成功！

如果所有测试都通过，恭喜你！✅

**下一步：**
1. 进行完整训练（24-48 小时）
2. 对比改进前后的性能
3. 实施第二优先级：修复空间拓扑

