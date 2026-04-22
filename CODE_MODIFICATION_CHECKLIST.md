# 代码修改清单 - 逐文件详细指南

## 📁 需要修改的文件列表

### 1️⃣ 数据加载层 (data_provider/data_loader.py)

**修改内容：** Dataset_PEMS_Climate 类

**当前问题：**
```python
# 第 678 行附近
def __getitem__(self, index):
    # ...
    seq_y_mark = self.data_stamp[r_begin:r_end]  # ❌ 这是历史天气，不是未来天气！
    return seq_x, seq_y, seq_x_mark, seq_y_mark
```

**修改步骤：**

1. **修改 `__getitem__` 方法返回值**
   - 位置：第 678-690 行
   - 改为返回 5 个值：`seq_x, seq_y, seq_x_mark, seq_y_mark_history, seq_y_mark_future`
   - `seq_y_mark_future` 应该是 `self.data_stamp[r_end-self.pred_len:r_end]`

2. **修改 `__init__` 方法**
   - 位置：第 624-638 行
   - 添加注释说明数据结构

**代码示例：**
```python
def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    
    # 历史天气：对应输入序列
    seq_x_mark = self.data_stamp[s_begin:s_end]
    
    # 历史天气标记（用于解码器初始化）
    seq_y_mark_history = self.data_stamp[r_begin:r_begin+self.label_len]
    
    # ✅ 未来天气：对应预测时间窗口
    seq_y_mark_future = self.data_stamp[r_end-self.pred_len:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark_history, seq_y_mark_future
```

---

### 2️⃣ 数据工厂 (data_provider/data_factory.py)

**修改内容：** 处理新的 5 元组返回值

**当前问题：**
```python
# DataLoader 期望 4 元组，但现在返回 5 元组
```

**修改步骤：**

1. **无需修改 data_factory.py 本身**
   - DataLoader 会自动处理任意数量的返回值
   - 但需要在 exp_main.py 中处理新的返回值

---

### 3️⃣ 模型架构 (models/CAST.py)

**修改内容：** 添加未来天气融合模块

**当前问题：**
```python
# 第 95-110 行
def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # ... 编码器处理 ...
    dec_out = self.projection(enc_out)  # ❌ 没有使用 x_mark_dec
```

**修改步骤：**

1. **在 `__init__` 中添加新模块**
   - 位置：第 80-95 行（在 `self.projection` 之后）
   - 添加：
     ```python
     # 未来天气融合模块
     weather_dim = getattr(configs, 'weather_dim', 4)
     self.future_weather_fusion = nn.Sequential(
         nn.Linear(weather_dim, configs.d_model),
         nn.GELU(),
         nn.Linear(configs.d_model, configs.d_model)
     )
     
     # 考虑未来天气的解码器
     self.decoder_with_weather = nn.Sequential(
         nn.Linear(configs.d_model + weather_dim, configs.d_model),
         nn.GELU(),
         nn.Linear(configs.d_model, configs.pred_len)
     )
     
     # 标记是否使用未来天气
     self.use_future_weather = getattr(configs, 'use_future_weather', False)
     ```

2. **修改 `forecast` 方法签名**
   - 位置：第 97 行
   - 改为：`def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_mark_future=None):`

3. **修改 `forecast` 方法实现**
   - 位置：第 97-120 行
   - 在投影层之前添加未来天气融合逻辑：
     ```python
     # 在 enc_out, attns = self.encoder(...) 之后
     
     # 融合未来天气
     if self.use_future_weather and x_mark_future is not None:
         # x_mark_future: [B, pred_len, weather_dim]
         # 对未来天气进行聚合
         future_weather_agg = torch.mean(x_mark_future, dim=1)  # [B, weather_dim]
         
         # 扩展到所有变量维度
         future_weather_expanded = future_weather_agg.unsqueeze(1).expand(
             -1, enc_out.shape[1], -1
         )  # [B, C, weather_dim]
         
         # 拼接并通过融合模块
         enc_out_with_weather = torch.cat([enc_out, future_weather_expanded], dim=-1)
         dec_out = self.decoder_with_weather(enc_out_with_weather)
     else:
         dec_out = self.projection(enc_out)
     ```

4. **修改 `forward` 方法**
   - 位置：第 122-124 行
   - 改为：`def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_mark_future=None, mask=None):`
   - 并传递 `x_mark_future` 给 `forecast`

---

### 4️⃣ 训练循环 (exp/exp_main.py)

**修改内容：** 处理新的 5 元组数据和未来天气

**当前问题：**
```python
# 第 50-60 行
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
    # ❌ 没有处理 batch_y_mark_future
```

**修改步骤：**

1. **修改 `vali` 方法**
   - 位置：第 45-75 行
   - 改为处理 5 元组：
     ```python
     def vali(self, vali_data, vali_loader, criterion):
         total_loss = []
         self.model.eval()
         with torch.no_grad():
             for i, batch_data in enumerate(vali_loader):
                 # 处理可变长度的返回值
                 if len(batch_data) == 5:
                     batch_x, batch_y, batch_x_mark, batch_y_mark_hist, batch_y_mark_future = batch_data
                 else:
                     batch_x, batch_y, batch_x_mark, batch_y_mark_hist = batch_data
                     batch_y_mark_future = None
                 
                 batch_x = batch_x.float().to(self.device)
                 batch_y = batch_y.float()
                 batch_x_mark = batch_x_mark.float().to(self.device)
                 batch_y_mark_hist = batch_y_mark_hist.float().to(self.device)
                 if batch_y_mark_future is not None:
                     batch_y_mark_future = batch_y_mark_future.float().to(self.device)

                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                 # 调用模型时传入未来天气
                 ret = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark_hist, batch_y_mark_future)
                 if isinstance(ret, tuple):
                     outputs = ret[0]
                 else:
                     outputs = ret
                 
                 # ... 后续处理保持不变 ...
     ```

2. **修改 `train` 方法中的训练循环**
   - 位置：第 76-180 行
   - 类似修改，处理 5 元组和传递 `batch_y_mark_future`

3. **修改 `test` 方法**
   - 位置：第 183-250 行
   - 类似修改

**关键代码片段：**
```python
# 在 train 方法中
for i, batch_data in enumerate(train_loader):
    # 处理可变长度返回值
    if len(batch_data) == 5:
        batch_x, batch_y, batch_x_mark, batch_y_mark_hist, batch_y_mark_future = batch_data
    else:
        batch_x, batch_y, batch_x_mark, batch_y_mark_hist = batch_data
        batch_y_mark_future = None
    
    # ... 数据移到设备 ...
    
    # 调用模型
    ret = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark_hist, batch_y_mark_future)
```

---

### 5️⃣ 主入口脚本 (run.py)

**修改内容：** 添加新参数和实验标记

**当前问题：**
```python
# 第 30-40 行
# 没有区分是否有天气数据
```

**修改步骤：**

1. **添加新参数**
   - 位置：第 30-40 行（在 `--ablation_mode` 之后）
   - 添加：
     ```python
     # 天气相关参数
     parser.add_argument('--has_weather', type=int, default=0, 
                        help='whether dataset has real weather data: 0=no, 1=yes')
     parser.add_argument('--use_future_weather', type=int, default=0,
                        help='whether to use future weather in decoder: 0=no, 1=yes')
     parser.add_argument('--dataset_group', type=str, default='standard',
                        help='dataset group: standard (no weather) or climate (with weather)')
     ```

2. **修改 setting 字符串生成**
   - 位置：第 95-110 行
   - 改为：
     ```python
     # 根据是否有天气数据生成不同的 setting 字符串
     weather_suffix = f"_weather_future" if args.use_future_weather else ""
     if args.has_weather:
         setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_df{}_eb{}{}_{}' .format(
             args.model_id,
             args.model,
             args.data,
             args.features,
             args.seq_len,
             args.label_len,
             args.pred_len,
             args.d_model,
             args.e_layers,
             args.d_ff,
             args.enc_in,
             weather_suffix,
             ii)
     else:
         setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_df{}_eb{}_{}'.format(
             args.model_id,
             args.model,
             args.data,
             args.features,
             args.seq_len,
             args.label_len,
             args.pred_len,
             args.d_model,
             args.e_layers,
             args.d_ff,
             args.enc_in,
             ii)
     ```

3. **将参数传递给模型**
   - 位置：第 115-120 行
   - 确保 `args.use_future_weather` 被传递给模型

---

### 6️⃣ 新增文件：图卷积网络 (layers/GCN.py)

**创建新文件：** `layers/GCN.py`

**内容：** 见前面的详细分析文档中的 "改进2.1" 部分

**关键类：**
- `LocalGraphConv` - 局部图卷积
- `DualTrackSpatialModule` - 双轨空间模块

---

### 7️⃣ 新增文件：实验配置 (experiments_config.yaml)

**创建新文件：** `experiments_config.yaml`

**内容：** 见前面的详细分析文档中的 "改进3.2" 部分

---

### 8️⃣ 新增文件：消融实验脚本 (scripts/ablation_extended.sh)

**创建新文件：** `scripts/ablation_extended.sh`

**内容：** 见前面的详细分析文档中的 "改进4.1" 部分

---

### 9️⃣ 新增文件：定性分析脚本 (analysis/qualitative_analysis.py)

**创建新文件：** `analysis/qualitative_analysis.py`

**内容：** 见前面的详细分析文档中的 "改进4.2" 部分

---

## 🔄 修改顺序（推荐）

### 第一轮：修复因果性（必须）
1. ✅ 修改 `data_provider/data_loader.py` - Dataset_PEMS_Climate
2. ✅ 修改 `models/CAST.py` - 添加未来天气融合
3. ✅ 修改 `exp/exp_main.py` - 处理新数据格式
4. ✅ 修改 `run.py` - 添加参数

**验证方法：** 运行单个 PEMS03 实验，确保模型能正常训练

### 第二轮：修复空间拓扑（可选但推荐）
5. ✅ 创建 `layers/GCN.py`
6. ✅ 修改 `models/CAST.py` - 集成双轨模块

**验证方法：** 在 PEMS03 上对比有无双轨模块的性能

### 第三轮：规整实验（关键）
7. ✅ 创建 `experiments_config.yaml`
8. ✅ 重新训练所有模型

**验证方法：** 检查所有结果的一致性

### 第四轮：扩展消融（补充）
9. ✅ 创建 `scripts/ablation_extended.sh`
10. ✅ 创建 `analysis/qualitative_analysis.py`
11. ✅ 运行消融实验

---

## ⚠️ 常见陷阱

### 陷阱 1：时间戳对齐错误
**问题：** 未来天气的时间戳与预测时间窗口不对齐
**解决：** 确保 `seq_y_mark_future = self.data_stamp[r_end-self.pred_len:r_end]`

### 陷阱 2：向后兼容性破坏
**问题：** 修改后的模型无法处理无天气数据集
**解决：** 在所有地方添加 `if x_mark_future is not None:` 检查

### 陷阱 3：检查点不兼容
**问题：** 旧的模型检查点无法加载到新模型
**解决：** 从头训练，不使用旧检查点

### 陷阱 4：数据泄露
**问题：** 验证/测试集中的未来天气信息泄露到训练集
**解决：** 严格按照时间顺序划分数据集

---

## 📝 测试清单

- [ ] 单个 PEMS03 实验能正常运行
- [ ] 模型能正确处理 5 元组数据
- [ ] 未来天气被正确传递到模型
- [ ] 模型性能相比之前有提升
- [ ] 无天气数据集仍能正常运行
- [ ] 所有实验结果前后一致
- [ ] 消融实验在 PEMS03 上完成
- [ ] 定性分析图表生成正确

