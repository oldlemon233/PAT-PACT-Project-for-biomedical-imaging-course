# PAT-PACT 生物医学成像课程项目

## 项目主题

基于PACT与深度学习的缺血性脑卒中半暗带自动化分割与量化评估研究

**Automated Segmentation and Quantitative Assessment of Ischemic Stroke Penumbra Based on PACT and Deep Learning**

---

## 项目概述

本项目旨在开发一个基于光声计算机断层扫描（PACT）技术和深度学习方法的系统，用于缺血性脑卒中半暗带区域的自动化分割和量化评估。项目分为三个主要阶段：

### 第一阶段：数据构建与基础模型（第1周 - 第2.5周）

**任务1：构建数字仿真数据集**
- 简化仿真模型
- 生成金标准数据
- 模拟PACT信号

**任务2：图像重建与分割网络**
- 设计并实现重建网络
- 设计并实现分割网络

### 第三阶段：系统整合与量化分析

**任务3：系统整合与评估**
- 构建端到端评估流水线
- 量化性能评估
- 结果验证与分析
- 撰写报告与PPT

---

## 项目结构

```
PAT-PACT-Project/
│
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 预处理后的数据
│   ├── gold_standard/         # 金标准数据
│   └── simulation/            # 仿真生成的数据
│
├── simulation/                # 仿真模块（任务1）
│   ├── models/                # 仿真模型定义
│   ├── scripts/               # 仿真执行脚本
│   ├── configs/               # 仿真配置文件
│   └── results/               # 仿真结果
│
├── reconstruction/            # 重建模块（任务2.1）
│   ├── models/                # 重建网络模型
│   ├── scripts/               # 训练和推理脚本
│   ├── configs/               # 配置文件
│   ├── checkpoints/           # 模型检查点
│   └── results/               # 重建结果
│
├── segmentation/              # 分割模块（任务2.2）
│   ├── models/                # 分割网络模型
│   ├── scripts/               # 训练和推理脚本
│   ├── configs/               # 配置文件
│   ├── checkpoints/           # 模型检查点
│   └── results/               # 分割结果
│
├── evaluation/                # 评估模块（任务3）
│   ├── metrics/               # 评估指标实现
│   ├── scripts/               # 评估流水线脚本
│   ├── results/               # 评估结果
│   └── visualizations/        # 可视化结果
│
├── docs/                      # 文档
│   ├── reports/               # 项目报告
│   ├── presentations/         # 演示文稿
│   └── papers/                # 参考论文
│
├── notebooks/                 # Jupyter notebooks
├── utils/                     # 通用工具函数
├── tests/                     # 单元测试
├── configs/                   # 全局配置文件
│
├── outputs/                   # 输出文件
│   ├── figures/               # 图表
│   └── logs/                  # 日志文件
│
└── README.md                  # 本文件
```

---

## 快速开始

### 环境配置

```bash
# 克隆仓库
git clone <repository_url>
cd PAT-PACT-Project-for-biomedical-imaging-course

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖（待后续添加requirements.txt）
pip install -r requirements.txt
```

### 工作流程

1. **数据仿真阶段**
   ```bash
   cd simulation
   # 配置仿真参数
   # 运行仿真脚本
   # 生成金标准数据
   ```

2. **图像重建阶段**
   ```bash
   cd reconstruction
   # 训练重建网络
   # 对仿真数据进行重建
   ```

3. **分割网络阶段**
   ```bash
   cd segmentation
   # 训练分割网络
   # 对重建图像进行半暗带分割
   ```

4. **评估与验证**
   ```bash
   cd evaluation
   # 运行端到端评估流水线
   # 生成评估报告和可视化结果
   ```

---

## 主要技术栈

- **仿真**: 光学传播模拟、声学传播模拟
- **深度学习框架**: PyTorch / TensorFlow
- **图像重建**: 深度学习重建网络
- **图像分割**: U-Net及其变体
- **评估指标**: Dice系数、IoU、Hausdorff距离等
- **可视化**: Matplotlib, Seaborn, Plotly

---

## 目录详细说明

每个主要目录都包含独立的README.md文件，详细说明该模块的功能、使用方法和注意事项。请参考各目录中的README.md文件获取更多信息。

- [data/README.md](data/README.md) - 数据目录说明
- [simulation/README.md](simulation/README.md) - 仿真模块说明
- [reconstruction/README.md](reconstruction/README.md) - 重建模块说明
- [segmentation/README.md](segmentation/README.md) - 分割模块说明
- [evaluation/README.md](evaluation/README.md) - 评估模块说明
- [docs/README.md](docs/README.md) - 文档目录说明
- [notebooks/README.md](notebooks/README.md) - Notebooks使用说明
- [utils/README.md](utils/README.md) - 工具函数说明
- [tests/README.md](tests/README.md) - 测试说明

---

## 开发规范

### 代码规范
- 遵循PEP 8（Python）编码规范
- 函数和类应包含文档字符串
- 使用有意义的变量和函数名称
- 保持代码简洁和可读性

### Git使用规范
- 提交信息应清晰描述改动内容
- 频繁提交，保持提交粒度适中
- 大型数据文件使用Git LFS或外部存储
- 不要提交临时文件、日志文件和模型权重到仓库

### 文件命名规范
- 使用小写字母和下划线
- 配置文件：`module_name_config.yaml`
- 脚本文件：`action_description.py`
- 数据文件：包含日期和版本信息

---

## 项目时间表

| 阶段 | 时间 | 主要任务 |
|------|------|----------|
| 第一阶段 | 第1-2.5周 | 数据仿真、网络设计 |
| 第二阶段 | 待定 | （未在需求中明确） |
| 第三阶段 | 待定 | 系统整合、评估分析 |

---

## 贡献者

（待添加团队成员信息）

---

## 参考资料

（待添加相关论文和参考文献）

---

## 许可证

（待定）

---

## 联系方式

（待添加联系信息）