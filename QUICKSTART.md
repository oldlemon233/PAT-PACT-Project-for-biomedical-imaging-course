# 快速开始指南
# Quick Start Guide

## 项目初始化

### 1. 克隆仓库
```bash
git clone https://github.com/moonlightbay/PAT-PACT-Project-for-biomedical-imaging-course.git
cd PAT-PACT-Project-for-biomedical-imaging-course
```

### 2. 创建Python虚拟环境
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 安装依赖
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 验证环境
```bash
python -c "import numpy; import matplotlib; print('Environment OK!')"
```

---

## 项目工作流程

### 阶段一：数字仿真（第1-2.5周）

#### Step 1: 配置仿真参数
```bash
cd simulation/configs/
# 编辑 simulation_params.yaml 文件
```

#### Step 2: 运行仿真
```bash
cd simulation/scripts/
# python generate_gold_standard.py
# python simulate_pact_signal.py
```

#### Step 3: 验证仿真数据
```bash
# 检查生成的数据
ls -lh ../results/
ls -lh ../../data/simulation/
```

---

### 阶段二：深度学习模型训练

#### Step 4: 训练重建网络
```bash
cd reconstruction/

# 配置网络参数
# 编辑 configs/network_config.yaml

# 开始训练
# python scripts/train.py --config configs/network_config.yaml

# 监控训练过程
tensorboard --logdir ../../outputs/logs/reconstruction
```

#### Step 5: 训练分割网络
```bash
cd segmentation/

# 配置网络参数
# 编辑 configs/model_config.yaml

# 开始训练
# python scripts/train.py --config configs/model_config.yaml

# 监控训练
tensorboard --logdir ../../outputs/logs/segmentation
```

---

### 阶段三：系统整合与评估

#### Step 6: 运行端到端评估
```bash
cd evaluation/

# 运行评估流水线
# python scripts/pipeline.py

# 查看结果
ls -lh results/
ls -lh visualizations/
```

#### Step 7: 生成报告
```bash
cd docs/reports/
# 编写实验报告
# 生成可视化结果用于报告
```

---

## 常用命令

### 数据管理
```bash
# 查看数据结构
tree data/ -L 2

# 检查数据大小
du -sh data/*
```

### 模型管理
```bash
# 列出所有检查点
find . -name "*.pth" -o -name "*.h5"

# 查看最新模型
ls -lt reconstruction/checkpoints/ | head -5
ls -lt segmentation/checkpoints/ | head -5
```

### 结果可视化
```bash
# 启动Jupyter notebook
jupyter notebook

# 打开可视化notebook
# notebooks/05_results_visualization.ipynb
```

### 测试
```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/test_simulation.py -v

# 查看测试覆盖率
pytest --cov=. tests/
```

---

## 目录导航

### 查看项目结构
```bash
# 完整结构
tree -L 3

# 仅查看目录
tree -d -L 2

# 查看特定模块
tree simulation/ -L 2
```

### 快速跳转
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias pact='cd /path/to/PAT-PACT-Project'
alias sim='cd /path/to/PAT-PACT-Project/simulation'
alias rec='cd /path/to/PAT-PACT-Project/reconstruction'
alias seg='cd /path/to/PAT-PACT-Project/segmentation'
alias eval='cd /path/to/PAT-PACT-Project/evaluation'
```

---

## 开发工具推荐

### IDE/编辑器
- **VS Code** - 推荐安装Python、Jupyter扩展
- **PyCharm** - 专业Python IDE
- **Jupyter Lab** - 交互式开发

### VS Code推荐扩展
```
- Python
- Pylance
- Jupyter
- GitLens
- YAML
- Markdown All in One
```

### 代码质量工具
```bash
# 安装代码质量工具
pip install black flake8 pylint mypy

# 格式化代码
black .

# 检查代码风格
flake8 .

# 类型检查
mypy .
```

---

## 常见问题

### Q: 如何查看项目整体进度？
A: 查看主README.md和docs/PROJECT_STRUCTURE.md

### Q: 数据应该放在哪里？
A: 
- 原始数据 → `data/raw/`
- 仿真数据 → `data/simulation/`
- 金标准 → `data/gold_standard/`
- 处理后数据 → `data/processed/`

### Q: 如何保存模型？
A:
- 训练中的模型 → `[module]/checkpoints/`
- 最终模型 → `[module]/checkpoints/best_model.pth`

### Q: 实验结果保存在哪？
A:
- 重建结果 → `reconstruction/results/`
- 分割结果 → `segmentation/results/`
- 评估结果 → `evaluation/results/`
- 可视化图表 → `outputs/figures/`

### Q: 如何记录实验？
A:
- 快速实验 → `notebooks/`
- 正式记录 → `docs/reports/`
- 配置参数 → 各模块的`configs/`目录

---

## 最佳实践

1. **版本控制**
   - 经常提交代码
   - 写清楚提交信息
   - 大文件使用.gitignore排除

2. **数据管理**
   - 保持原始数据不变
   - 记录数据处理步骤
   - 使用配置文件管理参数

3. **实验管理**
   - 使用配置文件而非硬编码
   - 记录实验参数和结果
   - 保存重要的可视化结果

4. **代码规范**
   - 遵循PEP 8
   - 添加文档字符串
   - 编写单元测试

5. **协作开发**
   - 使用分支开发新功能
   - 定期同步主分支
   - Code Review

---

## 获取帮助

- 查看各模块README：`cat [module]/README.md`
- 查看项目文档：`cat docs/PROJECT_STRUCTURE.md`
- 查看主README：`cat README.md`

---

## 下一步

1. ✅ 项目结构已创建
2. ⬜ 配置Python环境
3. ⬜ 开始实现仿真模块
4. ⬜ 开始实现重建网络
5. ⬜ 开始实现分割网络
6. ⬜ 集成评估流水线

祝你项目顺利！🎉
