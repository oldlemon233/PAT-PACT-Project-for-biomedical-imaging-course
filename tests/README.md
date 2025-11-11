# 测试目录 (Tests)

本目录包含项目的单元测试和集成测试。

## 测试结构

建议按照项目模块组织测试：
- `test_simulation.py` - 仿真模块测试
- `test_reconstruction.py` - 重建模块测试
- `test_segmentation.py` - 分割模块测试
- `test_evaluation.py` - 评估模块测试
- `test_utils.py` - 工具函数测试

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_simulation.py

# 查看测试覆盖率
pytest --cov=. tests/
```

## 测试要求

- 每个模块应有对应的测试
- 关键函数必须有单元测试
- 测试应该快速且可重复
- 使用mock和fixture减少依赖
