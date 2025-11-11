# 工具模块 (Utils)

本目录包含项目中使用的通用工具函数和辅助代码。

## 内容包括

- 数据加载和预处理工具
- 可视化工具函数
- 评估指标计算工具
- 文件I/O辅助函数
- 日志记录工具
- 配置文件解析器
- 常用数学函数
- 图像处理工具

## 使用示例

```python
from utils.data_loader import load_pact_data
from utils.visualization import plot_segmentation_result
from utils.metrics import calculate_dice_score
```

## 注意事项

- 保持工具函数的通用性和可复用性
- 每个函数应有清晰的文档字符串
- 编写单元测试确保工具函数正确性
