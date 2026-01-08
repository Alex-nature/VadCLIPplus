# Cherry Studio LLM Phrase Generator

使用Cherry Studio API为视频异常检测类别生成LLM增强的文本提示。

## 功能特性

- ✅ 使用Cherry Studio调用GPT模型
- ✅ 为13个UCF-Crime异常类别生成短语
- ✅ 每个类别生成5个描述短语
- ✅ JSON格式保存结果
- ✅ 智能错误处理和重试

## 环境要求

- Python 3.7+
- requests库: `pip install requests`
- Cherry Studio应用（运行中）

## 配置步骤

### 1. 获取Cherry Studio API密钥

1. 打开Cherry Studio应用
2. 进入设置 → API Keys
3. 创建新的API密钥或复制现有密钥

### 2. 配置环境变量

创建 `.env` 文件在项目根目录：

```bash
# 复制示例文件
cp env_cherry_studio_example.txt .env

# 编辑.env文件，填入您的API密钥
CHERRY_STUDIO_API_KEY=sk-cs-your-actual-api-key-here
```

### 3. 验证配置

```bash
python generate_phrases.py
```

## 使用方法

### 生成短语

```bash
# 基本使用（从.env文件读取API密钥）
python generate_phrases.py

# 自定义短语数量
python generate_phrases.py --num-phrases 3

# 指定输出文件
python generate_phrases.py --output custom_prompts.json --num-phrases 5
```

### 重新生成失败的类别

```bash
# 检查并重新生成失败的类别
python generate_phrases.py --regenerate-failed

# 自定义重新生成的短语数量
python generate_phrases.py --regenerate-failed --num-phrases 3
```

### 程序化使用

```python
from generate_phrases import CherryStudioPhraseGenerator
import os

# 初始化生成器
api_key = os.getenv('CHERRY_STUDIO_API_KEY')
gen = CherryStudioPhraseGenerator(api_key)

# 生成所有类别
result = gen.generate_all_categories(num_phrases=5)
gen.save_to_json(result, 'enhanced_prompts.json')

# 重新生成失败的类别
gen.regenerate_failed_categories('enhanced_prompts.json', num_phrases=5)

# 检查失败的类别
failed = gen.get_failed_categories('enhanced_prompts.json')
print(f"Failed categories: {failed}")
```

### 查看结果

```bash
# 查看生成的JSON文件
cat enhanced_prompts.json | head -20
```

## 输出格式

```json
{
  "metadata": {
    "description": "Enhanced text prompts for video anomaly detection categories",
    "generator": "CherryStudioPhraseGenerator",
    "total_categories": 13,
    "total_phrases": 65,
    "phrases_per_category": 5,
    "api_provider": "Cherry Studio"
  },
  "categories": {
    "abuse": [
      "physical violence observed",
      "harmful actions taken",
      "aggressive behavior displayed",
      "personal harm inflicted",
      "violent conduct shown"
    ],
    "fighting": [
      "physical confrontation occurring",
      "mutual combat engaged",
      "violent struggle witnessed",
      "aggressive altercation happening",
      "physical conflict unfolding"
    ]
  }
}
```

## 故障排除

### 常见问题

1. **API密钥无效**
   ```
   错误：API request failed... 401 Unauthorized
   解决：检查.env文件中的API密钥是否正确
   ```

2. **Cherry Studio未运行**
   ```
   错误：Connection refused
   解决：确保Cherry Studio应用正在运行
   ```

3. **网络连接问题**
   ```
   错误：Timeout
   解决：检查网络连接或增加超时时间
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试单个类别
gen = CherryStudioPhraseGenerator("your-api-key")
result = gen.generate_phrase_for_category("abuse", 1)
print("Test result:", result)
```

## 集成到训练流程

生成的 `enhanced_prompts.json` 可以直接用于VadCLIPplus的训练：

```python
# 在训练脚本中加载
import json

with open('enhanced_prompts.json', 'r') as f:
    data = json.load(f)

# 获取增强的prompt列表
categories = data['categories']
enhanced_prompts = []
for category, phrases in categories.items():
    enhanced_prompts.append(category)  # 类别名称
    enhanced_prompts.extend(phrases)   # 对应的短语

print(f"Total enhanced prompts: {len(enhanced_prompts)}")
```

## 文件说明

- `generate_phrases.py` - 主脚本
- `env_cherry_studio_example.txt` - 环境变量配置示例
- `enhanced_prompts.json` - 生成的短语数据（运行后产生）

## 性能优化

- 自动请求限流（1秒间隔）
- 智能缓存机制
- 错误重试机制
- 批量处理优化
