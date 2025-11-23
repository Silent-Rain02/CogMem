# LoCoMo 数据集评估方法详细指南

## 一、概述

LoCoMo（Long-Term Conversational Memory）数据集用于评估大语言模型在超长对话中的记忆能力。数据集包含多达35个会话，平均每个对话有300个回合。

## 二、评估流程

### 1. 数据准备

**数据格式：**
- 每个样本包含：`question`, `answer`, `category`, `evidence`, `sample_id`
- Category 编码：
  - 1 = Single Hop（单跳问题）
  - 2 = Multi-Hop（多跳问题）
  - 3 = Temporal（时间推理问题）
  - 4 = Open Domain（开放域问题）
  - 5 = Adversarial（对抗性问题）

**数据加载：**
```python
samples = json.load(open(args.data_file))
# 每个样本格式：
{
    "sample_id": "xxx",
    "qa": [
        {
            "question": "...",
            "answer": "...",
            "category": 1,
            "evidence": [...]
        }
    ]
}
```

### 2. 模型预测

对每个问题进行答案生成，保存预测结果：
```python
# 预测键名格式
prediction_key = f"{model_name}_prediction"
# 或使用RAG时：
prediction_key = f"{model_name}_{rag_mode}_top_{top_k}_prediction"
```

## 三、核心评估指标

### 1. 答案标准化（normalize_answer）

**官方实现：**
```python
def normalize_answer(s):
    """标准化答案文本"""
    s = s.replace(',', "")  # 移除逗号
    
    def remove_articles(text):
        # 移除冠词和连接词：a, an, the, and
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)
    
    def white_space_fix(text):
        # 规范化空白字符
        return ' '.join(text.split())
    
    def remove_punc(text):
        # 移除所有标点符号
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        # 转为小写
        return text.lower()
    
    # 按顺序执行
    return white_space_fix(remove_articles(remove_punc(lower(s))))
```

**关键点：**
- 必须移除逗号（`,`）
- 移除冠词和连接词：`a`, `an`, `the`, `and`
- 移除所有标点符号
- 转为小写
- 规范化空白字符

### 2. 精确匹配（Exact Match, EM）

**官方实现：**
```python
def exact_match_score(prediction, ground_truth):
    """精确匹配：使用词集合比较"""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    # 关键：使用词集合（set）比较，而不是字符串比较
    return set(prediction.split()) == set(ground_truth.split())
```

**关键点：**
- 不是字符串完全相等，而是**词集合相等**
- 标准化后，词顺序不重要
- 例如："May 7 2023" 和 "2023 May 7" 应该匹配

### 3. F1 分数

**官方实现（单答案）：**
```python
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from collections import Counter

def f1_score(prediction, ground_truth):
    """F1分数计算：使用词干提取（stemming）"""
    # 1. 标准化答案
    prediction_norm = normalize_answer(prediction)
    ground_truth_norm = normalize_answer(ground_truth)
    
    # 2. 词干提取（stemming）
    prediction_tokens = [ps.stem(w) for w in prediction_norm.split()]
    ground_truth_tokens = [ps.stem(w) for w in ground_truth_norm.split()]
    
    # 3. 计算共同词（考虑词频）
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    # 4. 计算精确率和召回率
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    
    # 5. 计算F1
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
```

**关键点：**
- **必须使用词干提取（PorterStemmer）**
- 使用 `Counter` 计算词频，考虑重复词
- 公式：`F1 = 2 * precision * recall / (precision + recall)`

**多答案F1（Multi-Hop问题）：**
```python
def f1(prediction, ground_truth):
    """多答案F1：预测和真实答案都可能有多个"""
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    
    # 对每个真实答案，找最匹配的预测
    # 然后取平均
    return np.mean([
        max([f1_score(pred, gt) for pred in predictions]) 
        for gt in ground_truths
    ])
```

## 四、不同类别问题的评估方式

### 1. Single Hop（category=2）

```python
# 使用单答案F1
all_ems.append(f1_score(output, answer))
```

### 2. Multi-Hop（category=1）

```python
# 使用多答案F1（预测和答案可能包含多个部分）
all_ems.append(f1(output, answer))  # 注意：这里是f1()，不是f1_score()
```

### 3. Temporal（category=3）

```python
# 特殊处理：如果答案包含分号，只取第一部分
if line['category'] == 3:
    answer = answer.split(';')[0].strip()

# 使用单答案F1
all_ems.append(f1_score(output, answer))
```

### 4. Open Domain（category=4）

```python
# 使用单答案F1
all_ems.append(f1_score(output, answer))
```

### 5. Adversarial（category=5）

```python
# 对抗性问题：检查是否识别出"无信息"
if 'no information available' in output.lower() or \
   'not mentioned' in output.lower():
    all_ems.append(1)
else:
    all_ems.append(0)
```

## 五、完整评估流程

### 步骤1：加载数据

```python
samples = json.load(open(data_file))
```

### 步骤2：生成预测

对每个问题生成答案：
```python
prediction = model.generate_answer(question, context)
line[prediction_key] = prediction
```

### 步骤3：评估每个QA样本

```python
def eval_question_answering(qas, eval_key='prediction'):
    all_ems = []  # 存储F1分数（注意：变量名是ems，但实际是F1）
    all_recall = []
    
    for line in qas:
        answer = str(line['answer'])
        output = line[eval_key]
        category = line['category']
        
        # 时间问题特殊处理
        if category == 3:
            answer = answer.split(';')[0].strip()
        
        # 根据不同类别选择评估方式
        if category in [2, 3, 4]:  # Single-hop, Temporal, Open-domain
            all_ems.append(f1_score(output, answer))
        elif category == 1:  # Multi-hop
            all_ems.append(f1(output, answer))  # 多答案F1
        elif category == 5:  # Adversarial
            if 'no information available' in output.lower() or \
               'not mentioned' in output.lower():
                all_ems.append(1.0)
            else:
                all_ems.append(0.0)
        
        # 计算召回率（recall accuracy）
        # ... (如果有evidence信息)
    
    return all_ems, lens, all_recall
```

### 步骤4：计算平均分数

```python
avg_f1 = np.mean(all_ems)
avg_em = sum([exact_match_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]) / len(predictions)
```

## 六、常见错误和注意事项

### ❌ 错误做法

1. **不使用词干提取**
   ```python
   # 错误
   tokens = normalize_answer(text).split()
   # 正确
   tokens = [ps.stem(w) for w in normalize_answer(text).split()]
   ```

2. **EM使用字符串比较**
   ```python
   # 错误
   return normalize_answer(pred) == normalize_answer(gt)
   # 正确
   return set(normalize_answer(pred).split()) == set(normalize_answer(gt).split())
   ```

3. **忽略类别差异**
   ```python
   # 错误：所有类别都用同一方法
   score = f1_score(output, answer)
   
   # 正确：区分不同类别
   if category == 1:
       score = f1(output, answer)  # 多答案
   else:
       score = f1_score(output, answer)  # 单答案
   ```

4. **标准化不完整**
   ```python
   # 错误：只转小写
   return text.lower()
   
   # 正确：完整的标准化流程
   return white_space_fix(remove_articles(remove_punc(lower(text))))
   ```

### ✅ 正确做法

1. **使用官方实现**
   - 直接使用 `locomo/task_eval/evaluation.py` 中的函数
   - 确保词干提取使用 `PorterStemmer`

2. **正确处理多答案**
   - Multi-Hop问题使用 `f1()` 而不是 `f1_score()`
   - 分割多个答案时使用逗号

3. **时间问题特殊处理**
   - 如果答案包含分号，只取第一部分

4. **完整的标准化流程**
   - 移除逗号 → 移除冠词 → 移除标点 → 转小写 → 规范化空白

## 七、评估结果格式

```python
{
    "total_samples": 1542,
    "avg_f1": 0.45,
    "avg_em": 0.12,
    "category_breakdown": {
        "Single Hop": {
            "count": 282,
            "avg_f1": 0.50,
            "avg_em": 0.15
        },
        "Multi-Hop": {
            "count": 321,
            "avg_f1": 0.42,
            "avg_em": 0.10
        },
        # ...
    }
}
```

## 八、参考实现

**官方代码位置：**
- `locomo/task_eval/evaluation.py` - 核心评估函数
- `locomo/task_eval/evaluate_qa.py` - 完整评估流程

**关键函数：**
- `normalize_answer()` - 答案标准化
- `exact_match_score()` - 精确匹配
- `f1_score()` - 单答案F1
- `f1()` - 多答案F1
- `eval_question_answering()` - 评估主函数

## 九、总结

LoCoMo评估的核心要点：
1. **答案标准化**：必须移除逗号、冠词、标点，转小写
2. **EM计算**：使用词集合比较，不是字符串比较
3. **F1计算**：必须使用词干提取（PorterStemmer）
4. **类别区分**：Multi-Hop使用多答案F1，其他用单答案F1
5. **特殊处理**：Temporal问题只取答案的第一部分（分号分割）

严格按照官方实现进行评估，确保结果的可比性！

