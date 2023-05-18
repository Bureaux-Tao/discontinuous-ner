# 基于UIE的小样本中文肺部CT病历实体关系抽取方法

本项目应用于抽取不连续实体，实现从标注到训练和预测。

## 标注与数据转换

不连续实体，即实体片段中间可能存在其他非本实体的情况与嵌套情况，如既往史“现病史：患者于1月前无意中出现左下腹不适，无腹泻、黑便，无头晕、头痛”中，“无腹泻、黑便”为省略写法，实际应表述为“无腹泻”与“无黑便”为两个实体，本项目实现了可以直接输入拆分的不连续实体功能。

### 标注工具

安装方式：使用Docker方式进行部。

拉取镜像

```shell
docker pull doccano/doccano
```

创建容器并初始化

```shell
docker container create --name doccano \\
  -e "ADMIN_USERNAME=[xxx]" \\
  -e "ADMIN_EMAIL=[xxx@xxx.com]" \\
  -e "ADMIN_PASSWORD=[password]" \\
  -v doccano-db:/data \\
  -p 8000:8000 doccano/doccano
```

启动容器

```shell
docker container start doccano
```

打开 `http://127.0.0.1:8000/` 即可看到

Doccano的使用方法：[链接](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/doccano.md)

### 标注方式

使用Doccano中关系标注的方式，用关系的指针从不连续实体的一端指向另一端，若为连续实体，则直接标注为实体类型。如下图所示：

![screenshot-01.png](assets%2Fscreenshot-01.png)

图中，“频繁恶心”是连续实体，所以仅使用“Span”实体标注方式标出即可，类别为A。而“频繁呕吐“为非连续实体，中间夹了”恶心、“，所以首先标出“频繁“、”呕吐“两个实体，类别一致，都为B，然后使用“Relation”关系标注方式，从“频繁“指向”呕吐“，关系类别也为B。

### 数据格式转换

Doccano导出的格式为jsonl，内容如下所示：

```json
{
    "id": 14275,
    "text": "现病史 : 患者 4月前 无明显诱 出现间断 腹痛、大便次数增多、可耐受。1天前再次腹痛、不能忍受、呈 阵发性加剧疼病无放射 、伴 频繁恶心、呕吐 、呕吐后腹病不减轻伴停止排便排气。 就诊于 当地医院 ，行保守治疗，小效果不佳。",
    "entities": [{
        "id": 238282,
        "label": "A",
        "start_offset": 66,
        "end_offset": 70
    }, {
        "id": 238283,
        "label": "B",
        "start_offset": 66,
        "end_offset": 68
    }, {
        "id": 238284,
        "label": "B",
        "start_offset": 71,
        "end_offset": 74
    }, {
        "id": 238286,
        "label": "A",
        "start_offset": 75,
        "end_offset": 83
    }, {
        "id": 238289,
        "label": "C",
        "start_offset": 84,
        "end_offset": 90
    }, {
        "id": 238290,
        "label": "C",
        "start_offset": 75,
        "end_offset": 78
    }],
    "relations": [{
        "id": 232,
        "from_id": 238283,
        "to_id": 238284,
        "type": "B"
    }, {
        "id": 233,
        "from_id": 238290,
        "to_id": 238289,
        "type": "C"
    }],
    "Comments": []
}
```

需要转换为逐字标签的形式，格式为json，如下所示：

```json
[{
    "sentence": ["现", "病", "史", " ", ":", " ", "患", "者", " ", "4", "月", "前", " ", "无", "明", "显", "诱", " ", "出", "现", "间", "断", " ", "腹", "痛", "、", "大", "便", "次", "数", "增", "多", "、", "可", "耐", "受", "。", "1", "天", "前", "再", "次", "腹", "痛", "、", "不", "能", "忍", "受", "、", "呈", " ", "阵", "发", "性", "加", "剧", "疼", "病", "无", "放", "射", " ", "、", "伴", " ", "频", "繁", "恶", "心", "、", "呕", "吐", " ", "、", "呕", "吐", "后", "腹", "病", "不", "减", "轻", "伴", "停", "止", "排", "便", "排", "气", "。", " ", "就", "诊", "于", " ", "当", "地", "医", "院", " ", "，", "行", "保", "守", "治", "疗", "，", "小", "效", "果", "不", "佳", "。"],
    "ner": [{
        "index": [66, 67, 71, 72, 73],
        "type": "B"
    }, {
        "index": [75, 76, 77, 84, 85, 86, 87, 88, 89],
        "type": "C"
    }, {
        "index": [66, 67, 68, 69],
        "type": "A"
    }, {
        "index": [75, 76, 77, 78, 79, 80, 81, 82],
        "type": "A"
    }]
}]
```

其中sentence为经过分字后的文本，为列表；ner为实体标注，每个实体为一个字典，index为实体每个字的下标，type为实体类型。

转换使用`doccano.py`脚本进行转换：

```shell
python doccano.py
--input_path data/changhai/chyy.jsonl   # doccano导出的jsonl文件
--output_directory data/changhai/       # 输出目录
--splits 0.75 0.25 0                    # 训练集、验证集、测试集划分比例，和为1
--maxlen 510                            # 文本最大长度，bert最大510
```

可以运行`python statistic.py`查看文本长度分布。

```
句数: 639
最长单句样本长度: 613
大于1000数量: 6
被截断比例: 0.00938967
```

若需要替换某类的一些高频词，以减少人工标注的工作量，可以标注之前使用`replace_frequency.py`脚本进行替换，然后再导入到Doccano中，其中：

```python
process("./data/changhai/data.txt",               # 原始数据
        "./data/changhai/data_replaced.jsonl",    # 替换后的数据
        {                                         # 替换字典，key为实体类型，value为文本中高频出现的实体，统一替换成key类型
            "消瘦": ["消瘦", "无消瘦"],
            "腹胀": ["无明显诱因出现腹痛腹胀", "无腹痛腹胀", "不伴腹痛腹胀", "不伴腹胀"],
            "便血": ["便血", "无明显诱因出现便血", "无便血", "不伴便血", "大便带血", "无明显诱因出现间断便血"],
            "腹痛": ["出现腹痛", "无腹痛", "不伴腹痛"],
            "大便习惯和性状改变": ["不伴大便习惯和性状改变", "无大便性状改变", "伴大便习惯和性状改变"],
            "腹泻": ["无腹泻", "伴腹泻", "有腹泻"],
            "大便形状改变": ["大便不成形", "大便变细", "粘液便"],
            ...
        }
)
```

## 结构

```
./
├── README.md
├── __pycache__
├── assets
├── config                                  运行配置
│   ├── cadec.json
│   ├── changhai.json
│   ├── example.json
│   └── resume-zh.json
├── config.py                               模型配置文件
├── data                                    数据集
│   ├── cadec
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── train.json
│   ├── changhai
│   │   ├── chyy.jsonl
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── train.json
│   ├── example
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── train.json
│   └── resume-zh
│       ├── dev.json
│       ├── test.json
│       └── train.json
├── data_loader.py                          数据加载器
├── doccano.py                              数据转换脚本
├── log                                     训练日志
├── main.py
├── model.py                                模型文件
├── plot.py
├── predict.py                              预测脚本
├── replace_frequency.py                    替换高频词
├── send_request.py
├── statistic.py                            统计数据集文本长度分布
├── train.py                                训练脚本
├── utils.py                                工具
└── weights                                 权重与类别词表
    ├── cadec.pt
    ├── cadec.vocab
    ├── changhai.vocab
    └── changhai_fgm.pt

13 directories, 63 files
```

## Requirements

```requirements.txt
colorama==0.4.5
colorlog==6.7.0
entrypoints==0.4
numpy==1.23.3
prettytable==3.7.0
protobuf==3.20.1
torch==1.12.1
transformers==4.28.1
tqdm
```

## 模型

[Unified Named Entity Recognition as Word-Word Relation Classification
](https://arxiv.org/abs/2112.10070)

文章所采取的Word-Pair标记方式，如下图所示：

![model-1.png](assets%2Fmodel-1.png)

这种方式可以看作是Token-Pair一种拓展：即建模Word和Word之间的关系，主要有两种Tag标记：

- **NNW(Next-Neighboring-Word)**:表示当前Word下一个连接接的Word；
- **THW(Tail-Head-Word-)**:实体的tail-Word到head-Word的连接，并附带实体的label信息。

![model-2.png](assets%2Fmodel-2.png)

通过上述的两种Tag标记方式连接任意两个Word，就可以解决如上图中各种复杂的实体抽取：（ABCD分别是一个Word）

- a): AB和CD代表两个扁平实体；
- b): 实体BC嵌套在实体ABC中；
- c): 实体ABC嵌套在非连续实体ABD；
- d): 两个非连续实体ACD和BCE；

下图展示了扁平实体(aching in legs)和非连续实体(aching in shoulders)的连接方式。

![model-3.png](assets%2Fmodel-3.png)

模型的连接结构：

![model-4.png](assets%2Fmodel-4.png)

## 训练

首先修改配置文件，可在`config`目录下找到对应的配置文件，或新建配置文件。配置文件示例：

```json
{
  "dataset": "example",                     // 数据集名称，文件夹名称需要统一
  "save_path": "weights/model.pt",          // 模型权重保存路径
  "vocab_path": "weights/vocab.vocab",      // 类别的词表保存路径

  "dist_emb_size": 20,
  "type_emb_size": 20,
  "lstm_hid_size": 512,
  "conv_hid_size": 128,
  "bert_hid_size": 768,
  "biaffine_size": 512,
  "ffnn_hid_size": 384,
  "dilation": [1, 2, 3, 4],

  "emb_dropout": 0.5,
  "conv_dropout": 0.5,
  "out_dropout": 0.33,

  "epochs": 10,                             // 训练轮数
  "batch_size": 8,                          // 批次大小

  "learning_rate": 1e-3,                    // 学习率
  "weight_decay": 0,                        // 权重衰减率
  "clip_grad_norm": 5.0,                    // 梯度裁剪

  "bert_name": "dmis-lab/biobert-v1.1",     // bert模型名称，对应huggingface中AutoModel支持的模型名称
  "bert_learning_rate": 5e-6,               // 下游学习率
  "warm_factor": 0.1,                       // warmup比例
  "early_stop_patience" : 3,                // 提前停止轮数

  "use_bert_last_4_layers": false,          // 是否使用bert的最后四层
  "seed": 123

}
```

训练命令：

```bash
python train.py --config config/example.json
```

输出：

```
2023-05-16 16:59:21 - INFO: dict_items([('dataset', 'changhai'), ('save_path', 'weights/changhai_fgm.pt'), ('vocab_path', 'weights/changhai.vocab'), ...])
2023-05-16 16:59:21 - INFO: Loading Data
2023-05-16 16:59:22 - INFO:
+----------+-----------+----------+
| changhai | sentences | entities |
+----------+-----------+----------+
|  train   |    479    |   2503   |
|   dev    |    160    |   779    |
|   test   |     0     |    0     |
+----------+-----------+----------+
2023-05-16 16:59:58 - INFO: Building Model ...
2023-05-16 17:00:02 - INFO: ----------Epoch: 1----------
Training epoch 1: 100%|██████████| 239/239 [01:27<00:00,  2.72it/s]
2023-05-16 17:01:42 - INFO:
+---------+--------+--------+-----------+--------+
| Train 1 |  Loss  |   F1   | Precision | Recall |
+---------+--------+--------+-----------+--------+
|  Label  | 0.3819 | 0.0639 |   0.0667  | 0.0660 |
+---------+--------+--------+-----------+--------+
Epoch 1 dev set: 100%|██████████| 80/80 [00:09<00:00,  8.42it/s]
2023-05-16 17:01:59 - INFO: VALID Label F1 [0.99990431 0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.        ]
2023-05-16 17:01:59 - INFO:
+---------+--------+-----------+--------+
| VALID 1 |   F1   | Precision | Recall |
+---------+--------+-----------+--------+
|  Label  | 0.0714 |   0.0714  | 0.0714 |
|  Entity | 0.0000 |   0.0000  | 0.0000 |
+---------+--------+-----------+--------+
2023-05-16 17:01:59 - INFO: Saved model to weights/changhai_fgm.pt.


... ...


2023-05-16 17:28:16 - INFO: ----------Epoch: 14----------
Training epoch 14: 100%|██████████| 239/239 [01:41<00:00,  2.36it/s]
/home/bureaux/miniconda3/envs/w2ner/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
2023-05-16 17:30:08 - INFO:
+----------+--------+--------+-----------+--------+
| Train 14 |  Loss  |   F1   | Precision | Recall |
+----------+--------+--------+-----------+--------+
|  Label   | 0.0002 | 0.7053 |   0.8263  | 0.6442 |
+----------+--------+--------+-----------+--------+
Epoch 14 dev set: 100%|██████████| 80/80 [00:09<00:00,  8.35it/s]
2023-05-16 17:30:26 - INFO: VALID Label F1 [0.99998335 0.94245385 0.88484848 0.93641618 0.93684211 0.87272727
 0.55555556 0.47058824 0.81617647 0.8974359  0.89814815 0.8
 0.83870968 0.57142857]
2023-05-16 17:30:26 - INFO:
+----------+--------+-----------+--------+
| VALID 14 |   F1   | Precision | Recall |
+----------+--------+-----------+--------+
|  Label   | 0.8158 |   0.7941  | 0.8656 |
|  Entity  | 0.9099 |   0.9256  | 0.8947 |
+----------+--------+-----------+--------+
2023-05-16 17:30:26 - INFO: Epoch did not improve: 3/3.
2023-05-16 17:30:26 - INFO: Early stopping at epoch 14.
2023-05-16 17:30:26 - INFO: Best DEV F1: 0.9126
```

## 性能

resume-zh中文简历数据集（无不连续实体）

![resume-zh.png](assets%2Fresume-zh.png)

CADEC英文医疗数据集（有不连续实体）

![cadec.png](assets%2Fcadec.png)

公司自标主诉既往史数据集（有不连续实体，不开源）

![changhai.png](assets%2Fchanghai.png)

主诉既往史数据集（加上了FGM对抗）

![changhai_fgm.png](assets%2Fchanghai_fgm.png)

实验发现加上对抗之后模型收敛更快，Precision和Recall差值更小，F1值更高。

## 预测

```shell
python predict.py --config ./config/example.json
```

其中的predict方法第二个参数需要传入分词方法，中英文不同，中文为按字分，英文为按词分

```python
def tokenize_word_en(text):
    # 将句号替换为空格+句号，这样句号就可以被当做单独的词处理
    text = re.sub(r'\.', ' .', text)
    # 将连续的空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 分词
    tokens = text.split()
    return tokens


def tokenize_char_cn(text):
    return list(text)


texts = [
    "现病史：患者于1月前无意中出现左下腹不适，无腹泻、黑便，无头晕、头痛、无恶心、呕吐等不适。遂就诊于新华医院素明分院，行肠镜检查示直肠息肉，横结肠瘤。",
    "现病史 : 患者 4月前 无明显诱 出现间断 腹痛、大便次数增多、可耐受。1天前再次腹痛、不能忍受、呈 阵发性加剧疼病无放射 、伴 频繁恶心、呕吐 、呕吐后腹病不减轻伴停止排便排气。 就诊于 当地医院 ，行保守治疗，小效果不佳。",
    "现病史：患者因大便带血、排便习惯及大便性状改变于2021-08-12至我院行结肠镜检查示：直肠癌？大肠多发息肉。",
    "主诉：腹泻1年。现病史：患者于2021年7月无明显诱因出现排便次数增多，少则每日3次，多则8次，粪便不成形或呈水样，无便血黑便等不适。"
]
print(predict(texts,
              # tokenize_word_en,
              tokenize_char_cn,
              config,
              batch_size=255))
```

预测输出：

```json
[{
    "text": "现病史：患者于1月前无意中出现左下腹不适，无腹泻、黑便，无头晕、头痛、无恶心、呕吐等不适。遂就诊于新华医院素明分院，行肠镜检查示直肠息肉，横结肠瘤。",
    "entities": [
      {"text": "无呕吐", "category": "呕吐"}, 
      {"text": "无恶心", "category": "恶心"}, 
      {"text": "无腹泻", "category": "腹泻"}, 
      {"text": "无黑便", "category": "黑便"}
    ]
}, {
    "text": "现病史:患者4月前无明显诱出现间断腹痛、大便次数增多、可耐受。1天前再次腹痛、不能忍受、呈阵发性加剧疼病无放射、伴频繁恶心、呕吐、呕吐后腹病不减轻伴停止排便排气。就诊于当地医院，行保守治疗，小效果不佳。",
    "entities": [
      {"text": "大便次数增多", "category": "大便习惯和性状改变"}, 
      {"text": "呕吐", "category": "呕吐"}
    ]
}, {
    "text": "现病史：患者因大便带血、排便习惯及大便性状改变于2021-08-12至我院行结肠镜检查示：直肠癌？大肠多发息肉。",
    "entities": [
      {"text": "大便带血", "category": "便血"}
    ]
}, {
    "text": "主诉：腹泻1年。现病史：患者于2021年7月无明显诱因出现排便次数增多，少则每日3次，多则8次，粪便不成形或呈水样，无便血黑便等不适。",
    "entities": [
      {"text": "排便次数增多", "category": "大便习惯和性状改变"}, 
      {"text": "无便血", "category": "便血"}, 
      {"text": "腹泻", "category": "腹泻"}, 
      {"text": "无黑便", "category": "黑便"}
    ]
}]
```

## 训练配置

| 配置       | 参数                                      |
| ---------- |-----------------------------------------|
| CPU        | Intel Xeon Silver 4214R (48) @ 2.401GHz |
| GPU        | NVIDIA Tesla V100S                      |
| 内存       | 128273 MB                               |
| 显存       | 32480 MB                                |
| 操作系统   | CentOS Linux 7 (Core) x86_64            |
| Python版本 | 3.9.13                                  |
| PyTorch    | 1.12.1                                  |

## 参考

> [https://github.com/ljynlp/W2NER](https://github.com/ljynlp/W2NER)
>
> [https://github.com/taishan1994/W2NER_predict](https://github.com/taishan1994/W2NER_predict)
