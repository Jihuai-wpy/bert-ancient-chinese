# 中文版

## 引言

当下人工智能和数字人文浪潮风靡全球，现代汉语自动分析已取得很大成果。而古代汉语的自动分析研究相对薄弱，难以满足国学、史学、文献学、汉语史的研究和国学、传统文化教育的实际需求。古汉语存在字词、词语、词类的诸多争议，资源建设困难重重。数字人文研究需要大规模语料库和高性能古文自然语言处理工具支持。鉴于预训练语言模型已经在英语和现代汉语文本上极大的提升了文本挖掘的精度，目前亟需专门面向古文自动处理领域的预训练模型。

2021年产生了两个较为高效的面向古文智能处理任务的预训练模型[`SikuBERT`和`SikuRoBERTa`](https://github.com/hsc748NLP/SikuBERT-for-digital-humanities-and-classical-Chinese-information-processing)，并被第一个古汉语领域NLP工具评估比赛——**[EvaHan 2022](https://circse.github.io/LT4HALA/2022/EvaHan)** 作为封闭环境下的预训练模型。**`bert-ancient-chinese`** 是我们为了进一步优化开放环境下模型效果得到的。

如果要引用我们的工作，您可以引用这篇[论文](https://aclanthology.org/2022.lt4hala-1.25/)：

```
@inproceedings{wang2022uncertainty,
  title={The Uncertainty-based Retrieval Framework for Ancient Chinese CWS and POS},
  author={Wang, Pengyu and Ren, Zhichen},
  booktitle={Proceedings of the Second Workshop on Language Technologies for Historical and Ancient Languages},
  pages={164--168},
  year={2022}
}
```

## 预训练

**相比于之前的预训练模型，`bert-ancient-chinese`主要有以下特点：**

- 古汉语文本多以繁体字出现，并且包含大量生僻汉字，这使得预训练模型的`vocab表`（词表）中并不存在部分生僻汉字。`bert-base-chinese`通过在大规模语料中进行学习，进一步扩充了预训练模型的`vocab`（词典），最终的`vocab表`大小为**38208**，相比于`bert-base-chinese`词表大小为**21128**，`siku-bert`词表大小为**29791**，`bert-ancient-chinese`拥有**更大的词表**，也收录了更多的生僻字，更有利于提升模型在下游任务的表现性能。`vocab表`即词表，收录在预训练模型中的`vocab.txt`中。
- `bert-ancient-chinese`使用了更大规模的训练集。相比于`siku-bert`只使用《四库全书》作为预训练数据集，我们使用了更大规模的数据集（约为《四库全书》的六倍），涵盖了从部、道部、佛部、集部、儒部、诗部、史部、医部、艺部、易部、子部，相比于四库全书内容更为丰富、范围更加广泛。

- 基于领域适应训练（Domain-Adaptive Pretraining）的思想，`bert-ancient-chinese`在`bert-base-chinese`的基础上结合古文语料进行继续训练，以获取面向古文自动处理领域的预训练模型。

## 使用方法

### Huggingface Transformers

基于[Huggingface Transformers](https://github.com/huggingface/transformers)的`from_pretrained`方法可以直接在线获取`bert-ancient-chinese`模型。

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Jihuai/bert-ancient-chinese")

model = AutoModel.from_pretrained("Jihuai/bert-ancient-chinese")
```

## 模型下载

我们提供的模型是`PyTorch`版本。

### 调用

通过Huggingface官网直接下载，目前官网的模型已同步更新至最新版本:

- **bert-ancient-chinese: [Jihuai/bert-ancient-chinese · Hugging Face](https://huggingface.co/Jihuai/bert-ancient-chinese)**

### 云盘

下载地址:

|       模型名称       |                           网盘链接                           |
| :------------------: | :----------------------------------------------------------: |
| bert-ancient-chinese | [链接](https://pan.baidu.com/s/1JC5_64gLT07wgG2hjzqxjg ) 提取码: qs7x |

## 验证与结果

我们在比赛[EvaHan 2022](https://circse.github.io/LT4HALA/2022/EvaHan)提供的训练集、测试集上对不同的预训练模进行了测试和比较。我们通过对模型在下游任务`自动分词`、`词性标注`上微调(fine-tuning)的性能进行了比较。

我们以`BERT+CRF`作为基线模型，对比了`siku-bert`、`siku-roberta`和`bert-ancient-chinese`在下游任务上的性能。为了充分利用整个训练数据集，我们采用` K 折交叉验证法`，同时其他超参均保持一致。评测指标为`F1值`。



<table>
   <tr>
      <td></td>
      <td colspan="2" align="center">《左传》</td>
      <td colspan="2" align="center">《史记》</td>
   </tr>
   <tr>
      <td></td>
      <td align="center">自动分词</td>
      <td align="center">词性标注</td>
      <td align="center">自动分词</td>
      <td align="center">词性标注</td>
   </tr>
   <tr>
      <td align="center">siku-bert</td>
      <td align="center">96.0670%</td>
      <td align="center">92.0156%</td>
      <td align="center">92.7909%</td>
      <td align="center">87.1188%</td>
   </tr>
   <tr>
      <td align="center">siku-roberta</td>
      <td align="center">96.0689%</td>
      <td align="center">92.0496%</td>
      <td align="center">93.0183%</td>
      <td align="center">87.5339%</td>
   </tr>
   <tr>
      <td align="center">bert-ancient-chinese</td>
      <td align="center"> <b>96.3273%</b> </td>
      <td align="center"> <b>92.5027%</b> </td>
      <td align="center"> <b>93.2917%</b> </td>
      <td align="center"> <b>87.8749%</b> </td>
   </tr>
</table>

## 引用

如果我们的内容有助您研究工作，欢迎在论文中引用。

## 免责声明

报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。实验结果可能因随机数种子，计算设备而发生改变。**使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。**

## 致谢

`bert-ancient-chinese`是基于[bert-base-chinese](https://huggingface.co/bert-base-chinese)继续训练得到的。

感谢[邱锡鹏教授](https://xpqiu.github.io/)和[复旦大学自然语言处理实验室](https://nlp.fudan.edu.cn/)。

## 联系我们

Pengyu Wang：wpyjihuai@gmail.com





# English version

## Introduction

With the current wave of Artificial Intelligence and Digital Humanities sweeping the world, the automatic analysis of modern Chinese has achieved great results. However, the automatic analysis and research of ancient Chinese is relatively weak, and it is difficult to meet the actual needs of Sinology, history, philology, Chinese history and the education of Sinology and traditional culture. There are many controversies about characters, words and parts of speech in ancient Chinese, and there are many difficulties in resource construction. Digital Humanities research requires large-scale corpora and high-performance ancient natural language processing tools. In view of the fact that pre-trained language models have greatly improved the accuracy of text mining in English and modern Chinese texts, there is an urgent need for pre-trained models for the automatic processing of ancient texts.

In 2021, two efficient pre-trained models for ancient Chinese intelligent processing tasks, [`SikuBERT` and `SikuRoBERTa`](https://github.com/hsc748NLP/SikuBERT-for-digital-humanities-and-classical-Chinese-information-processing), were produced and selected as pretrained models in closed environment by **[EvaHan 2022](https://circse.github.io/LT4HALA/2022/EvaHan)**, the first NLP tool evaluation competition in the field of ancient Chinese. We trained **`bert-ancient-chinese`** to  further optimize the model effect in open environment. 

If you want to refer to our work, you can refer to this [paper](https://aclanthology.org/2022.lt4hala-1.25/)：

```
@inproceedings{wang2022uncertainty,
  title={The Uncertainty-based Retrieval Framework for Ancient Chinese CWS and POS},
  author={Wang, Pengyu and Ren, Zhichen},
  booktitle={Proceedings of the Second Workshop on Language Technologies for Historical and Ancient Languages},
  pages={164--168},
  year={2022}
}
```

## Further Pre-training

**Compared with the previous pre-trained models, `bert-ancient-chinese` mainly has the following characteristics:**

- Ancient Chinese texts mostly appear in traditional Chinese characters and contain a large number of uncommon Chinese characters, which makes the `vocab table` (vocabulary) of the pre-trained model without some uncommon Chinese characters. `bert-ancient-chinese` further expands the `vocab` (dictionary) of the pre-trained model by learning in a large-scale corpus. The final `vocab table` size is **38208**, compared to `bert-base-chinese` vocabulary size of **21128**, `siku-bert` vocabulary size of **29791**, `bert-ancient-chinese` has a larger vocabulary, and also includes more uncommon vocabulary word, which is more conducive to improving the performance of the model in downstream tasks. The `vocab table` is the vocabulary table, which is included in the `vocab.txt` in the pre-trained model.
- `bert-ancient-chinese` uses a larger training set. Compared with `siku-bert` only using `"Siku Quanshu"` as training dataset, we use a larger-scale dataset (about six times that of `"Siku Quanshu"`), covering from the Ministry of Cong, the Ministry of Taoism, the Ministry of Buddhism, the Ministry of Confucianism, the Ministry of Poetry, the Ministry of History, the Ministry of Medicine, the Ministry of Art, the Ministry of Yi, and the Ministry of Zi, are richer in content and wider in scope than the `"Siku Quanshu"`.

- Based on the idea of `Domain-Adaptive Pretraining`, `bert-ancient-chinese` was trained on the basis of `bert-base-chinese ` and was combined with ancient Chinese corpus to obtain a pre-trained model for the field of automatic processing of ancient Chinese.

## How to use

### Huggingface Transformers

The `from_pretrained` method based on [Huggingface Transformers](https://github.com/huggingface/transformers) can directly obtain `bert-ancient-chinese` model online.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Jihuai/bert-ancient-chinese")

model = AutoModel.from_pretrained("Jihuai/bert-ancient-chinese")
```

## Download PTM

The model we provide is the `PyTorch` version.

### From Huggingface

Download directly through Huggingface's official website, and the model on the official website has been updated to the latest version simultaneously:

- **bert-ancient-chinese:[Jihuai/bert-ancient-chinese · Hugging Face](https://huggingface.co/Jihuai/bert-ancient-chinese)**

### From Cloud Disk

Download address:

|        Model         |                             Link                             |
| :------------------: | :----------------------------------------------------------: |
| bert-ancient-chinese | [Link](https://pan.baidu.com/s/1JC5_64gLT07wgG2hjzqxjg )  Extraction code: qs7x |

## Evaluation & Results

We tested and compared different pre-trained models on the training and test sets provided by the competition [EvaHan 2022](https://circse.github.io/LT4HALA/2022/EvaHan). We compare the performance of the models by fine-tuning them on the downstream tasks of `Chinese Word Segmentation(CWS)` and `part-of-speech tagging(POS Tagging)`.

We use `BERT+CRF` as the baseline model to compare the performance of `siku-bert`, `siku-roberta` and `bert-ancient-chinese` on downstream tasks. To fully utilize the entire training dataset, we employ `K-fold cross-validation`, while keeping other hyperparameters the same. The evaluation index is the `F1 value`.



<table>
   <tr>
      <td></td>
       <td colspan="2" align="center"> <i>Zuozhuan</i> </td>
       <td colspan="2" align="center"> <i>Shiji</i> </td>
   </tr>
   <tr>
      <td></td>
      <td align="center">CWS</td>
      <td align="center">POS</td>
      <td align="center">CWS</td>
      <td align="center">POS</td>
   </tr>
   <tr>
      <td align="center">siku-bert</td>
      <td align="center">96.0670%</td>
      <td align="center">92.0156%</td>
      <td align="center">92.7909%</td>
      <td align="center">87.1188%</td>
   </tr>
   <tr>
      <td align="center">siku-roberta</td>
      <td align="center">96.0689%</td>
      <td align="center">92.0496%</td>
      <td align="center">93.0183%</td>
      <td align="center">87.5339%</td>
   </tr>
   <tr>
      <td align="center">bert-ancient-chinese</td>
      <td align="center"> <b>96.3273%</b> </td>
      <td align="center"> <b>92.5027%</b> </td>
      <td align="center"> <b>93.2917%</b> </td>
      <td align="center"> <b>87.8749%</b> </td>
   </tr>
</table>

## Citing

If our content is helpful for your research work, please quote it in the paper.

## Disclaim

The experimental results presented in the report only show the performance under a specific data set and hyperparameter combination, and cannot represent the essence of each model. The experimental results may change due to random number seeds and computing equipment. **Users can use the model arbitrarily within the scope of the license, but we are not responsible for the direct or indirect losses caused by using the content of the project.**

## Acknowledgment

`bert-ancient-chinese` is based on [bert-base-chinese](https://huggingface.co/bert-base-chinese) to continue training.

Thanks to Prof. [Xipeng Qiu](https://xpqiu.github.io/) and the [Natural Language Processing Laboratory of Fudan University](https://nlp.fudan.edu.cn/).

## Contact us

Pengyu Wang：wpyjihuai@gmail.com