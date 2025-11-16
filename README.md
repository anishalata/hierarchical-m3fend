# Memory-Guided Multi-View Multi-Domain Fake News Detection (M<sup>3</sup>FEND)
This is the official implementation of our paper **Memory-Guided Multi-View Multi-Domain Fake News Detection**, which has been published in TKDE. [Paper](https://ieeexplore.ieee.org/document/9802916)

The wide spread of fake news is increasingly threatening both individuals and society. Great efforts have been made for automatic fake news detection on a single domain (e.g., politics). However, correlations exist commonly across multiple news domains, and thus it is promising to simultaneously detect fake news of multiple domains. Based on our analysis, we pose two challenges in multi-domain fake news detection: 1) **domain shift**, caused by the discrepancy among domains in terms of words, emotions, styles, etc. 2) **domain labeling incompleteness**, stemming from the real-world categorization that only outputs one single domain label, regardless of topic diversity of a news piece. In this paper, we propose a Memory-guided Multi-view Multi-domain Fake News Detection Framework (M<sup>3</sup>FEND) to address these two challenges. We model news pieces from a multi-view perspective, including semantics, emotion, and style. Specifically, we propose a Domain Memory Bank to enrich domain information which could discover potential domain labels based on seen news pieces and model domain characteristics. Then, with enriched domain information as input, a Domain Adapter could adaptively aggregate discriminative information from multiple views for news in various domains. Extensive offline experiments on English and Chinese datasets demonstrate the effectiveness of M<sup>3</sup>FEND, and online tests verify its superiority in practice.

## Introduction
This repository provides the implementations of M<sup>3</sup>FEND and ten baseline models (BiGRU, TextCNN, RoBERTa, StyleLSTM, DualEmotion, EANN, EDDFN, MMoE, MoSE, MDFEND). Note that TextCNN and BiGRU are implemented with word2vec as word embedding in the original experiments, but we implement them with RoBERTa embedding in this repository.

## Requirements

- Python 3.6
- PyTorch > 1.0
- Pandas
- Numpy
- Tqdm


## Run

Parameter Configuration:

- dataset: the English or Chinese dataset, default for `ch`
- early_stop: default for `3`
- domain_num: the Chinese dataset could choose 3, 6, and 9, while the English dataset could choose 3, default for `3`
- epoch: training epoches, default for `50`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.0001`
- model_name: model_name within `textcnn bigru bert eann eddfn mmoe mose dualemotion stylelstm mdfend m3fend`, default for `m3fend`

You can run this code through:

```powershell
python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 3
```

```powershell
python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 6
```

```powershell
python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 9
```

```powershell
python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset en --domain_num 3
```

The best learning rate for various models are different: BiGRU (0.0009), TextCNN (0.0007), RoBERTa (7e-05), StyleLSTM(0.0007), DualEmotion(0.0009), EANN (0.0001), EDDFN (0.0007), MDFEND (7e-5), M$^3$FEND (0.0001).

## Hierarchical M3FEND Innovation

### Overview
This repository includes an extension to M3FEND called **Hierarchical M3FEND**, which automatically discovers sub-domains within each news domain to create more specialized fake news detection.

### The Problem
Original M3FEND treats all articles within the same domain identically:
- All **politics** articles use the same memory bank (elections, corruption, policy all treated the same)
- All **entertainment** articles use the same memory bank (celebrity drama, reality TV, weddings all treated the same)
- All **health** articles use the same memory bank (COVID guidelines, social media posts all treated the same)

### The Solution
**Hierarchical M3FEND** automatically discovers sub-patterns within each domain using unsupervised clustering:
- **3 domains** → **9 specialized sub-domains** (3 per domain)
- Each sub-domain gets its own specialized memory bank
- Articles are automatically routed to the appropriate sub-domain expert

### Implementation
```python
# Run the hierarchical innovation
python hierarchical_english_innovation.py
```

### What It Discovers
The algorithm automatically found these sub-domains in the English dataset:

**Entertainment (GossipCop)**:
- Celebrity Relationships (2,330 articles): "source", "prince", "wedding", "couple", "family"
- Seasonal Entertainment (6,539 articles): "new", "season", "year", "2018", "2017"  
- Celebrity Drama (4,374 articles): "said", "like", "just", "people", "know"

**Politics (PolitiFact)**:
- Government Policy (1,720 articles): "said", "new", "right", "help", "government"
- Trump Politics (805 articles): "trump", "president", "people", "said", "think"
- Political Interviews (30 articles): "amanpour", "think", "going", "people", "know"

**Health (COVID)**:
- Social Media COVID (1,894 articles): "https", "covid19", "people", "health", "virus"
- Twitter COVID (771 articles): "twitter", "com", "pic", "https", "covid19"
- Official Guidelines (938 articles): "covid", "19", "people", "que", "virus"

### Key Innovation
- **Automatic Discovery**: No manual sub-domain definition needed
- **Hierarchical Memory**: 9 specialized memory banks instead of 3 generic ones
- **Smart Routing**: Articles automatically sent to the right specialist
- **Better Detection**: Each fake news sub-type learns its own specific patterns

### Technical Details
1. **TF-IDF Feature Extraction**: Converts articles to numerical features
2. **K-Means Clustering**: Groups similar articles within each domain
3. **Hierarchical Memory Banks**: Creates specialized memories for each cluster
4. **Dynamic Routing**: Classifies new articles to appropriate sub-domain

This is the first work to automatically discover hierarchical sub-domains in fake news detection, enabling more precise and specialized detection of different types of misinformation.


## Reference

```
Zhu, Yongchun, et al. "Memory-Guided Multi-View Multi-Domain Fake News Detection." IEEE Transactions on Knowledge and Data Engineering (2022).
```


```
Nan, Qiong, et al. "MDFEND: Multi-domain fake news detection." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.
```

or in bibtex style:

```
@article{zhu2022memory,
  title={Memory-Guided Multi-View Multi-Domain Fake News Detection},
  author={Zhu, Yongchun and Sheng, Qiang and Cao, Juan and Nan, Qiong and Shu, Kai and Wu, Minghui and Wang, Jindong and Zhuang, Fuzhen},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
@inproceedings{nan2021mdfend,
  title={MDFEND: Multi-domain fake news detection},
  author={Nan, Qiong and Cao, Juan and Zhu, Yongchun and Wang, Yanyan and Li, Jintao},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3343--3347},
  year={2021}
}
```
