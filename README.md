# Memory-Guided Multi-View Multi-Domain Fake News Detection (M³FEND)

## Hierarchical M3FEND Innovation - DS340W Project

This repository extends M3FEND with **Hierarchical M3FEND**, which automatically discovers sub-domains within each news domain to create more specialized fake news detection.

---

## Quick Start for TAs / Evaluators

### 1. Clone the Repository
```bash
git clone <repository-url>
cd hierarchical-m3fend-clean
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- Python 3.6+
- PyTorch >= 1.0.0
- transformers >= 4.0.0
- pandas >= 1.0.0
- numpy >= 1.19.0
- scikit-learn >= 0.24.0
- tqdm >= 4.60.0

### 3. Download and Setup Data

The data files should be placed in the following structure:

```
data/
├── ch/
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
└── en/
    ├── train.pkl
    ├── val.pkl
    └── test.pkl
```

**Note:** 
- Data files are included in the repository (using Git LFS for the 192MB file)
- If you see `ch.zip` or `en.zip` files, you can extract them:
  ```bash
  cd data/ch && unzip ch.zip && cd ../en && unzip en.zip && cd ../..
  ```
- The `.pkl` files are already present and ready to use
- The data files (`.pkl` files) are expected to be pandas DataFrames with the following columns:
- `content`: Text content of the news article
- `comments`: Comments on the article
- `category`: Domain category (e.g., 'gossipcop', 'politifact', 'COVID' for English)
- `label`: Binary label (0 = real, 1 = fake)
- `content_emotion`: Emotion features for content
- `comments_emotion`: Emotion features for comments
- `emotion_gap`: Gap between content and comments emotions
- `style_feature`: Stylistic features

If you need to download the original datasets, please refer to the original M3FEND paper or contact the repository maintainer.

### 4. Verify Setup (Optional)

Run the verification script to check if everything is set up correctly:

```bash
python verify_setup.py
```

This checks Python version, dependencies, files, and data files.

### 5. Run the Main M3FEND Model

**For English dataset:**
```bash
python main.py --gpu 0 --lr 0.0001 --model_name m3fend --dataset en --domain_num 3 --epoch 50
```

**For Chinese dataset:**
```bash
python main.py --gpu 0 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 3 --epoch 50
```

### 6. Run the Hierarchical Innovation

To test the hierarchical sub-domain discovery and DAMA innovation:

```bash
python hierarchical_english_innovation.py
```

This will:
1. Discover sub-domains within each main domain using clustering
2. Create hierarchical memory banks
3. Test both the original hierarchical method and the new DAMA approach
4. Display comparative results

---

## Detailed Documentation

### Original M3FEND Framework

This is the official implementation of the paper **Memory-Guided Multi-View Multi-Domain Fake News Detection**, published in TKDE. [Paper](https://ieeexplore.ieee.org/document/9802916)

The wide spread of fake news is increasingly threatening both individuals and society. Great efforts have been made for automatic fake news detection on a single domain (e.g., politics). However, correlations exist commonly across multiple news domains, and thus it is promising to simultaneously detect fake news of multiple domains. Based on our analysis, we pose two challenges in multi-domain fake news detection: 1) **domain shift**, caused by the discrepancy among domains in terms of words, emotions, styles, etc. 2) **domain labeling incompleteness**, stemming from the real-world categorization that only outputs one single domain label, regardless of topic diversity of a news piece. In this paper, we propose a Memory-guided Multi-view Multi-domain Fake News Detection Framework (M³FEND) to address these two challenges. We model news pieces from a multi-view perspective, including semantics, emotion, and style. Specifically, we propose a Domain Memory Bank to enrich domain information which could discover potential domain labels based on seen news pieces and model domain characteristics. Then, with enriched domain information as input, a Domain Adapter could adaptively aggregate discriminative information from multiple views for news in various domains. Extensive offline experiments on English and Chinese datasets demonstrate the effectiveness of M³FEND, and online tests verify its superiority in practice.

### Introduction

This repository provides the implementations of M³FEND and ten baseline models (BiGRU, TextCNN, RoBERTa, StyleLSTM, DualEmotion, EANN, EDDFN, MMoE, MoSE, MDFEND). Note that TextCNN and BiGRU are implemented with word2vec as word embedding in the original experiments, but we implement them with RoBERTa embedding in this repository.

### Parameter Configuration

- `dataset`: the English or Chinese dataset, default for `ch`
- `early_stop`: default for `3`
- `domain_num`: the Chinese dataset could choose 3, 6, and 9, while the English dataset could choose 3, default for `3`
- `epoch`: training epochs, default for `50` (can be reduced for faster testing)
- `gpu`: the index of gpu you will use, default for `0`
- `lr`: learning_rate, default for `0.0001`
- `model_name`: model_name within `textcnn bigru bert eann eddfn mmoe mose dualemotion stylelstm mdfend m3fend`, default for `m3fend`
- `batchsize`: batch size, default for `16`
- `max_len`: maximum sequence length, default for `128`

### Example Commands

**English dataset with 3 domains:**
```bash
python main.py --gpu 0 --lr 0.0001 --model_name m3fend --dataset en --domain_num 3 --epoch 50
```

**Chinese dataset with 3 domains:**
```bash
python main.py --gpu 0 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 3 --epoch 50
```

**Chinese dataset with 6 domains:**
```bash
python main.py --gpu 0 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 6 --epoch 50
```

**Chinese dataset with 9 domains:**
```bash
python main.py --gpu 0 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 9 --epoch 50
```

**Recommended learning rates for different models:**
- BiGRU: 0.0009
- TextCNN: 0.0007
- RoBERTa: 7e-05
- StyleLSTM: 0.0007
- DualEmotion: 0.0009
- EANN: 0.0001
- EDDFN: 0.0007
- MDFEND: 7e-5
- M³FEND: 0.0001

---

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

```bash
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
- **DAMA Algorithm**: Dynamic Adaptive Memory Attention with BERT embeddings and cross-domain transfer learning

### Technical Details

1. **TF-IDF Feature Extraction**: Converts articles to numerical features
2. **K-Means Clustering**: Groups similar articles within each domain
3. **Hierarchical Memory Banks**: Creates specialized memories for each cluster
4. **Dynamic Routing**: Classifies new articles to appropriate sub-domain
5. **DAMA (Dynamic Adaptive Memory Attention)**: Advanced attention mechanism with BERT embeddings

This is the first work to automatically discover hierarchical sub-domains in fake news detection, enabling more precise and specialized detection of different types of misinformation.

---

## File Structure

```
hierarchical-m3fend-clean/
├── main.py                          # Main entry point for M3FEND training
├── grid_search.py                   # Grid search and training orchestration
├── hierarchical_english_innovation.py # Hierarchical innovation with DAMA
├── hierarchical_innovation.py       # (Deprecated) Older prototype version
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── verify_setup.py                  # Setup verification script
├── CHANGES_SUMMARY.md               # Summary of changes made
├── NOVELTY_CONTRIBUTIONS.md         # Technical documentation
├── data/                            # Data directory
│   ├── ch/                          # Chinese dataset
│   │   ├── train.pkl
│   │   ├── val.pkl
│   │   └── test.pkl
│   └── en/                          # English dataset
│       ├── train.pkl
│       ├── val.pkl
│       └── test.pkl
├── models/                          # Model implementations
│   ├── m3fend.py                    # M3FEND model
│   ├── mdfend.py                    # MDFEND baseline
│   ├── bert.py                      # BERT baseline
│   └── ...                          # Other baseline models
├── utils/                           # Utility functions
│   ├── dataloader.py                # Data loading utilities
│   └── utils.py                     # General utilities
├── logs/                            # Training logs
└── param_model/                     # Saved model parameters
```

---

## Troubleshooting

### Common Issues

1. **"FileNotFoundError: data/en/train.pkl"**: Data files should be present after cloning. If missing, extract from zip files:
   ```bash
   cd data/ch && unzip ch.zip && cd ../en && unzip en.zip && cd ../..
   ```

2. **"CUDA out of memory"**: Reduce batch size or sequence length:
   ```bash
   python main.py --batchsize 8 --max_len 64 --gpu 0 --model_name m3fend --dataset en --domain_num 3
   ```

3. **"ModuleNotFoundError"**: Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **CUDA/GPU errors**: Code works on CPU automatically. Ensure PyTorch is installed correctly.

5. **Model download**: First run downloads BERT/RoBERTa models from HuggingFace (requires internet).

---

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
