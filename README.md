# DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning

**ICLR 2026**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2503.11892) [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://taco-group.github.io/DecAlign/)

Authors: [Chengxuan Qian](https://qiancx.com/), [Shuo Xing](https://shuoxing98.github.io/), [Shawn Li](https://lili0415.github.io/), [Yue Zhao](https://viterbi-web.usc.edu/~yzhao010/lab), [Zhengzhong Tu](https://vztu.github.io/)

DecAlign is a novel hierarchical cross-modal alignment framework that explicitly disentangles multimodal representations into modality-unique (heterogeneous) and modality-common (homogeneous) components, which not only facilitates fine-grained alignment through prototype-guided optimal transport but also enhances semantic consistency via latent distribution matching. Moreover, DecAlign effectively mitigates distributional discrepancies while preserving modality-specific characteristics, yielding consistent performance improvements across multiple multimodal benchmarks.

<div align="center">
  <img src="figs\decalign_pip.png" alt="EMMA diagram" width="800"/>
  <p><em>Figure 1. The Framework of our proposed DecAlign approach.</em></p>
</div>

### Installation

Clone this repository:

```
git clone https://github.com/taco-group/DecAlign.git
```

Prepare the Python environment:

```
cd DecAlign
conda create --name decalign python=3.9 -y
conda activate decalign
```

Install all the required libraries:

`pip install -r requirements.txt `

### Dataset Preparation

The preprocess of CMU-MOSI, CMU-MOSEI and CH-SIMS datasets follows [MMSA](https://github.com/thuiar/MMSA). For IEMOCAP, please refer to the link: https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view?usp=share_link.
After downloading, organize the data in the following structure:
```
data/
├── MOSI/
│   └── mosi_aligned_50.pkl
├── MOSEI/
│   └── mosei_aligned_50.pkl
├── CH-SIMS/
│   └── chsims.pkl
└── IEMOCAP/
    └── iemocap_data.pkl
```

### Training

Train DecAlign on CMU-MOSI dataset:

```bash
python main.py --dataset mosi --data_dir ./data --mode train --seeds 1111 --gpu_ids 0
```

Train on CMU-MOSEI dataset:

```bash
python main.py --dataset mosei --data_dir ./data --mode train --seeds 1111 --gpu_ids 0
```

Train on IEMOCAP dataset:

```bash
python main.py --dataset iemocap --data_dir ./data --mode train --seeds 1111 --gpu_ids 0
```

**Command Line Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (mosi, mosei, iemocap) | mosi |
| `--data_dir` | Path to data directory | ./data |
| `--mode` | Run mode (train or test) | train |
| `--seeds` | Random seeds for reproducibility | 1111 |
| `--gpu_ids` | GPU device IDs to use | 0 |
| `--model_save_dir` | Directory to save trained models | ./pt |
| `--res_save_dir` | Directory to save results | ./result |
| `--log_dir` | Directory to save logs | ./log |

### Evaluation

Evaluate a trained model:

```bash
python main.py --dataset mosi --data_dir ./data --mode test --gpu_ids 0
```

### Project Structure

```
DecAlign/
├── main.py                 # Entry point
├── config.py               # Configuration settings
├── data_loader.py          # Data loading utilities
├── config/
│   └── dec_config.json     # Model hyperparameters
├── models/
│   └── model.py            # DecAlign model architecture
├── trains/
│   ├── ATIO.py             # Training logic
│   └── subNets/            # Sub-network modules
│       ├── BertTextEncoder.py
│       └── transformer.py
├── utils/
│   ├── functions.py        # Utility functions
│   └── metrices.py         # Evaluation metrics
└── scripts/                # Training scripts
    ├── run_mosi.sh
    ├── run_mosei.sh
    └── run_iemocap.sh
```

### Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{qian2026decalign,
  title={DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning},
  author={Qian, Chengxuan and Xing, Shuo and Li, Shawn and Zhao, Yue and Tu, Zhengzhong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

### Acknowledgement

This codebase is built upon [MMSA](https://github.com/thuiar/MMSA). We thank the authors for their excellent work.
