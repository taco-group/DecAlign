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

```bash
git clone https://github.com/taco-group/DecAlign.git
```

Prepare the Python environment:

```bash
cd DecAlign
conda create --name decalign python=3.9 -y
conda activate decalign
```

Install all the required libraries:

```bash
pip install -r requirements.txt
```

### Dataset Preparation

The preprocessing of CMU-MOSI, CMU-MOSEI and CH-SIMS follows [MMSA](https://github.com/thuiar/MMSA). For IEMOCAP, please refer to this file: https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view?usp=share_link.

After downloading or preprocessing the features, organize the data as follows:

```text
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

Dataset notes:

- `mosi` uses the aligned CMU-MOSI feature file `MOSI/mosi_aligned_50.pkl`.
- `mosei` uses the aligned CMU-MOSEI feature file `MOSEI/mosei_aligned_50.pkl`.
- `sims` refers to CH-SIMS in this repository. The recommended file is `CH-SIMS/chsims.pkl`.
- The aliases `ch-sims`, `ch_sims`, and `chsims` are normalized to `sims`.
- If `CH-SIMS/chsims.pkl` is not found, the loader can fall back to `SIMS/Processed/unaligned_39.pkl` when available, but for reproduction we recommend using the explicit `CH-SIMS/chsims.pkl` path above.

The `--data_dir` argument should point to the parent directory containing these dataset folders. For example, if the files are under `./data/MOSI`, `./data/MOSEI`, and `./data/CH-SIMS`, use `--data_dir ./data`.

### Configuration Files

Each dataset has a default JSON config:

| Dataset argument | Default config |
|------------------|----------------|
| `mosi` | `config/dec_mosi_config.json` |
| `mosei` | `config/dec_mosei_config.json` |
| `sims`, `ch-sims`, `ch_sims`, `chsims` | `config/dec_sims_config.json` |
| `iemocap` | `config/iemocap_decalign_config.json` |

`config.py` also sets dataset-specific paths, feature dimensions, task type, and safety defaults after the JSON file is loaded. Command-line overrides passed through `--config_override key=value` have the highest priority and are useful for controlled ablations.

### Training

Train DecAlign on CMU-MOSI dataset:

```bash
python main.py --dataset mosi --data_dir ./data --mode train --seeds 1111 --gpu_ids 0
```

Train on CMU-MOSEI dataset:

```bash
python main.py --dataset mosei --data_dir ./data --mode train --seeds 1111 --gpu_ids 0
```

Train on CH-SIMS dataset:

```bash
python main.py --dataset sims --data_dir ./data --mode train --seeds 1 --gpu_ids 0
```

Train on IEMOCAP dataset:

```bash
python main.py --dataset iemocap --data_dir ./data --mode train --seeds 1111 --gpu_ids 0
```

You can also run multiple seeds in one command:

```bash
python main.py --dataset mosi --data_dir ./data --mode train --seeds 1111 1112 1113 1114 1115 --gpu_ids 0
```

The output directories are:

```text
pt/      saved checkpoints
result/  CSV result files
log/     training logs
```

To keep different experiments separate, pass explicit output directories:

```bash
python main.py \
  --dataset sims \
  --data_dir ./data \
  --mode train \
  --seeds 1 \
  --gpu_ids 0 \
  --model_save_dir ./runs/sims_seed1/pt \
  --res_save_dir ./runs/sims_seed1/result \
  --log_dir ./runs/sims_seed1/log
```

**Command Line Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (`mosi`, `mosei`, `sims`, `ch-sims`, `ch_sims`, `chsims`, `iemocap`) | mosi |
| `--data_dir` | Path to data directory | ./data |
| `--mode` | Run mode (train or test) | train |
| `--seeds` | Random seeds for reproducibility | 1111 |
| `--gpu_ids` | GPU device IDs to use | 0 |
| `--model_save_dir` | Directory to save trained models | ./pt |
| `--res_save_dir` | Directory to save results | ./result |
| `--log_dir` | Directory to save logs | ./log |
| `--config_file` | Optional path to a custom JSON config | dataset default |
| `--config_override` | Override config values, e.g. `learning_rate=3e-5` | none |

Example with overrides:

```bash
python main.py \
  --dataset sims \
  --data_dir ./data \
  --mode train \
  --seeds 1 \
  --gpu_ids 0 \
  --config_override selection_metric='F1_score' \
  --config_override learning_rate=3.5e-5
```

### Evaluation

Evaluate a trained model:

```bash
python main.py --dataset mosi --data_dir ./data --mode test --gpu_ids 0
```

By default, test mode loads the checkpoint from `--model_save_dir` using the same model and dataset name. Make sure the checkpoint path matches the dataset you are evaluating.

### Project Structure

```text
DecAlign/
├── main.py                 # Entry point
├── config.py               # Configuration settings
├── data_loader.py          # Data loading utilities
├── config/
│   ├── dec_config.json
│   ├── dec_mosi_config.json
│   ├── dec_mosei_config.json
│   ├── dec_sims_config.json
│   └── iemocap_decalign_config.json
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
    ├── run_decalign.sh
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
