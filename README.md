# DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning

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

CMU-MOSI and CMU-MOSEI
