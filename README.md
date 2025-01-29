# Causal Discovery via Bayesian Optimization

<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.10-green.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/abs/2501.14997"><img src="https://img.shields.io/badge/arXiv-2501.14997-b31b1b.svg" alt="arXiv" height="18"></a>
</p>

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#citation">Citation</a>
</p>

This is an implementation of our study: Bao Duong, Sunil Gupta, and Thin Nguyen. "Causal Discovery via Bayesian Optimization", to appear at ICLR'25.

**DrBO (DAG Recovery via Bayesian Optimization)** introduces a novel approach to score-based causal discovery, emphasizing sample efficiency, that is, recovering an accurate DAG with minimal trials. By leveraging Bayesian Optimization (BO), DrBO strategically utilizes exploration data to predict which DAGs are likely to have higher scores, prioritizing their evaluation. This targeted approach accelerates convergence to the highest-scoring DAG, reducing both the number of trials and overall computational time.

# Setup

```bash
conda create -n drbo python=3.10
conda activate drbo
pip install -r requirements.txt
```

# Usage

Please see our [demo](demo.ipynb) for how to use our method.

# Citation

If you find our study helpful, please consider citing us as:

```
@article{duong2025causal,
    title={Causal Discovery via Bayesian Optimization},
    author={Bao Duong and Sunil Gupta and Thin Nguyen},
    journal={arXiv preprint arXiv:2501.14997},
    year={2025},
}
```