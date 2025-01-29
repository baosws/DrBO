# Implementation for "Causal Discovery via Bayesian Optimization"

This is an implementation of our study:Bao Duong, Sunil Gupta, and Thin Nguyen. "Causal Discovery via Bayesian Optimization", to appear at ICLR'25.

DrBO (DAG Recovery via Bayesian Optimization) introduces a novel approach to score-based causal discovery, emphasizing sample efficiency, that is, recovering an accurate DAG with minimal trials. By leveraging Bayesian Optimization (BO), DrBO strategically utilizes exploration data to predict which DAGs are likely to have higher scores, prioritizing their evaluation. This targeted approach accelerates convergence to the highest-scoring DAG, reducing both the number of trials and overall computational time.

# Setup

```bash
conda create -n drbo python=3.10
conda activate drbo
pip install -r requirements.txt
```

# Usage

Please see our [demo](demo.ipynb) for how to use our method.

# Citations

If you find our study helpful, please consider citing us as:

```
@article{duong2025causal,
    title={Causal Discovery via Bayesian Optimization},
    author={Bao Duong and Sunil Gupta and Thin Nguyen},
    journal={arXiv preprint arXiv:2501.14997},
    year={2025},
}
```