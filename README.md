# Enhancing urban flow prediction via mutual reinforcement with multi-scale regional information

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neunet.2024.106900-blue)](https://doi.org/10.1016/j.neunet.2024.106900)
[![Paper page](https://img.shields.io/badge/paper-page-blue)](https://codezx6.github.io/papers/mr-ufp.html)

Official implementation of **MR-UFP** (Neural Networks 2025): *Enhancing urban flow prediction via mutual reinforcement with multi-scale regional information*. The paper calls the method **MR-UFP**; this repository is named MR-UPF.

MR-UFP predicts the inflow and outflow of city grid regions, even with limited training data, by pre-training encoders with spatial-temporal masking and contrastive learning and training flow prediction jointly with a multi-scale region-classification task. On full TaxiBJ and BikeNYC it beats all 13 baselines (RMSE 14.32 and 4.45).

📄 Paper: https://doi.org/10.1016/j.neunet.2024.106900 · 🌐 Paper page with quoted results, FAQ and BibTeX: https://codezx6.github.io/papers/mr-ufp.html


A deep learning framework for spatio-temporal prediction incorporating multi-scale semantic representations and heterogeneous feature fusion mechanisms.

## Architecture

The framework consists of hierarchical encoding modules with cross-modal attention mechanisms and adaptive feature aggregation strategies.

## Citation

If you use this code, please cite the corresponding paper.

```bibtex
@article{zhang2025mrufp,
  title        = {Enhancing urban flow prediction via mutual reinforcement with multi-scale regional information},
  author       = {Zhang, Xu and Cao, Mengxin and Gong, Yongshun and Wu, Xiaoming and Dong, Xiangjun and Guo, Ying and Zhao, Long and Zhang, Chengqi},
  journal      = {Neural Networks},
  year         = {2025},
  volume       = {182},
  pages        = {106900},
  doi          = {10.1016/j.neunet.2024.106900},
  issn         = {0893-6080},
  url          = {https://doi.org/10.1016/j.neunet.2024.106900}
}
```
