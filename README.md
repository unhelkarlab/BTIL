# Bayesian Team Imitation Learner

This is an implementation of BTIL proposed in [Semi-Supervised Imitation Learning of Team Policies from Suboptimal Demonstrations](https://arxiv.org/abs/2205.02959), IJCAI 2022.

## Installation

Clone this repository to your local PC and install it using the following commands:

```
cd BTIL/
pip install -e .
```

## Usage

Please import `BTIL_Decen` in the file `BTIL/algs/btil_decentral.py` to use BTIL.

You can train btil using methods in the following order:

```
btil = BTIL_Decen(...)
btil.set_dirichlet_prior(...)
btil.set_bx_and_Tx(...)  # optional
btil.do_inference()
```

You can access the trained policies and mental state dynamics as follows:

```
btil.list_np_policy  # policy output
btil.list_Tx  # mental state dynamics output
```

## Citation
```
@inproceedings{seo2022semi,
  title     = {Semi-Supervised Imitation Learning of Team Policies from Suboptimal Demonstrations},
  author    = {Seo, Sangwon and Unhelkar, Vaibhav V.},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {2492--2500},
  year      = {2022},
  month     = {7},
  url       = {https://doi.org/10.24963/ijcai.2022/346}
}
```
