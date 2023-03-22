# Bayesian Team Imitation Learner

This is the official implementation of BTIL proposed in [Semi-Supervised Imitation Learning of Team Policies from Suboptimal Demonstrations](https://arxiv.org/abs/2205.02959), IJCAI 2022.

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
  title={Semi-Supervised Imitation Learning of Team Policies from Suboptimal Demonstrations},
  author={Seo, Sangwon and Vaibhav Unhelkar},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  url={https://arxiv.org/abs/2205.02959},
  year={2022},
  month={07}
}
```
