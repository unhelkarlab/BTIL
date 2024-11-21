# Bayesian Team Imitation Learner

This is an implementation of BTIL proposed in [Semi-Supervised Imitation Learning of Team Policies from Suboptimal Demonstrations](https://arxiv.org/abs/2205.02959), IJCAI 2022.

## Installation

Clone this repository to your local PC and install it using the following commands:

```
cd BTIL/
pip install -e .
```

## Usage

Please import `BTIL_SVI` to use BTIL.

```
from BTIL.algs.btil_svi import BTIL_SVI
```

You can create a script that trains BTIL as follows:

```
btil = BTIL_SVI(...)
btil.set_prior(...)
btil.initialize_param()
btil.do_inference()
```

You can access the trained policies and mental state dynamics as follows:

```
btil.list_np_policy[i]  # learned policy for agent i
btil.list_Tx[i].np_Tx  # learned mental state dynamics for agent i
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
