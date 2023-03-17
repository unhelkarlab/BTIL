# Bayesian Team Imitation Learner

## Installation

We recommend you use `conda` environment. You can set up the virtual environment with following command:
`conda env create -f env.yml `

## Instruction

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
