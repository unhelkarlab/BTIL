import logging
import numpy as np
from scipy.special import digamma


class TransitionX:

  def __init__(self, num_x, num_s, tup_num_a, num_sn, num_xn):
    assert (num_x > 0)
    assert (num_xn > 0)
    logging.info("Gen TransX")

    self.num_x = num_x
    self.num_s = num_s
    self.tup_num_a = tup_num_a
    self.num_sn = num_sn
    self.num_xn = num_xn

    shape = []

    shape.append(self.num_x)
    if self.num_s:
      shape.append(self.num_s)
    for num_a in tup_num_a:
      if num_a:
        shape.append(num_a)
    if self.num_sn:
      shape.append(self.num_sn)
    shape.append(self.num_xn)

    self.shape = shape
    self.np_lambda_Tx = None
    self.np_Tx = None  # share between Tx_tilda and final Tx

  def get_index(self, s, tup_a, sn):
    index = [slice(None)]
    if self.num_s:
      index.append(s)
    for idx, num_a in enumerate(self.tup_num_a):
      if num_a:
        index.append(tup_a[idx])
    if self.num_sn:
      index.append(sn)
    index.append(slice(None))

    return tuple(index)

  def get_q_xxn(self, s, tup_a, sn):
    index = self.get_index(s, tup_a, sn)
    return self.np_Tx[index]

  def set_lambda_Tx_prior_param(self, beta):
    self.np_lambda_Tx = np.full(self.shape, beta)

  def add_to_lambda_Tx(self, s, tup_a, sn, q_xxn):
    index = self.get_index(s, tup_a, sn)
    self.np_lambda_Tx[index] += q_xxn

  def conv_to_Tx_tilda(self):
    sum_lambda_Tx = np.sum(self.np_lambda_Tx, axis=-1)
    ln_Txi = digamma(self.np_lambda_Tx) - digamma(sum_lambda_Tx)[..., None]
    self.np_Tx = np.exp(ln_Txi)

  def get_Tx_prop(self, s, tup_a, sn):
    index = self.get_index(s, tup_a, sn)
    return self.np_Tx[index]

  def conv_to_Tx(self):
    # following "Matthew J. Beal, 2003" and "MacKay, 1998", we don't subtract -1
    numerator = self.np_lambda_Tx
    next_latent_sums = np.sum(numerator, axis=-1)
    self.np_Tx = numerator / next_latent_sums[..., np.newaxis]

  def init_lambda_Tx(self, low, high):
    self.np_lambda_Tx = np.random.uniform(low=low, high=high, size=self.shape)

  def update_lambda_Tx(self, s, tup_a, sn, lambda_hat, lr):
    index = self.get_index(s, tup_a, sn)
    self.np_lambda_Tx[index] = ((1 - lr) * self.np_lambda_Tx[index] +
                                lr * lambda_hat)

  def get_lambda_Tx(self, s, tup_a, sn):
    index = self.get_index(s, tup_a, sn)
    return self.np_lambda_Tx[index]
