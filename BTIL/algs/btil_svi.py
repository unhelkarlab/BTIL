from typing import Optional, Tuple, Callable, Sequence
import numpy as np
import copy
from tqdm import tqdm
from scipy.special import digamma, logsumexp, softmax
from .transition_x import TransitionX

T_SAXSeqence = Sequence[Tuple[int, Tuple[int, int], Tuple[int, int]]]


class BTIL_SVI:

  def __init__(
      self,
      trajectories: Sequence[T_SAXSeqence],
      num_states: int,
      tuple_num_latents: Tuple[int, ...],
      tuple_num_actions: Tuple[int, ...],
      trans_x_dependency=(True, True, True, False),  # s, a1, ..., ak, s'
      max_iteration: int = 1000,
      epsilon: float = 0.001,
      lr: float = 0.1,
      lr_beta: float = 0.001,
      decay: float = 0.01,
      no_gem: bool = True) -> None:
    '''
      trajectories: list of list of (state, joint action)-tuples
      tuple_num_latents: number of possible latents
                        (or, for DP, truncated stick-breaking number + 1)
      trans_x_dependency: a tuple of booleans, each representing whether the
          latent transition function (Tx) depends on a specific variable. For
          example, in a two-agent task where Tx takes s,a1,a2 as its inputs
          (i.e., Tx(x'|s,a1,a2,x)), the tuple would be (True,True,True,False).
      epsilon: tolerance for the convergence criterion
      lr: learning rate for the pi and Tx parameters
      lr_beta: learning rate for the beta (needed only when no_gem=False)
      decay: decay rate for the learning rate
      no_gem: if False, GEM prior is used (default: True).
              (As in the case of our paper where you know the number of latents, 
              you don't need to use GEM.)
    '''

    assert len(tuple_num_actions) + 2 == len(trans_x_dependency)

    HYPER_GEM = 3
    HYPER_TX = 3
    HYPER_PI = 3

    self.trajectories = trajectories

    self.hyper_gem = HYPER_GEM
    self.hyper_tx = HYPER_TX
    self.list_hyper_pi = [HYPER_PI / n_a for n_a in tuple_num_actions]
    self.no_gem = no_gem

    self.num_agents = len(tuple_num_actions)
    self.num_ostates = num_states
    self.tuple_num_latents = tuple_num_latents
    self.tuple_num_actions = tuple_num_actions

    self.list_np_policy = [None for dummy_i in range(self.num_agents)
                           ]  # type: list[np.ndarray]
    self.tx_dependency = trans_x_dependency
    self.list_Tx = []  # type: list[TransitionX]
    self.list_bx = [None for dummy_i in range(self.num_agents)
                    ]  # type: list[np.ndarray]
    self.cb_bx = None

    self.max_iteration = max_iteration
    self.epsilon = epsilon

    self.list_param_beta = None  # type: list[np.ndarray]
    self.list_param_bx = None  # type: list[np.ndarray]
    self.list_param_pi = None  # type: list[np.ndarray]
    self.lr = lr
    self.decay = decay
    self.lr_beta = lr_beta

  def set_bx_and_Tx(self, cb_bx, cb_Tx=None):
    self.cb_bx = cb_bx

  def get_Tx(self, agent_idx, sidx, tup_aidx, sidx_n):
    return self.list_Tx[agent_idx].get_Tx_prop(sidx, tup_aidx, sidx_n)

  def get_bx(self, agent_idx, sidx):
    if self.cb_bx is not None:
      return self.cb_bx(agent_idx, sidx)
    else:
      return self.list_bx[agent_idx][sidx]

  def set_prior(self, gem_prior: float, tx_prior: float, pi_prior: float):
    self.hyper_gem = gem_prior
    self.hyper_tx = tx_prior
    self.list_hyper_pi = [pi_prior / n_a for n_a in self.tuple_num_actions]

  def compute_local_variables(self, samples):

    self.list_np_policy = [
        self.get_prob_tilda_from_lambda(self.list_param_pi[idx_a])
        for idx_a in range(self.num_agents)
    ]
    self.list_bx = [
        self.get_prob_tilda_from_lambda(self.list_param_bx[idx_a])
        for idx_a in range(self.num_agents)
    ]

    for idx_a in range(self.num_agents):
      self.list_Tx[idx_a].conv_to_Tx_tilda()

    list_list_q_x = []
    list_list_q_x_xn = []

    for m_th in range(len(samples)):
      trajectory = samples[m_th]

      list_q_x = []
      list_q_x_xn = []
      for idx_a in range(self.num_agents):
        n_lat = self.tuple_num_latents[idx_a]
        # Forward messaging
        with np.errstate(divide='ignore'):
          np_log_forward = np.log(np.zeros((len(trajectory), n_lat)))

        t = 0
        stt_p, joint_a_p, joint_x_p = trajectory[t]

        idx_xp = joint_x_p[idx_a]
        len_xp = 1
        if joint_x_p[idx_a] is None:
          idx_xp = slice(None)
          len_xp = n_lat

        with np.errstate(divide='ignore'):
          np_log_forward[t][idx_xp] = 0.0

          np_log_forward[t][idx_xp] += np.log(
              self.get_bx(idx_a, stt_p)[idx_xp] *
              self.list_np_policy[idx_a][idx_xp, stt_p, joint_a_p[idx_a]])

        # t = 1:N-1
        for t in range(1, len(trajectory)):
          t_p = t - 1
          stt, joint_a, joint_x = trajectory[t]

          idx_x = joint_x[idx_a]
          len_x = 1
          if joint_x[idx_a] is None:
            idx_x = slice(None)
            len_x = n_lat

          # yapf: disable
          with np.errstate(divide='ignore'):
            np_log_prob = np_log_forward[t_p][idx_xp].reshape(len_xp, 1)

            np_log_prob = np_log_prob + np.log(
              self.get_Tx(idx_a, stt_p, joint_a_p, stt)[idx_xp, idx_x].reshape(len_xp, len_x)  # noqa: E501
            )

            np_log_prob = np_log_prob + np.log(
              self.list_np_policy[idx_a][idx_x, stt, joint_a[idx_a]].reshape(1, len_x)  # noqa: E501
            )

          np_log_forward[t][idx_x] = logsumexp(np_log_prob, axis=0)
          # yapf: enable

          stt_p = stt
          joint_a_p = joint_a
          idx_xp = idx_x
          len_xp = len_x

        # Backward messaging
        with np.errstate(divide='ignore'):
          np_log_backward = np.log(np.zeros((len(trajectory), n_lat)))
        # t = N-1
        t = len(trajectory) - 1

        stt_n, joint_a_n, joint_x_n = trajectory[t]

        idx_xn = joint_x_n[idx_a]
        len_xn = 1
        if joint_x_n[idx_a] is None:
          idx_xn = slice(None)
          len_xn = n_lat

        np_log_backward[t][idx_xn] = 0.0

        # t = 0:N-2
        for t in reversed(range(0, len(trajectory) - 1)):
          t_n = t + 1
          stt, joint_a, joint_x = trajectory[t]

          idx_x = joint_x[idx_a]
          len_x = 1
          if joint_x[idx_a] is None:
            idx_x = slice(None)
            len_x = n_lat

          # yapf: disable
          with np.errstate(divide='ignore'):
            np_log_prob = np_log_backward[t_n][idx_xn].reshape(1, len_xn)  # noqa: E501

            np_log_prob = np_log_prob + np.log(
              self.get_Tx(idx_a, stt, joint_a, stt_n)[idx_x, idx_xn].reshape(len_x, len_xn)  # noqa: E501
            )

            np_log_prob = np_log_prob + np.log(
              self.list_np_policy[idx_a][idx_xn, stt_n, joint_a_n[idx_a]].reshape(1, len_xn)  # noqa: E501
            )

          np_log_backward[t][idx_x] = logsumexp(np_log_prob, axis=1)  # noqa: E501
          # yapf: enable

          stt_n = stt
          joint_a_n = joint_a
          idx_xn = idx_x
          len_xn = len_x

        # compute q_x, q_x_xp
        log_q_x = np_log_forward + np_log_backward

        q_x = softmax(log_q_x, axis=1)

        # n_x = self.num_lstates
        with np.errstate(divide='ignore'):
          log_q_x_xn = np.log(np.zeros((len(trajectory) - 1, n_lat, n_lat)))

        for t in range(len(trajectory) - 1):
          stt, joint_a, joint_x = trajectory[t]
          sttn, joint_a_n, joint_x_n = trajectory[t + 1]

          # yapf: disable
          with np.errstate(divide='ignore'):
            log_q_x_xn[t] = (
              np_log_forward[t].reshape(n_lat, 1) +
              np_log_backward[t + 1].reshape(1, n_lat)
            )

            log_q_x_xn[t] += np.log(
              self.list_Tx[idx_a].get_q_xxn(stt, joint_a, sttn)
            )

            log_q_x_xn[t] += np.log(
              self.list_np_policy[idx_a][:, sttn, joint_a_n[idx_a]].reshape(1, n_lat)
            )
          # yapf: enable

        q_x_xn = softmax(log_q_x_xn, axis=(1, 2))

        list_q_x.append(q_x)
        list_q_x_xn.append(q_x_xn)
      list_list_q_x.append(list_q_x)
      list_list_q_x_xn.append(list_q_x_xn)

    return list_list_q_x, list_list_q_x_xn

  def update_global_variables(self, samples, lr, lr_beta,
                              list_list_q_x: Sequence[Sequence[np.ndarray]],
                              list_list_q_x_xn: Sequence[Sequence[np.ndarray]]):

    batch_ratio = len(self.trajectories) / len(samples)
    list_param_pi_hat = [
        np.zeros_like(self.list_param_pi[idx_a])
        for idx_a in range(self.num_agents)
    ]
    list_param_bx_hat = [
        np.zeros_like(self.list_param_bx[idx_a])
        for idx_a in range(self.num_agents)
    ]
    list_param_tx_hat = [
        np.zeros_like(self.list_Tx[idx_a].np_lambda_Tx)
        for idx_a in range(self.num_agents)
    ]

    for m_th in range(len(samples)):
      traj = samples[m_th]
      list_q_x = list_list_q_x[m_th]
      list_q_xx = list_list_q_x_xn[m_th]

      t = 0
      state, joint_a, _ = traj[t]
      for idx_a in range(self.num_agents):
        list_param_bx_hat[idx_a] += list_q_x[idx_a][t]
        list_param_pi_hat[idx_a][:, state, joint_a[idx_a]] += list_q_x[idx_a][t]

      for t in range(1, len(traj)):
        tp = t - 1
        state_p, joint_a_p, _ = traj[tp]
        state, joint_a, _ = traj[t]
        for idx_a in range(self.num_agents):
          q_xx = list_q_xx[idx_a]
          q_x = list_q_x[idx_a]
          tx_index = self.list_Tx[idx_a].get_index(state_p, joint_a_p, state)
          list_param_tx_hat[idx_a][tx_index] += q_xx[tp]
          list_param_pi_hat[idx_a][:, state, joint_a[idx_a]] += q_x[t]

    # - update agent-level global variables
    for idx_a in range(self.num_agents):
      x_prior = self.hyper_tx * self.list_param_beta[idx_a]
      list_param_pi_hat[idx_a] = (batch_ratio * list_param_pi_hat[idx_a] +
                                  self.list_hyper_pi[idx_a])
      list_param_bx_hat[idx_a] = (batch_ratio * list_param_bx_hat[idx_a] +
                                  x_prior)
      list_param_tx_hat[idx_a] = (batch_ratio * list_param_tx_hat[idx_a] +
                                  x_prior)

      self.list_param_pi[idx_a] = ((1 - lr) * self.list_param_pi[idx_a] +
                                   lr * list_param_pi_hat[idx_a])
      self.list_param_bx[idx_a] = ((1 - lr) * self.list_param_bx[idx_a] +
                                   lr * list_param_bx_hat[idx_a])
      self.list_Tx[idx_a].np_lambda_Tx = (
          (1 - lr) * self.list_Tx[idx_a].np_lambda_Tx +
          lr * list_param_tx_hat[idx_a])

      if not self.no_gem:
        # -- update beta.
        # ref: https://people.eecs.berkeley.edu/~jordan/papers/liang-jordan-klein-haba.pdf
        # ref: http://proceedings.mlr.press/v32/johnson14.pdf
        num_K = len(self.list_param_beta[idx_a]) - 1
        grad_ln_p_beta = (np.ones(num_K) * (1 - self.hyper_gem) /
                          self.list_param_beta[idx_a][-1])
        for k in range(num_K):
          for i in range(k + 1, num_K):
            sum_beta = np.sum(self.list_param_beta[idx_a][:i])
            grad_ln_p_beta[k] += 1 / (1 - sum_beta)

        const_tmp = -digamma(x_prior) + digamma(sum(x_prior))
        param_tx = self.list_Tx[idx_a].np_lambda_Tx
        param_bx = self.list_param_bx[idx_a]
        sum_param_tx = np.sum(param_tx, axis=-1)
        sum_param_bx = np.sum(param_bx, axis=-1)
        grad_ln_E_p_tx = np.sum(digamma(param_tx) -
                                digamma(sum_param_tx)[..., None],
                                axis=tuple(range(param_tx.ndim - 1)))
        grad_ln_E_p_tx += np.sum(digamma(param_bx) -
                                 digamma(sum_param_bx)[..., None],
                                 axis=tuple(range(param_bx.ndim - 1)))
        num_x_dists = np.prod(param_tx.shape[:-1]) + np.prod(
            param_bx.shape[:-1])
        grad_ln_E_p_tx = (self.hyper_tx * grad_ln_E_p_tx[:-1] +
                          num_x_dists * const_tmp[:-1])

        grad_beta = grad_ln_p_beta + grad_ln_E_p_tx
        grad_beta_norm = np.linalg.norm(grad_beta)
        grad_beta /= grad_beta_norm

        # line search
        reach = np.zeros(num_K + 1)
        # distance to each canonical hyperplane
        reach[:-1] = -self.list_param_beta[idx_a][:-1] / grad_beta
        # signed distance to all-one hyperplane
        reach[-1] = self.list_param_beta[idx_a][-1] / np.sum(grad_beta)
        max_reach = min(reach[reach > 0])
        search_reach = min(max_reach, grad_beta_norm)
        self.list_param_beta[idx_a][:-1] = (self.list_param_beta[idx_a][:-1] +
                                            lr_beta * search_reach * grad_beta)
        self.list_param_beta[idx_a][-1] = (
            1 - np.sum(self.list_param_beta[idx_a][:-1]))

  def get_prob_tilda_from_lambda(self, np_lambda):

    sum_lambda_pi = np.sum(np_lambda, axis=-1)
    ln_pi = digamma(np_lambda) - digamma(sum_lambda_pi)[..., None]
    return np.exp(ln_pi)

  def initialize_param(self):
    INIT_RANGE = (1, 1.1)

    self.list_param_beta = []
    self.list_param_bx = []
    self.list_param_pi = []
    for idx_a in range(self.num_agents):
      num_x = self.tuple_num_latents[idx_a]
      # init beta
      if self.no_gem:
        self.list_param_beta.append(np.ones(num_x) / num_x)
      else:
        tmp_np_v = np.random.beta(1, self.hyper_gem, num_x - 1)
        tmp_np_beta = np.zeros(num_x)
        for idx in range(num_x - 1):
          tmp_np_beta[idx] = tmp_np_v[idx]
          for pidx in range(idx):
            tmp_np_beta[idx] *= 1 - tmp_np_v[pidx]
        tmp_np_beta[-1] = 1 - np.sum(tmp_np_beta[:-1])
        self.list_param_beta.append(tmp_np_beta)
      # init bx param
      self.list_param_bx.append(
          np.random.uniform(low=INIT_RANGE[0],
                            high=INIT_RANGE[1],
                            size=(self.num_ostates, num_x)))
      self.list_param_pi.append(
          np.random.uniform(low=INIT_RANGE[0],
                            high=INIT_RANGE[1],
                            size=(num_x, self.num_ostates,
                                  self.tuple_num_actions[idx_a])))
      # init tx param
      num_s = self.num_ostates if self.tx_dependency[0] else None
      list_num_a = []
      for i_a in range(self.num_agents):
        if self.tx_dependency[i_a + 1]:
          list_num_a.append(self.tuple_num_actions[i_a])
        else:
          list_num_a.append(None)

      num_sn = self.num_ostates if self.tx_dependency[-1] else None

      var_param_tx = TransitionX(num_x, num_s, tuple(list_num_a), num_sn, num_x)
      var_param_tx.init_lambda_Tx(*INIT_RANGE)
      self.list_Tx.append(var_param_tx)

  def do_inference(self, batch_size):
    num_traj = len(self.trajectories)
    batch_iter = int(num_traj / batch_size)
    count = 0
    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      if batch_size >= num_traj:
        samples = self.trajectories
      else:
        batch_idx = count % batch_iter
        if batch_idx == 0:
          perm_index = np.random.permutation(len(self.trajectories))

        samples = [
            self.trajectories[idx]
            for idx in perm_index[batch_idx * batch_size:(batch_idx + 1) *
                                  batch_size]
        ]

      count += 1

      delta_team = 0
      prev_list_param_bx = [
          np.copy(self.list_param_bx[idx_a]) for idx_a in range(self.num_agents)
      ]
      prev_list_param_pi = [
          np.copy(self.list_param_pi[idx_a]) for idx_a in range(self.num_agents)
      ]
      prev_list_param_tx = [
          np.copy(self.list_Tx[idx_a].np_lambda_Tx)
          for idx_a in range(self.num_agents)
      ]

      list_list_q_x, list_list_q_xx = self.compute_local_variables(
          samples)  # TODO: use batch

      # lr = (count + 1)**(-self.forgetting_rate)
      if self.lr == 1:
        lr = self.lr
      else:
        lr = self.lr / (count * self.decay + 1)
      lr_beta = self.lr_beta / (count * self.decay + 1)
      self.update_global_variables(samples, lr, lr_beta, list_list_q_x,
                                   list_list_q_xx)

      # compute delta
      for idx_a in range(self.num_agents):
        delta_team = max(
            delta_team,
            np.max(np.abs(self.list_param_bx[idx_a] -
                          prev_list_param_bx[idx_a])))
        delta_team = max(
            delta_team,
            np.max(np.abs(self.list_param_pi[idx_a] -
                          prev_list_param_pi[idx_a])))
        delta_team = max(
            delta_team,
            np.max(
                np.abs(self.list_Tx[idx_a].np_lambda_Tx -
                       prev_list_param_tx[idx_a])))

      if delta_team < self.epsilon:
        break
      progress_bar.update()
      progress_bar.set_postfix({'delta': delta_team})
    progress_bar.close()

    self.convert_params_to_prob()

  def convert_params_to_prob(self):
    for i_a in range(self.num_agents):
      numerator = self.list_param_pi[i_a]
      action_sums = np.sum(numerator, axis=-1)
      self.list_np_policy[i_a] = numerator / action_sums[..., np.newaxis]

      self.list_Tx[i_a].conv_to_Tx()

      if self.cb_bx is None:
        numerator = self.list_param_bx[i_a]
        latent_sums = np.sum(numerator, axis=-1)
        self.list_bx[i_a] = numerator / latent_sums[..., np.newaxis]
