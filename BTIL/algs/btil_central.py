from typing import Optional, Tuple, Callable, Sequence
import numpy as np
from tqdm import tqdm
from scipy.special import digamma, logsumexp, softmax
from .transition_x import TransitionX

T_SAXSeqence = Sequence[Tuple[int, Tuple[int, int], Tuple[int, int]]]

# input: trajectories, mental models (optional), number of agents,
# output: policy table

# pi: |X| x |S| x |A|


class BTIL_Central:
  'deprecated - this is BTIL version 1.0. please use BTIL_Decen'

  def __init__(
      self,
      trajectories: Sequence[T_SAXSeqence],
      num_states: int,
      tuple_num_latents: Tuple[int, ...],
      tuple_num_actions: Tuple[int, ...],
      cb_transition_s,
      trans_x_dependency=(True, True, True, False),  # s, a1, a2, sn
      max_iteration: int = 100,
      epsilon: float = 0.001) -> None:

    assert len(tuple_num_actions) + 2 == len(trans_x_dependency)

    DIRICHLET_PARAM_PI = 3
    self.trajectories = trajectories

    self.beta_pi = DIRICHLET_PARAM_PI
    self.beta_Tx = DIRICHLET_PARAM_PI
    self.num_agents = len(tuple_num_actions)
    self.num_ostates = num_states
    self.tuple_num_latents = tuple_num_latents
    self.tuple_num_actions = tuple_num_actions
    self.cb_transition_s = cb_transition_s
    # num_agent x |X| x |S| x |A|
    self.list_np_policy = [None for dummy_i in range(self.num_agents)
                           ]  # type: list[np.ndarray]
    self.tx_dependency = trans_x_dependency
    self.list_Tx = None  # type: list[TransitionX]
    self.cb_Tx = None
    self.cb_bx = (
        lambda a, s: np.ones(tuple_num_latents[a]) / tuple_num_latents[a])

    self.max_iteration = max_iteration
    self.epsilon = epsilon

  def set_bx_and_Tx(self, cb_bx, cb_Tx=None):
    '''
    optional - use this function only if you know the prior distribution (cb_bx)
                or dynamics of the mental state (cb_Tx)
    '''
    self.cb_bx = cb_bx
    self.cb_Tx = cb_Tx

  def get_Tx(self, agent_idx, sidx, tup_aidx, sidx_n):
    if self.cb_Tx is not None:
      return self.cb_Tx(agent_idx, sidx, tup_aidx, sidx_n)
    else:
      return self.list_Tx[agent_idx].get_Tx_prop(sidx, tup_aidx, sidx_n)

  def set_dirichlet_prior(self, beta_pi: float, beta_Tx: float):
    # beta
    self.beta_pi = beta_pi
    self.beta_Tx = beta_Tx

  def estep_local_variables(self, list_policy):

    list_q_x = []  # Ntraj x Nstep x X^2
    list_q_x_xn = []  # Ntraj x Nstep x X^4
    for m_th in range(len(self.trajectories)):
      trajectory = self.trajectories[m_th]

      # Forward messaging
      with np.errstate(divide='ignore'):
        np_log_forward = np.log(
            np.zeros((len(trajectory), *self.tuple_num_latents)))

      t = 0
      stt_p, joint_a_p, joint_x_p = trajectory[t]

      idx_xp = [None] * self.num_agents
      len_xp = [None] * self.num_agents
      for idx in range(self.num_agents):
        if joint_x_p[idx] is None:
          idx_xp[idx] = slice(None)
          len_xp[idx] = self.tuple_num_latents[idx]
        else:
          idx_xp[idx] = joint_x_p[idx]
          len_xp[idx] = 1

      # yapf: disable
      with np.errstate(divide='ignore'):
        np_log_forward[t][tuple(idx_xp)] = 0.0

        for i_a in range(self.num_agents):
          none_1hot = [None] * self.num_agents
          none_1hot[i_a] = idx_xp[i_a]
          one_1hot = [1] * self.num_agents
          one_1hot[i_a] = len_xp[i_a]

          np_log_forward[t][tuple(idx_xp)] += np.log(
            self.cb_bx(i_a, stt_p)[tuple(none_1hot)].reshape(*one_1hot) *
            list_policy[i_a][:, stt_p, joint_a_p[i_a]][tuple(none_1hot)].reshape(*one_1hot))  # noqa: E501
      # yapf: enable

      # t = 1:N-1
      for t in range(1, len(trajectory)):
        t_p = t - 1
        stt, joint_a, joint_x = trajectory[t]

        idx_x = [None] * self.num_agents
        len_x = [None] * self.num_agents
        for idx in range(self.num_agents):
          if joint_x[idx] is None:
            idx_x[idx] = slice(None)
            len_x[idx] = self.tuple_num_latents[idx]
          else:
            idx_x[idx] = joint_x[idx]
            len_x[idx] = 1

        # yapf: disable
        with np.errstate(divide='ignore'):
          nones_wo_xp = idx_xp + [None] * self.num_agents
          ones_wo_xp = len_xp + [1] * self.num_agents
          np_log_prob = np_log_forward[t_p][tuple(nones_wo_xp)].reshape(*ones_wo_xp)  # noqa: E501

          for i_a in range(self.num_agents):
            none_xp_x_2hot = [None] * (2 * self.num_agents)
            none_xp_x_2hot[i_a] = idx_xp[i_a]
            none_xp_x_2hot[i_a + self.num_agents] = idx_x[i_a]

            one_xp_x_2hot = [1] * (2 * self.num_agents)
            one_xp_x_2hot[i_a] = len_xp[i_a]
            one_xp_x_2hot[i_a + self.num_agents] = len_x[i_a]

            np_log_prob = np_log_prob + np.log(
              self.get_Tx(i_a, stt_p, joint_a_p, stt)[tuple(none_xp_x_2hot)].reshape(*one_xp_x_2hot)  # noqa: E501
            )

            none_x_1hot = [None] * (2 * self.num_agents)
            none_x_1hot[i_a + self.num_agents] = idx_x[i_a]
            one_x_1hot = [1] * (2 * self.num_agents)
            one_x_1hot[i_a + self.num_agents] = len_x[i_a]
            np_log_prob = np_log_prob + np.log(
              list_policy[i_a][:, stt, joint_a[i_a]][tuple(none_x_1hot)].reshape(*one_x_1hot)  # noqa: E501
            )

          np_log_prob = np_log_prob + np.log(self.cb_transition_s(stt_p, joint_a_p, stt))  # noqa: E501

        np_log_forward[t][tuple(idx_x)] = logsumexp(np_log_prob, axis=tuple(range(self.num_agents)))  # noqa: E501
        # yapf: enable

        stt_p = stt
        joint_a_p = joint_a
        idx_xp = idx_x
        len_xp = len_x

      # Backward messaging
      with np.errstate(divide='ignore'):
        np_log_backward = np.log(
            np.zeros((len(trajectory), *self.tuple_num_latents)))
      # t = N-1
      t = len(trajectory) - 1

      stt_n, joint_a_n, joint_x_n = trajectory[t]

      idx_xn = [None] * self.num_agents
      len_xn = [None] * self.num_agents
      for idx in range(self.num_agents):
        if joint_x_n[idx] is None:
          idx_xn[idx] = slice(None)
          len_xn[idx] = self.tuple_num_latents[idx]
        else:
          idx_xn[idx] = joint_x_n[idx]
          len_xn[idx] = 1

      np_log_backward[t][tuple(idx_xn)] = 0.0

      # t = 0:N-2
      for t in reversed(range(0, len(trajectory) - 1)):
        t_n = t + 1
        stt, joint_a, joint_x = trajectory[t]

        idx_x = [None] * self.num_agents
        len_x = [None] * self.num_agents
        for idx in range(self.num_agents):
          if joint_x[idx] is None:
            idx_x[idx] = slice(None)
            len_x[idx] = self.tuple_num_latents[idx]
          else:
            idx_x[idx] = joint_x[idx]
            len_x[idx] = 1

        # yapf: disable
        with np.errstate(divide='ignore'):
          nones_wo_x = [None] * self.num_agents + idx_xn
          ones_wo_x = [1] * self.num_agents + len_xn
          np_log_prob = np_log_backward[t_n][tuple(nones_wo_x)].reshape(*ones_wo_x)  # noqa: E501

          for i_a in range(self.num_agents):
            none_x_xn_2hot = [None] * (2 * self.num_agents)
            none_x_xn_2hot[i_a] = idx_x[i_a]
            none_x_xn_2hot[i_a + self.num_agents] = idx_xn[i_a]
            one_x_xn_2hot = [1] * (2 * self.num_agents)
            one_x_xn_2hot[i_a] = len_x[i_a]
            one_x_xn_2hot[i_a + self.num_agents] = len_xn[i_a]

            np_log_prob = np_log_prob + np.log(
              self.get_Tx(i_a, stt, joint_a, stt_n)[tuple(none_x_xn_2hot)].reshape(*one_x_xn_2hot)  # noqa: E501
            )

            none_xn_1hot = [None] * (2 * self.num_agents)
            none_xn_1hot[i_a + self.num_agents] = idx_xn[i_a]
            one_xn_1hot = [1] * (2 * self.num_agents)
            one_xn_1hot[i_a + self.num_agents] = len_xn[i_a]

            np_log_prob = np_log_prob + np.log(
              list_policy[i_a][:, stt_n, joint_a_n[i_a]][tuple(none_xn_1hot)].reshape(*one_xn_1hot)  # noqa: E501
            )

          np_log_prob = np_log_prob + np.log(self.cb_transition_s(stt, joint_a, stt_n))  # noqa: E501

        np_log_backward[t][tuple(idx_x)] = logsumexp(np_log_prob, axis=tuple(range(self.num_agents, 2 * self.num_agents)))  # noqa: E501
        # yapf: enable

        stt_n = stt
        joint_a_n = joint_a
        # joint_x_n = joint_x
        idx_xn = idx_x
        len_xn = len_x

      # compute q_x, q_x_xp
      log_q_joint_x = np_log_forward + np_log_backward

      qx_all = []
      for i_a in range(self.num_agents):
        axis = tuple(range(1, i_a + 1)) + tuple(
            range(i_a + 2, self.num_agents + 1))
        log_q_x = logsumexp(log_q_joint_x, axis=axis)
        q_x = softmax(log_q_x, axis=1)
        qx_all.append(q_x)

      list_q_x.append(qx_all)

      if self.cb_Tx is None:
        # n_x = self.num_lstates
        with np.errstate(divide='ignore'):
          log_q_xx_xnxn = np.log(
              np.zeros((len(trajectory) - 1, *self.tuple_num_latents,
                        *self.tuple_num_latents)))  # noqa: E501

        for t in range(len(trajectory) - 1):
          stt, joint_a, joint_x = trajectory[t]
          sttn, joint_a_n, joint_x_n = trajectory[t + 1]

          # yapf: disable
          with np.errstate(divide='ignore'):
            ones = [1] * self.num_agents
            log_q_xx_xnxn[t] = (
              np_log_forward[t].reshape(*self.tuple_num_latents, *ones) +
              np_log_backward[t + 1].reshape(*ones, *self.tuple_num_latents)
            )

            for i_a in range(self.num_agents):
              one_x_xn_2hot = [1] * (2 * self.num_agents)
              one_x_xn_2hot[i_a] = self.tuple_num_latents[i_a]
              one_x_xn_2hot[i_a + self.num_agents] = self.tuple_num_latents[i_a]

              log_q_xx_xnxn[t] += np.log(
                self.list_Tx[i_a].get_q_xxn(stt, joint_a, sttn).reshape(*one_x_xn_2hot)  # noqa: E501
              )

              one_xn_1hot = [1] * (2 * self.num_agents)
              one_xn_1hot[i_a + self.num_agents] = self.tuple_num_latents[i_a]

              log_q_xx_xnxn[t] += np.log(
                list_policy[i_a][:, sttn, joint_a_n[i_a]].reshape(*one_xn_1hot)
              )

            log_q_xx_xnxn[t] += np.log(self.cb_transition_s(stt, joint_a, sttn))
          # yapf: enable

        q_x_xn_all = []
        for i_a in range(self.num_agents):
          axis = (
              tuple(range(1, i_a + 1)) +
              tuple(range(i_a + 2, self.num_agents + 1)) +
              tuple(range(self.num_agents + 1, self.num_agents + i_a + 1)) +
              tuple(range(self.num_agents + i_a + 2, 2 * self.num_agents + 1)))
          log_q_x_xn = logsumexp(log_q_xx_xnxn, axis=axis)
          q_x_xn = softmax(log_q_x_xn, axis=(1, 2))
          q_x_xn_all.append(q_x_xn)

        list_q_x_xn.append(q_x_xn_all)

    return list_q_x, list_q_x_xn

  def mstep_global_variables(self, list_q_x: Sequence[Sequence[np.ndarray]],
                             list_q_x_xn: Sequence[Sequence[np.ndarray]]):

    list_lambda_pi = []
    # policy
    for idx in range(self.num_agents):
      lambda_pi = np.full((self.tuple_num_latents[idx], self.num_ostates,
                           self.tuple_num_actions[idx]), self.beta_pi)
      list_lambda_pi.append(lambda_pi)

    for m_th in range(len(self.trajectories)):
      q_x_all = list_q_x[m_th]
      traj = self.trajectories[m_th]
      for t, (state, joint_a, joint_x) in enumerate(traj):
        for i_a in range(self.num_agents):
          list_lambda_pi[i_a][:, state, joint_a[i_a]] += q_x_all[i_a][t, :]

    # transition_x
    if self.cb_Tx is None and len(list_q_x_xn) > 0:
      for i_a in range(self.num_agents):
        self.list_Tx[i_a].set_lambda_Tx_prior_param(self.beta_Tx)

      for m_th in range(len(self.trajectories)):
        # q_x_xn1, q_x_xn2 = list_q_x_xn[m_th]
        q_x_xn_all = list_q_x_xn[m_th]
        traj = self.trajectories[m_th]
        # for t, state, joint_a, joint_x in enumerate(traj):
        for t in range(len(traj) - 1):
          state, joint_a, _ = traj[t]
          state_n, _, _ = traj[t + 1]

          for i_a in range(self.num_agents):
            self.list_Tx[i_a].add_to_lambda_Tx(state, joint_a, state_n,
                                               q_x_xn_all[i_a][t, :, :])

    return list_lambda_pi

  def get_pi_tilda_from_lambda(self, list_lambda_pi):
    list_pi_tilda = []

    for idx in range(self.num_agents):
      sum_lambda_pi = np.sum(list_lambda_pi[idx], axis=-1)
      ln_pi = digamma(list_lambda_pi[idx]) - digamma(sum_lambda_pi)[:, :, None]
      list_pi_tilda.append(np.exp(ln_pi))

    return list_pi_tilda

  def do_inference(self,
                   callback: Optional[Callable[[int, Sequence[np.ndarray]],
                                               None]] = None):
    list_q_x = []
    for m_th in range(len(self.trajectories)):
      traj = self.trajectories[m_th]
      np_q_x_all = []
      for i_a in range(self.num_agents):
        np_q_x = np.zeros((len(traj), self.tuple_num_latents[i_a]))
        for t in range(len(traj)):
          stt, _, joint_x = traj[t]
          if joint_x[i_a] is not None:
            np_q_x[t, joint_x[i_a]] = 1
          else:
            np_q_x[t, :] = self.cb_bx(i_a, stt)
        np_q_x_all.append(np_q_x)
      list_q_x.append(np_q_x_all)

    list_q_x_xn = []
    if self.cb_Tx is None:
      for m_th in range(len(self.trajectories)):
        traj = self.trajectories[m_th]
        np_q_x_xn_all = []
        for i_a in range(self.num_agents):
          np_q_x_xn = np.zeros((len(traj), self.tuple_num_latents[i_a],
                                self.tuple_num_latents[i_a]))
          for t in range(len(traj) - 1):
            stt, _, joint_x = traj[t]
            sttn, _, joint_x_n = traj[t + 1]
            if joint_x[i_a] is not None:
              if joint_x_n[i_a] is not None:
                np_q_x_xn[t, joint_x[i_a], joint_x_n[i_a]] = 1
              else:
                np_q_x_xn[t, joint_x[i_a], :] = self.cb_bx(i_a, sttn)
            else:
              if joint_x_n[i_a] is not None:
                np_q_x_xn[t, :, joint_x_n[i_a]] = self.cb_bx(i_a, stt)
              else:
                np_q_x_xn[t, :, :] = (self.cb_bx(i_a, stt)[:, None] *
                                      self.cb_bx(i_a, sttn)[None, :])
          np_q_x_xn_all.append(np_q_x_xn)

        list_q_x_xn.append(np_q_x_xn_all)

      num_s = self.num_ostates if self.tx_dependency[0] else None
      list_num_a = []
      for i_a in range(self.num_agents):
        if self.tx_dependency[i_a + 1]:
          list_num_a.append(self.tuple_num_actions[i_a])
        else:
          list_num_a.append(None)

      num_sn = self.num_ostates if self.tx_dependency[-1] else None
      self.list_Tx = []
      for i_a in range(self.num_agents):
        self.list_Tx.append(
            TransitionX(self.tuple_num_latents[i_a], num_s, tuple(list_num_a),
                        num_sn, self.tuple_num_latents[i_a]))

        self.list_Tx[i_a].set_lambda_Tx_prior_param(self.beta_Tx)

    list_lambda_pi = []
    list_lambda_pi = [
        np.full((self.tuple_num_latents[i_a], self.num_ostates,
                 self.tuple_num_actions[i_a]), self.beta_pi)
        for i_a in range(self.num_agents)
    ]

    list_lambda_pi_prev = None

    count = 0
    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      count += 1
      list_lambda_pi_prev = list_lambda_pi

      # start with mstep to make use of labeled data
      list_lambda_pi = self.mstep_global_variables(list_q_x, list_q_x_xn)

      list_pi_tilda = self.get_pi_tilda_from_lambda(list_lambda_pi)

      if self.list_Tx is not None:
        for i_a in range(self.num_agents):
          self.list_Tx[i_a].conv_to_Tx_tilda()

      list_q_x, list_q_x_xn = self.estep_local_variables(list_pi_tilda)

      delta_team = 0
      for i_a in range(self.num_agents):
        delta = np.max(np.abs(list_lambda_pi[i_a] - list_lambda_pi_prev[i_a]))
        delta_team = max(delta_team, delta)

      if delta_team < self.epsilon:
        break
      progress_bar.update()
      progress_bar.set_postfix({'delta': delta_team})
    progress_bar.close()

    for idx in range(self.num_agents):
      numerator = list_lambda_pi[idx]
      action_sums = np.sum(numerator, axis=-1)
      self.list_np_policy[idx] = numerator / action_sums[:, :, np.newaxis]

    if self.list_Tx is not None:
      for i_a in range(self.num_agents):
        self.list_Tx[i_a].conv_to_Tx()


if __name__ == "__main__":
  pass
