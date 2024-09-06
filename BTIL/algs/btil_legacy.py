from typing import Optional, Tuple, Callable, Sequence
import numpy as np
from tqdm import tqdm
from scipy.special import digamma, logsumexp, softmax
from .transition_x import TransitionX

T_SAXSeqence = Sequence[Tuple[int, Tuple[int, int], Tuple[int, int]]]


class BTIL_Decen:
  def __init__(
      self,
      trajectories: Sequence[T_SAXSeqence],
      num_states: int,
      tuple_num_latents: Tuple[int, ...],
      tuple_num_actions: Tuple[int, ...],
      trans_x_dependency=(True, True, True, False),  # s, a1, a2, sn
      max_iteration: int = 100,
      epsilon: float = 0.001) -> None:
    '''
      trajectories: list of list of (state, joint action, latent)-tuples
          Please set latent to None if it is unknown.
          Joint action should be a tuple of each agent's action index
      num_states: number of states |S|
      tuple_num_latents: a tuple of the number of latent states of each agent
                        (|X_1|, |X_2|, ..., |X_n|)
      tuple_num_actions: a tuple of the number of actions of each agent
                        (|A_1|, |A_2|, ..., |A_n|)
      trans_x_dependency: a tuple of booleans
          The length should be the same as (num_agents + 2).
          Starting from the left, each boolean value corresponds to a dependency
            of T_x on the previous state, agent1's previous action,
              ..., action_n's previous action, and current state, respectively.
      max_iteration: number of maximum EM iterations
      epsilon: termination condition for convergence
    '''

    assert len(tuple_num_actions) + 2 == len(trans_x_dependency)

    DIRICHLET_PARAM_PI = 3
    self.trajectories = trajectories

    self.beta_pi = DIRICHLET_PARAM_PI
    self.beta_Tx = DIRICHLET_PARAM_PI
    self.num_agents = len(tuple_num_actions)
    self.num_ostates = num_states
    self.tuple_num_latents = tuple_num_latents
    self.tuple_num_actions = tuple_num_actions
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

    list_q_x = []
    list_q_x_xn = []
    for m_th in range(len(self.trajectories)):
      trajectory = self.trajectories[m_th]

      qx_all = []
      q_x_xn_all = []
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
              self.cb_bx(idx_a, stt_p)[idx_xp] *
              list_policy[idx_a][idx_xp, stt_p, joint_a_p[idx_a]])

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
              list_policy[idx_a][idx_x, stt, joint_a[idx_a]].reshape(1, len_x)  # noqa: E501
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
              list_policy[idx_a][idx_xn, stt_n, joint_a_n[idx_a]].reshape(1, len_xn)  # noqa: E501
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
        qx_all.append(q_x)

        if self.cb_Tx is None:
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
                list_policy[idx_a][:, sttn, joint_a_n[idx_a]].reshape(1, n_lat)
              )
            # yapf: enable

          q_x_xn = softmax(log_q_x_xn, axis=(1, 2))
          q_x_xn_all.append(q_x_xn)

      list_q_x.append(qx_all)
      if self.cb_Tx is None:
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
        q_x_xn_all = list_q_x_xn[m_th]
        traj = self.trajectories[m_th]
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
    list_lambda_tx_prev = None

    count = 0
    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      count += 1
      list_lambda_pi_prev = list_lambda_pi
      list_lambda_tx_prev = [
          np.copy(self.list_Tx[idx_a].np_lambda_Tx)
          for idx_a in range(self.num_agents)
      ]

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
        delta = np.max(
            np.abs(self.list_Tx[i_a].np_lambda_Tx - list_lambda_tx_prev[i_a]))
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
