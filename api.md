# 1️⃣ Notations

## 🪢 Networks

- $g_r(s, a)$ dymanics function - reward model, predicts $r$
- $g_s(s, a)$ dynamics function - transition model, predicts $s$
- $f_v(s)$ prediction function, predicts $v$
- $f_p(s)$ prediction function, predicts $p$
- $h(o_0, \ldots, o_t)$ representation function, encodes $s_t^0$

## ⏲️ Hyperparameters

- $A$ action space size (_nb: $A$ can be multi-dimensionnal_)
- $O$ state space size (_same_)
- $N$ number of past observations fed to $h$
- $K$ number of future predictions used during training
- $n$ value bootstrap range
- $M$ mini-batch size
- $\gamma$ bootstrap returns discount

## 🪱 Ground truth values

- $a_t$ action at timestep $t$
- $o_t$ observation at timestep $t$
- $u_t$ real reward at timestep $t$

## 🌳 MCTS-predicted values

- $\pi_t$ recommended policy, distribution on the action space
- $\nu_t$ estimated value
- $z_t = \sum_{k=1}^{n-1} \gamma^k u_{t+k} + \gamma^n \nu_{t+n}$ bootstrapped returns

## 📡 Network-predicted values

- $s_t^0$ initial hidden state, calculated by $h$ from $o_0, \ldots, o_t$
- $s_t^k$ hidden state, predicted by $g_s$ from $s_t^{k-1}$ and $a_t^k$
- $r_t^k$ predicted reward, predicted by $g_r$ from $s_t^{k-1}$ and $a_t^k$
- $v_t^k$ predicted value, predicted by $f_v$ from $s_t^k$
- $\mathbf p_t^k$ predicted policy, predicted by $f_p$ from $s_t^k$

# 2️⃣ Building pieces

## 🧑‍🔬 Planning

```python
def planning(h, g_r, g_s, f_v, f_p, o: Tensor, mask: Tensor) -> float, Tensor:
```

### Inputs

- `h, g_r, g_s, f_v, f_p` the five networks
- `o` list of the last observations, of size $N \times S$
- `mask` mask of allowed actions, of size $A$, in which 1 means allowed and 0 means forbidden

### Outputs

- `float` $\nu_t$ estimated value
- `Tensor` $\pi_t$ recommended policy, discrete distribution over the actions, of size $O$

## 💃 Acting

```python
def acting(env: Environment) -> List[tuple[Tensor, Tensor, int, float, float]]:
```

### Inputs

- `env` game environment, class with the following methods:
  - `initial_state() -> state` where
    - `state: Tensor` of size $O$
  - `step(action) -> reward, state, is_terminal` where
    - `reward: float`
    - `state: Tensor` of size $O$
    - `is_terminal: bool`

### Outputs

- episode: list of tuples made of
  - `Tensor` observation, of size $O$
  - `Tensor` policy, of size $A$
  - `int` action
  - `float` reward
  - `float` value

## 🧑‍🏫 Training

```python
def training(
    observations: Tensor,
    target_policies: Tensor,
    target_actions: Tensor,
    target_returns: Tensor,
    h, g_r, g_s, f_v, f_p
) -> float:
```

### Inputs

- `observations` passed observations $o_{t-N+1}, \ldots, o_{t}$, of size $M \times N \times O$
- `target_policies` policy used by the MCTS during acting $\pi_{t}, \ldots, \pi_{t+K}$, of size $M \times (K+1) \times A$
- `target_actions` actions chosen during acting $a_{t}, \ldots, a_{t+K}$, of size $M \times (K+1)$
- `target_returns` $z_t, \ldots, z_{t+K}$, of size $M \times (K+1)$
- `h, g_r, g_s, f_v, f_p` the five networks

### Outputs

- `float` the mean loss over the mini-batch

# 3️⃣ Assembly

TBD...
