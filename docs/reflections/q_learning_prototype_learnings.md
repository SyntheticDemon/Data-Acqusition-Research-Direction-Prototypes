# Reflections on the Adult Q-Learning Prototype

Personal notes from building and interrogating `adult_prototype_q_learning.py`.
Written to remember *why* the current tabular prototype is not a good path to
carry forward, and *why* a continuous (function-approximation) Q-learning
approach may be the natural next step.

---

## What we were trying to do

We framed sequential data acquisition as a Markov decision process:

- Start with a small labeled training set (100 rows).
- Choose repeatedly among **Acquire**, **Retrain**, and **Evaluate** under hard
  budgets (acquire / retrain / eval).
- **Acquire** adds a batch to a pending queue (batch choice via preview entropy).
- **Retrain** merges pending batches into the training set and refits the classifier.
- **Evaluate** measures AUROC on a labeled evaluation set but does not change the model.
- Episode ends when no actions remain (we removed **Stop**; budgets drive termination).
- **Reward** is sparse: hidden-validation AUROC on the final model, applied at episode end.

We used **tabular Q-learning**: a Python dict `q_values[(state, action)]`,
updated online with the Bellman equation after each step, shared across many
training episodes, then one greedy final episode for deployment.

---

## What we actually learned about how the code works

### Episodes are not different datasets

Each training episode **resets** the world (model, budgets, pool shuffle) but
**keeps the same Q-table**. Episodes are repeated simulations on the same 1,000-row
split, not new train/test partitions.

### The Q-table is not a simple 1D “reward per action” vector

We initially imagined something like:

```
Q = [Q(Acquire), Q(Retrain), Q(Evaluate)]
```

That is a **bandit**, not what the code implements.

The real object is:

```
Q(state, action) → expected long-term return
```

where `state` is an **11-dimensional discretized tuple** (training size, pending
batches, remaining pool, three budgets, eval AUC bin, counters since retrain/eval,
mean/max batch utility bins).

The policy at each step: build state → list feasible actions → pick
`argmax Q(s, a)` (or ε-greedy during training).

### Learning is online, but the environment is episodic

Q-values update **after every step** during an episode (true online TD learning).
The episode loop only resets the **environment**; the **Q-table persists**. This
is standard episodic Q-learning, not offline batch fitting.

### Evaluate does not affect the outcome we care about

The terminal return is **hidden-validation AUROC of the final model**.

**Evaluate**:

- does not retrain,
- does not add data,
- does not enter the reward,

It only writes an optional number into state (`latest_evaluation_auc`). The Q-table
*can* be built without ever using Evaluate; states where eval AUC appears are extra
rows, not the core learning signal. For the research question “what model do we end
up with?”, Evaluate is logically inert.

### Observed symptoms of a weak learner

In practice (including short smoke runs):

- Many Q-values stay at **0.0** (unvisited or barely updated `(s, a)` pairs).
- Training episodes often collapse to **very short trajectories** early on.
- Perceived Q at decision time can be **far from** the actual terminal reward.
- Final policy quality is often dominated by **whether acquisition/retrain happened
  at all**, not by a finely learned timing policy.

---

## Why we cannot continue this approach (tabular prototype Q-learning)

### 1. State space is far too large for a table

The cross-product of discretized state components is enormous in theory. Tabular
Q-learning only works when the agent **revisits the same `(state, action)` many
times** so estimates converge.

Here:

- Pool shuffle and preview sampling change batch utilities every episode → states
  rarely repeat exactly.
- Only a tiny fraction of the logical table is ever filled (`len(q_values)` stays
  small relative to the full space).
- Most decisions are made with Q = 0 for all feasible actions → ties broken
  randomly among “best” actions.

We are carrying a **high-dimensional MDP** with a **lookup table** — a known mismatch.

### 2. Sparse, delayed reward makes credit assignment brutal

Non-terminal steps have reward 0. The only strong signal is terminal AUROC on the
last transition when budgets are exhausted. That signal must propagate backward
through a chain of unique states. With a sparse table and little state repetition,
Bellman backups barely move Q-values in a coherent direction.

### 3. The prototype conflates “research framing” with “learning that works”

The code successfully **instantiates** the MDP story (budgets, actions, Bellman
updates, final evaluation). It does **not** reliably **learn** a good sequential
policy. Continuing to tune episode counts or ε schedules on this tabular setup is
unlikely to fix the structural problem.

### 4. Evaluate (and much of the state) adds complexity without improving the objective

Keeping Evaluate inflates the action space and state space for information that
does not change the final model. Keeping batch-utility bins in state adds dimensions
that change every episode. Both work against tabular learning without a clear
benefit for terminal model quality.

### 5. What “continuing this approach” would mean

Continuing means:

- more episodes on a table that mostly stays empty,
- debugging policies that are often random among Q = 0,
- reporting results that may not beat simple heuristics (“retrain every k acquires”),

without addressing representation. That is poor return on effort.

---

## Why we might have to switch to continuous Q-learning

“Continuous Q-learning” here means **function approximation**: replace

```python
q_values[(state, action)]  # discrete lookup
```

with something like

```python
Q(state, action; θ)  # parameterized, generalizes across similar states
```

e.g. linear weights over features, or a small neural network (DQN-style).

### Why tabular → continuous is the natural upgrade

| Problem in tabular setup | What continuous Q addresses |
|---|---|
| States almost never repeat | Similar states share parameters; learning generalizes |
| 11-D discrete tuple is coarse and huge | Real-valued features without hand-tuned bins |
| New situation → Q = 0 | Network can interpolate from seen situations |
| Pool shuffle changes utilities every time | Features can include utilities as continuous inputs |

We are not abandoning Q-learning as an idea. We are abandoning **storing every
(state, action) in a dict** as the representation.

### What we would still need to get right

Function approximation does not magically solve everything:

- **Reward design** may still need intermediate signal (e.g. delta in validation
  AUC after retrain) if terminal reward alone is too sparse.
- **State design** should shrink to what actually matters: pending batches,
  budgets left, training size, maybe recent batch scores — not every bin we added
  for tabular convenience.
- **Evaluate** should probably be dropped unless the research question explicitly
  requires mid-episode observable performance.
- **Baselines** remain essential: fixed acquire/retrain schedules should be beaten
  before claiming the RL layer adds value.

### Alternative paths (if continuous Q still feels heavy)

Not everything requires neural Q-learning:

1. **Fixed or searched schedules** — acquire/retrain rhythm as parameters to search.
2. **Tiny tabular Q** — 3–4 state features only; honest attempt at inspectable RL.
3. **RL for batch choice only** — Thompson / entropy under fixed retrain rules.
4. **Policy search** — direct optimization over a small policy class.

Continuous Q-learning is the path if we want to **keep the full sequential MDP**
and **rich state**, but need a representation that can actually learn from limited
episodes.

---

## Summary judgment

| Question | Answer |
|---|---|
| Did the prototype clarify the problem? | **Yes** — MDP, budgets, delayed reward, episode structure. |
| Does tabular Q-learning work here? | **Not convincingly** — state too large, table too sparse, reward too delayed. |
| Should we keep iterating on this exact code? | **No** — structural limits, not hyperparameter limits. |
| What is the likely next step? | **Simplify actions/state**, add **strong baselines**, then **continuous Q**
  (or schedule search) if we still want learned timing. |

---

## Open questions to carry forward

1. What is the **minimal state** that captures “should I retrain now?”
2. Can we add **intermediate reward** (e.g. post-retrain validation delta) without
   breaking the cost-constrained story?
3. What is the **simplest baseline** that must be beaten before any RL claim?
4. Is the research contribution **when to acquire/retrain**, **which batch**, or
   both — and should the method split those two levels?

---

*Last updated from design discussions on the Adult income acquisition prototype.*
