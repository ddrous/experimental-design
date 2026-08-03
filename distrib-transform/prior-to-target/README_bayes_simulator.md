# AdaLN Bayesian simulator and sequential design policy

Place these three Python files in the same directory:

1. `bayes_simulator_common.py` — shared simulator, loaders, Equinox modules, permutation-symmetric NLL, and checkpoint utilities.
2. `adaptive_posterior_particle_transformer.py` — notebook-style (`#%%`) posterior-density training script.
3. `train_design_policy.py` — notebook-style (`#%%`) downstream sequential design-policy script.

Run the posterior script first. It creates a timestamped directory under `./runs/posterior_adaln_*` and saves `artefacts/model_best.eqx`. Then run the policy script; by default it loads the newest posterior run.

The posterior model learns

```
q_phi(theta | current_belief_particles, x, y)
```

as a diagonal Gaussian mixture using conditional NLL. For `K>1`, the loader randomly permutes source labels and the loss marginalises over source permutations.

The policy recursively performs

```
current belief -> design x_t -> simulated y_t -> frozen posterior update -> next belief
```

and optimises final posterior NLL after the design budget.

Suggested packages:

```bash
pip install -U jax equinox optax torch numpy matplotlib tqdm pyyaml ipython
```

Choose the JAX installation appropriate for CPU/CUDA/TPU.
