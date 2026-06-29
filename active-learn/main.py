#%%
import jax
import os
from config import Config
from dataloaders import OracleFamily
from training import train_policy
from evaluation import run_experiments
from visualisation import plot_learning_curves


#%%
# def main():
print("Initializing environment and configurations...")
cfg = Config()

# Anchor seed for reproducible pilot
key = jax.random.PRNGKey(cfg.seed)
key, train_key, test_oracle_key, eval_key = jax.random.split(key, 4)

print("Building Oracle family...")
oracle_family = OracleFamily(
    x_dim=cfg.surrogate.in_dim,
    y_dim=cfg.surrogate.out_dim,
    hidden_dim=64,
    n_hidden=2,
    x_min=cfg.x_min,
    x_max=cfg.x_max
)

#%%

print(f"\n--- Phase 1: Policy Training ---")
print(f"Training AL-BED design policy over {cfg.training.n_episodes} episodes...")
policy, train_losses = train_policy(cfg, oracle_family, train_key)
print("Policy training successfully resolved.")

#%%
print(f"\n--- Phase 2: Experimental Evaluation ---")
print(f"Sampling testing Oracle architecture...")
oracle_params = oracle_family.sample(test_oracle_key)

print(f"Generating performance baseline across {cfg.training.n_test_seeds} seeded episodes...")
results = run_experiments(cfg, oracle_family, oracle_params, policy, eval_key)

print(f"\n--- Phase 3: Analytics & Export ---")
plot_learning_curves(results, cfg)
print(f"Pilot execution successfully completed.")





# if __name__ == "__main__":
#     main()