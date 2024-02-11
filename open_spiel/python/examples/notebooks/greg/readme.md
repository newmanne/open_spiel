# Running Experiments
To run experiments:

1. Sample games with `sample_games.ipynb`:
    1. Sample valuations with `greg/sample_games.py`. (Adapted from `2024-01-08-BCSampler.ipynb`.)
    2. Write game config files with `write_configs.py`. (Adapted from `SATSGameSampler.ipynb`.)
2. Launch Slurm jobs with `launch_experiment.ipynb`. (Adapted from `CFRLauncher-Dec11.ipynb`.)
3. After jobs finish, parse results with `parse_results.ipynb`.
4. Make plots with `plot_comparative_statics.py`. (TODO: clean this up to make plots a bit more reproducible.)
5. Analyze a single run with `analyze_run.ipynb`.

To generate configs for PPO hyperparameter tuning, see `sample_ppo_configs.ipynb`.
