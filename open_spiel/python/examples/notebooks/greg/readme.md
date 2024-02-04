# Running Experiments
To generate experiment plots:

1. Sample games with `2024-01-18-SampleGames.ipynb`:
    1. Sample valuations with `greg/sample_games.py`. (Adapted from `2024-01-08-BCSampler.ipynb`.)
    2. Write game config files with `write_configs.py`. (Adapted from `SATSGameSampler.ipynb`.)
3. Launch Slurm jobs with `CFRLauncher-Dec11.ipynb`. (TODO: clean this up.)
4. After jobs finish, parse results with `parse_results.ipynb`.
5. Make plots with `plot_comparative_statics.py`.



