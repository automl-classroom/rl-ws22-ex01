# rl-ws22-ex01
RL Training Demo with Popular Libraries.

We train and optimize a BipedalWalker.

For more info see the notebook.

## Setup
- Install the open-source-distribution [anaconda](https://www.anaconda.com/products/individual).
- Create a [new environment and activate it](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
- Inside this environment, install pip with `conda install pip`.
- Install requirements with `pip install -r requirements.txt`.
- Open jupyter notebook with `jupyter-lab` in this directory (activate your conda env first).

## Train
For training, activate your conda env and run the following in this dir:
```bash
python train.py
```

### Visualize Training Progress
```bash
tensorboard --logdir .
```

## Replay
See the notebook.
