# Proximal Policy Optimization applied to Super Mario Bros
## How to start
### Step. 1 prepare an environment (e.g. venv)
```bash
python3 -m venv myenv
source myenv/bin/activate
pip3 install requirements.txt
```

### Step. 2 train for stage 1-1
```bash
python3 train.py --output test1-1 --world 1 --stage 1
```
During the training, you can monitor tensorboard:
```bash
tensorboard --logdir ./experiments/test1-1/runs
```


### Step. 3 inference
```bash
python3 test.py --ckpt ./experiments/test1-1/checkpoints/best_model.ckpt
```
Then, a display pops up, and you can watch playing mario by the agent.
