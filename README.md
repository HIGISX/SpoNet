# Deep Reinforcement Learning for Solveing the Maximum Covering Location Problem
This project is based on the research of [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm) which has been accepted at ICLR 2019.

## Dependencies
All experiments were implemented python and run on two RTX3090 GPUs and Intel(R) Xeon(R) Gold 6230 CPU @2.10GHz. 

* Python==3.7
* pytorch=1.11.0+cuda11.3
* SciPy
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

## Quick start
python run.py --problem MCLP --graph_size 20 --baseline rollout --run_name 'MCLP20_rollout'

## Usage
### Training
For training MCLP instances with 50 nodes and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 50 --baseline rollout --run_name 'MCLP50_rollout'
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option:
```bash
python run.py --graph_size 100 --load_path pretrained/tsp_100/epoch-99.pt
```

The `--load_path` option can also be used to load an earlier run, in which case also the optimizer state will be loaded:
```bash
python run.py --graph_size 20 --load_path 'outputs/tsp_20/tsp20_rollout_{datetime}/epoch-0.pt'
```

The `--resume` option can be used instead of the `--load_path` option, which will try to resume the run, e.g. load additionally the baseline state, set the current epoch/step counter and set the random number generator state.

### Example MCLP solution
gurobi solver slove MCLP instances
```bash
python gurobi_MCLP 
```
![MCLP100](images/MCLP.png)
