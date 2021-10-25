# Review Classification

Source code to finetune a transformer model for review classification. For the data cleaning process and model validation see [this colab](https://colab.research.google.com/drive/1MxW5e0By5cjG4BgzREOriPFONGTProlY?usp=sharing)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to setup the requirements.

```bash
pip install requirements.txt
```

## Usage
Train with default parameters. Edit configs/default.yaml or use  [hydra](https://hydra.cc/) syntax to add, remove and edit parameters thorugh the command line
```bash
python train.py
```

Run a hyperparameter tunning sweep following configs/sweep.yml configuration
```bash
wandb sweep configs/sweep.yml
wandb agent AGENTID
```
