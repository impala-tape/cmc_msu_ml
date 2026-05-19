# COMMANDS.md

## Python Environment Setup

### Create virtual environment
```bash
cd /home/nikita/Code/MSU
uv venv .venv --python 3.11
```
- `uv venv` — create virtual environment using uv package manager
- `.venv` — path where venv is created
- `--python 3.11` — specify Python version

### Activate virtual environment
```bash
source /home/nikita/Code/MSU/.venv/bin/activate
```
- `source` — execute shell script in current session
- `.venv/bin/activate` — path to activation script

### Install dependencies
```bash
cd /home/nikita/Code/MSU
uv pip install -r ML/Base_12_Clusterization/requirements_2025_26_for_colab_small.txt
```
- `uv pip install` — install packages using uv
- `-r` — read requirements from file
- Path to requirements file with all pinned versions

### Install Jupyter support
```bash
cd /home/nikita/Code/MSU
uv pip install jupyter ipykernel
```
- `jupyter` — Jupyter notebook server
- `ipykernel` — IPython kernel for Jupyter

### Register IPython kernel
```bash
cd /home/nikita/Code/MSU
source .venv/bin/activate
python -m ipykernel install --user --name=msu --display-name="MSU Python"
```
- `python -m ipykernel install` — register kernel
- `--user` — install for current user only
- `--name=msu` — internal kernel name
- `--display-name="MSU Python"` — name shown in Jupyter UI

## Running the Notebook

### Start Jupyter
```bash
cd /home/nikita/Code/MSU
source .venv/bin/activate
jupyter notebook ML/Base_12_Clusterization/[Base]_Clusterization_[2025_2026].ipynb
```
- `jupyter notebook` — launch Jupyter notebook server
- Path to the notebook file
