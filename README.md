# Inference of Boolean Networks via Dynamic Bayesian Networks

---

## 1. Project Description

The purpose of this project is to investigate how sampling parameters affects training and what are optimal values. 


The purpose of this project is to investigate how sampling parameters influence network dynamics influence the accuracy of reconstructing network structures within the framework of **dynamic Bayesian networks (DBNs)**.

In **Part I**, we tried to find optimal data sampling 

synthetic Boolean networks of varying sizes are constructed with randomly generated Boolean functions and limited parent dependencies. These networks are simulated under both synchronous and asynchronous update modes to generate trajectory datasets. The datasets vary in trajectory length, sampling frequency, and the proportion of attractor versus transient states, enabling a systematic study of factors affecting network reconstruction.

Dynamic Bayesian networks are inferred from the simulated datasets using the **BNFinder2** software tool, employing the **Minimal Description Length (MDL)** and **Bayesian–Dirichlet equivalence (BDe)** scoring functions. The reconstructed networks are then evaluated against the original Boolean networks using graph-based distance metrics, providing a quantitative assessment of reconstruction accuracy.

In **Part II**, the methodology and insights gained from synthetic experiments are applied to reconstruct the structure of a validated biological network model selected from the **Biodivine** repository, demonstrating the practical relevance of the approach for real biological systems.



### Results 


## 2. Usage

You can:
* Create Boolean networks using the `BN` class with following functionality:
  - attractor calculation,
  - simulating trajectories, for given length and subsampling parameter 
  - Visualize state transition systems.
  - Save ground-truth networks to CSV.
* Load `.bnet` models using `load_bnet_to_BN`.
* TODO notebook with 

---

## 2. Features / Highlights

| Feature                    | Description                                       |
| -------------------------- | ------------------------------------------------- |
| Boolean network generation | Configurable size and random Boolean functions    |
| Simulation modes           | Synchronous and asynchronous updates              |
| BN inference               | Dynamic Bayesian networks via BNFinder2           |
| Scoring functions          | MDL and BDe                                       |
| Evaluation                 | Graph-based distance metrics                      |
| Visualization              | State transition systems and attractors           |
| File support               | Import/export `.bnet` and save ground-truth edges |

---

## Relevant File Structure

| Path                                 | Purpose                                              |
|--------------------------------------|------------------------------------------------------|
| `src`            | Folder with whole library
| `src/sad2_final_project/`            | Core project source code                             |
| `src/sad2_final_project/boolean_bn/`| Boolean network generation and simulation logic      |
| `src/sad2_final_project/bnfinder/`   | BNFinder2 wrapper and interface                      |
| `scripts/`                           | Utility scripts for BNFinder and setup               |
| `data/`                             | Generated datasets, ground truth, and results        |
| `docs/`                             | BNFinder documentation and references                |
| `notebooks/`                        | Jupyter notebooks for experiments and reports        |
| `download_models/`                  | Imported biological models (`.bnet` files)           |
| `task3.py`                          | Main experimental controller script                  |
| `run.sh`                            | Project execution helper script                      |
| `pyproject.toml`                    | Project configuration and dependencies               |
| `README.md`                         | Project documentation                                |


--------------------------------------------------------------------- TO JEST KONIEC

## Setup up repository

To use this project after forking the repository, follow these steps:

### Python 3

#### Step 0 — Install `uv`

[`uv`](https://github.com/astral-sh/uv) is a fast Python package and environment manager.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Step 1 — Synchronize dependencies

```bash
uv sync
```

#### Step 2 — Activate environment

```bash
source .venv/bin/activate
```

#### Step 3 — Install the project in development mode

```bash
uv pip install -e .
```

#### Step 4 — Run a script

```bash
uv run python path/to/script.py
```

### Python 2

To install python2 you need to run following script **in repository folder**, it can take 20-30 minutes.

```shell
wget https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
tar -xvf Python-2.7.18.tgz
cd Python-2.7.18
./configure --enable-optimizations
make
sudo make install
python2 --version
python2 -m pip --version
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
python2 get-pip.py
python2 -m pip --version
python2 -m pip install "numpy==1.16.6"
python2 -m pip install fpconst
python2 -m pip install "scipy==1.2.3"
python2 pip install BNfinder
python2 -m pip install BNfinder
rm Python-2.7.18.tgz 
which bnf
```

#### Troubleshooting

| Error                               | Solution                                                        |
| ----------------------------------- | --------------------------------------------------------------- |
| `"Could not find 'bnf' executable"` | Ensure BNFinder2 is in your PATH                                |
| `"ValueError: '1' is not in list"`  | Input formatting issue — wrapper auto-fixes with `#default 0 1` |

---

## 8. References / Resources

### **BNFinder2**

* Official website:
  [https://bioputer.mimuw.edu.pl/software/bnf/](https://bioputer.mimuw.edu.pl/software/bnf/)
* Tutorial:
  [https://bioputer.mimuw.edu.pl/software/bnf/bnfinder_tutorial.pdf](https://bioputer.mimuw.edu.pl/software/bnf/bnfinder_tutorial.pdf)
* User Manual:
  [https://bioputer.mimuw.edu.pl/software/bnf/bnfinder_manual.pdf](https://bioputer.mimuw.edu.pl/software/bnf/bnfinder_manual.pdf)
  *(Note: BNFinder2 requires Python 2 and is not compatible with Python 3.)*

### **Biodivine Boolean Models Repository**

* [https://github.com/sybila/biodivine-boolean-models](https://github.com/sybila/biodivine-boolean-models)

