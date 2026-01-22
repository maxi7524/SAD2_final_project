# Inference of Boolean Networks via Dynamic Bayesian Networks

---

## Initialization

To use this project after forking the repository, follow these steps:

### Step 0 — Install `uv`

[`uv`](https://github.com/astral-sh/uv) is a fast Python package and environment manager.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 1 — Synchronize dependencies

```bash
uv sync
```

### Step 2 — Activate environment

```bash
source .venv/bin/activate
```

### Step 3 — Install the project in development mode

```bash
uv pip install -e .
```

### Step 4 — Run a script

```bash
uv run python path/to/script.py
```

---

## 1. Project Description

The purpose of this project is to investigate how the type and amount of data describing network dynamics influence the accuracy of reconstructing network structures within the framework of **dynamic Bayesian networks (DBNs)**.

In **Part I**, synthetic Boolean networks of varying sizes are constructed with randomly generated Boolean functions and limited parent dependencies. These networks are simulated under both synchronous and asynchronous update modes to generate trajectory datasets. The datasets vary in trajectory length, sampling frequency, and the proportion of attractor versus transient states, enabling a systematic study of factors affecting network reconstruction.

Dynamic Bayesian networks are inferred from the simulated datasets using the **BNFinder2** software tool, employing the **Minimal Description Length (MDL)** and **Bayesian–Dirichlet equivalence (BDe)** scoring functions. The reconstructed networks are then evaluated against the original Boolean networks using graph-based distance metrics, providing a quantitative assessment of reconstruction accuracy.

In **Part II**, the methodology and insights gained from synthetic experiments are applied to reconstruct the structure of a validated biological network model selected from the **Biodivine** repository, demonstrating the practical relevance of the approach for real biological systems.

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

## 3. Installation / Setup

This project uses a **wrapper architecture** integrating **Python 3** (project code) and **Python 2** (BNFinder2).

---

### Python 3 Environment (Main Project Code)

**Dependencies:**

| Package      | Purpose              |
| ------------ | -------------------- |
| `pandas`     | Data handling        |
| `numpy`      | Numerical operations |
| `matplotlib` | Visualization        |
| `networkx`   | Graph representation |

**Setup:**

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib networkx
```

---

### Python 2.7 Environment (BNFinder2)

BNFinder2 requires Python 2.7.

```bash
# Download and install Python 2.7.18
wget https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
tar -xvf Python-2.7.18.tgz
cd Python-2.7.18
./configure --enable-optimizations
make
sudo make install

# Verify
python2 --version

# Install pip
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
python2 get-pip.py

# Install dependencies
python2 -m pip install numpy==1.16.6 scipy==1.2.3 fpconst BNfinder

# Clean up
rm -rf Python-2.7.18.tgz
which bnf
```

> ⚠️ **Note:** BNFinder2 is not compatible with Python 3. The wrapper bridges both environments.

---

### Optional: Using `uv` (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# Windows:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv sync
source .venv/bin/activate
uv pip install -e .
```

Run scripts using:

```bash
uv run python path/to/script.py
```

---

### Troubleshooting

| Error                               | Solution                                                        |
| ----------------------------------- | --------------------------------------------------------------- |
| `"Could not find 'bnf' executable"` | Ensure BNFinder2 is in your PATH                                |
| `"ValueError: '1' is not in list"`  | Input formatting issue — wrapper auto-fixes with `#default 0 1` |

---

## Relevant File Structure

| Path                                 | Purpose                                              |
|--------------------------------------|------------------------------------------------------|
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

## 4. Usage

You can:

* Create Boolean networks using the `BN` class.
* Simulate trajectories and retrieve attractors.
* Save ground-truth networks to CSV.
* Visualize state transition systems.
* Load `.bnet` models using `load_bnet_to_BN`.

*(Optional: Add example code snippets or screenshots here.)*

---

## 5. Project Workflow / Methodology

The project follows a structured experimental pipeline consisting of two main parts.

---

### **Part I — Synthetic Boolean Networks**

#### 1. Network Construction

Boolean networks are generated with configurable sizes and random Boolean functions. Each node is restricted to a small number of parent nodes (typically 1–3) to reflect realistic biological dependencies.

#### 2. Trajectory Simulation

Networks are simulated under both synchronous and asynchronous update modes. Trajectories originate from random initial states and evolve according to the network’s Boolean functions. Sampling frequency, trajectory length, and the ratio of attractor to transient states are varied to create diverse datasets.

#### 3. Dataset Creation

Multiple trajectories are combined into datasets suitable for Bayesian network reconstruction. Each dataset is registered, and only information indicating whether it satisfies the specified attractor-to-transient ratio criteria is recorded, ensuring experimental consistency without discarding any datasets.

#### 4. Bayesian Network Reconstruction

Dynamic Bayesian networks are inferred using BNFinder2. Two scoring functions are applied: **Minimal Description Length (MDL)** and **Bayesian–Dirichlet equivalence (BDe)**.

#### 5. Evaluation

Reconstructed networks are compared to ground truth using graph-based metrics, including:

| Metric                           | Purpose                  |
| -------------------------------- | ------------------------ |
| True Positives / False Positives | Structural accuracy      |
| Precision / Recall               | Reconstruction quality   |
| Average Hamming Distance         | Structural dissimilarity |

---

### **Part II — Real Biological Networks**

#### 1. Model Selection

A validated Boolean network model is selected from the Biodivine repository.

#### 2. Dataset Generation and BN Inference

Using insights from Part I, suitable datasets are generated and BNFinder2 is applied to reconstruct the network.

#### 3. Evaluation

The inferred network is compared to the known biological network using the same evaluation metrics.

---

### **Experimental Design & Parameter Variation**

| Parameter                     | Variations                 |
| ----------------------------- | -------------------------- |
| Number of nodes               | 5–16                       |
| Update mode                   | Synchronous / Asynchronous |
| Trajectory length             | Variable                   |
| Number of trajectories        | Variable                   |
| Sampling frequency            | Variable                   |
| Attractor vs. transient ratio | Controlled                 |

Experiments are repeated across multiple random seeds to ensure statistical robustness. All steps are automated and support batch and parallel execution.

---

## 6. Evaluation / Results

This section summarizes:

* Evaluation methodology (graph distance metrics, comparison with ground truth).
* Key findings and insights.
* Optionally: example plots, tables, or figures.

*(You may link to your full report or include summary figures here.)*

---

## 7. Project Report

The full project report includes:

* Detailed experimental setup.
* Generated datasets.
* Methodological justification.
* Results and visualizations.
* Individual contributions (for group work).

*(Insert link or file path here.)*

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

