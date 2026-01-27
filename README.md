# SAD2 Final Project — Boolean Network Sampling Analysis

## Project Description

This project investigates **sampling strategies for Boolean Network reconstruction** using **BNFinder2**.
The main goal was to test how different **sampling parameters** influence reconstruction quality and to **identify optimal parameter configurations** under various network settings.

The work was conducted as a **semester project** for a university course; the full course assignment and background materials are available in the [docs/](https://github.com/maxi7524/SAD2_final_project/blob/main/docs/bnfinder_tutorial.pdf) directory.
All experiments and optimization procedures were designed specifically for compatibility with the **BNFinder2** library.

The experiments were conducted by explicitly separating the analysis into distinct groups:
* Synchronous vs. asynchronous Boolean networks, treated as fundamentally different classes of dynamical processes.
These groups were analyzed independently to identify differences in behavior and reconstruction performance across cost functions.
* Network size (number of vertices), to assess scalability and determine how reconstruction quality changes with increasing model complexity.
* Experiment-specific parameters, where each experiment isolates a single parameter (e.g. sampling factor, direction changes, or scoring function) in order to directly evaluate its impact on reconstruction quality.

Reconstruction quality was evaluated using AHD and SID metrics, as they are normalized and provide information about topological and reasoning errors.

### Final Results — Optimal Parameters

| Network type  | Cost function | Network size condition | Optimal sampling factor | Recommended k value |
|---------------|---------------|-------------------------|----------------------------|----------------|
| Synchronous   | —             | N ≤ 5                   | 1                          | 20             |
| Synchronous   | —             | N > 5                   | 1                          | 40             |
| Asynchronous  | BDe           | any                     | 3                          | 100            |
| Asynchronous  | MDL           | any                     | 2                          | 100            |

For all networks recommended length of trajectory is $0.8 \cdot \text{number of nodes}$.

---

## Full Report and Methodology

A detailed description of the **methodology**, **experimental setup**, and **result analysis** is available in the following notebook:

* [`notebooks/notebook_report.ipynb`](https://github.com/maxi7524/SAD2_final_project/blob/main/notebooks/notebook_report.ipynb)
### Experimental Data

The datasets used in the experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1fbIJiBIGMP9HivtoPEkko7hT6t2mXH0n):

Alternatively, you can use the provided script to download and unpack the data automatically:

* [`scripts/download_and_unzip_report_data.sh`](https://github.com/maxi7524/SAD2_final_project/blob/main/scripts/download_and_unzip_report_data.sh)

---

## Authors

The project was developed by:

* [**Michał Chmura**](https://github.com/Chmuradin)
* [**Joanna Huba**](https://github.com/joannahuba)
* [**Jan Szot**](https://github.com/jszot)
* [**Max Stróżyk**](https://github.com/maxi7524)

---

## Usage

### Generating Datasets

If you want to generate datasets with metrics for your own experiments, use:

* `notebooks/notebook_datasets.ipynb`

This notebook allows you to reproduce or modify the data generation process used in the experiments.

### Validating Your Own Boolean Network

To test your own Boolean network model:

1. Place the model in `.bnet` format inside the `download_models/` directory.
2. Follow the directory structure:

   ```
   download_models/model_id/model_bnet.bnet
   ```
3. Open and run:

   * [`notebooks/tutorial_validate_model_example.ipynb`](https://github.com/maxi7524/SAD2_final_project/blob/main/notebooks/tutorial_validate_model_example.ipynb)

The notebook provides a ready-to-use validation script; only the model file needs to be supplied.

### Reproducing Report Results

To fully reproduce the results presented in the report:

1. Configure the environment using **uv** (see Setup section below).
2. Download the experimental datasets [section-link](#Experimental-Data).
3. Run the notebook follwing notebook - [notebooks/notebook_report.ipynb](https://github.com/maxi7524/SAD2_final_project/blob/main/notebooks/notebook_report.ipynb).

Configuration of python2 is not required.

---

## Setup up Repository

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

#### Step 1 — Run script for configuration

The script installs all required dependencies and links the local library in production mode.

```bash
bash run.sh
```

---

### Python 2 (Required for BNFinder2)

To install Python 2 locally, run the following script **inside the repository folder**.
This process may take **20–30 minutes**.

```bash
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
python2 -m pip install BNfinder
rm Python-2.7.18.tgz
which bnf
```

More detailed description can be find [here](https://github.com/maxi7524/SAD2_final_project/blob/main/task3_readme.md)

#### Troubleshooting

| Error                               | Solution                                                        |
| ----------------------------------- | --------------------------------------------------------------- |
| `"Could not find 'bnf' executable"` | Ensure BNFinder2 is available in your PATH                      |
| `"ValueError: '1' is not in list"`  | Input formatting issue — wrapper auto-fixes with `#default 0 1` |

---

## Relevant File Structure

| Path                                 | Purpose                                                      |
| ------------------------------------ | ------------------------------------------------------------ |
| `src/`                               | Main library source code                                     |
| `src/sad2_final_project/`            | Core project implementation                                  |
| `src/sad2_final_project/boolean_bn/` | Boolean Network (`BN`) class                                 |
| `src/sad2_final_project/bnfinder/`   | BNFinder2 wrapper and interface                              |
| `scripts/`                           | Utility and helper scripts                                   |
| `data/`                              | Experimental datasets (`experiment_name/configuration_name`) |
| `docs/`                              | BNFinder documentation and course materials                  |
| `notebooks/`                         | Jupyter notebooks (report and tutorials)                     |
| `download_models/`                   | Imported biological models (`.bnet` format)                  |
| `run.sh`                             | Dependency installation script                               |

---

## References / Resources

### BNFinder2

* Official website:
  [https://bioputer.mimuw.edu.pl/software/bnf/](https://bioputer.mimuw.edu.pl/software/bnf/)

* Tutorial:
  [https://bioputer.mimuw.edu.pl/software/bnf/bnfinder_tutorial.pdf](https://bioputer.mimuw.edu.pl/software/bnf/bnfinder_tutorial.pdf)

* User manual:
  [https://bioputer.mimuw.edu.pl/software/bnf/bnfinder_manual.pdf](https://bioputer.mimuw.edu.pl/software/bnf/bnfinder_manual.pdf)

> **Note:** BNFinder2 requires **Python 2** and is not compatible with Python 3.

### Biodivine Boolean Models Repository

* [https://github.com/sybila/biodivine-boolean-models](https://github.com/sybila/biodivine-boolean-models)

---
