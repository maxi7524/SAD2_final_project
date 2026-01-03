## Prerequisites

This solution uses a **Wrapper Architecture** to mix Python 3 (Project Code) and Python 2 (BNFinder).

### 1. Python 3 Environment

Your main environment where the project logic runs.

* **Dependencies:** `pandas`, `numpy` (install via `pip` or `uv`).

### 2. Python 2.7 Environment (Legacy)

To even be able to run task3.py, you need to have python2 installed
It's old, so I did the following to install it

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
which bnf
```

## File Structure

* **`task3.py`**: The **Main Controller**. It loads external CSV data, orchestrates the inference process, and calculates evaluation metrics (Precision/Recall).
* **`bnfinder_wrapper.py`**: The **Driver Module**. It handles the low-level formatting of input files (`.txt`), finds the `bnf` executable on your system, and manages the subprocess calls to Python 2.

## Usage

### 1. Prepare Data

Ensure you have your simulation data ready. By default, `task3.py` looks for:

* `simulation_output.csv`: The time-series data (Columns = Genes, Rows = Time Steps).
* `ground_truth_edges.csv`: (Optional) The true edges for evaluation (Format: `Parent,Child`).

### 2. Run the Script

Execute the controller using your Python 3 interpreter:

```bash
# Standard Python
python3 task3.py

# Or using uv
uv run python3 task3.py
```

## Input & Output Formats

### Inputs

* **External Data (`.csv`)**: A standard CSV file containing Boolean values (0/1).
  ```csv
  G1,G2,G3
  0,0,1
  1,0,0
  ...
  ```

* **BNFinder Input (`.txt`)**: The wrapper automatically converts the CSV to a specific BNFinder format containing the `#default 0 1` preamble and `Series:Time` headers required for DBN inference.

### Outputs

* **Console Output**: Displays the inference progress and a summary of evaluation metrics (True Positives, False Positives, Precision, Recall).
* **SIF Files (`.sif`)**: The raw inferred networks saved to disk.
  * `inferred_network_MDL.sif`
  * `inferred_network_BDE.sif`

## Troubleshooting

**"Could not find 'bnf' executable"**

* The script attempts to find `bnf` in your `$PATH` or in `~/.local/bin/`.
* If installed elsewhere, ensure the folder containing the `bnf` script is in your system PATH.

**"ValueError: '1' is not in list" (BNFinder Error)**

* This error occurs if the input file preamble is incorrect.
* The wrapper handles this automatically by writing `#default 0 1` at the top of the input file, defining the data as Boolean.