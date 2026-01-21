# python libraries
from ast import Dict
from pathlib import Path
from itertools import product
import pandas as pd
import multiprocessing as mp
from typing import Iterable, Literal, Optional
# sad2 library
from sad2_final_project.boolean_bn import BN, simulate_trajectories_to_csv
from sad2_final_project.bnfinder import run_bnfinder



class BooleanNetworkExperiment:
    """
    Class responsible ONLY for:
    - defining experimental conditions
    - managing paths
    - orchestrating experiment execution
    """

    # =========================
    # INIT
    # =========================
    def __init__(
        self,
        *,
        data_path: str | Path,
        experiment_name: str,

        # core experimental parameters (iterables!)
        num_nodes: Iterable[int],
        update_mode: Iterable[Literal["synchronous", "asynchronous"]],
        trajectory_length: Iterable[int],
        n_trajectories: Iterable[int],
        sampling_frequency: Iterable[int],
        score_functions: Iterable[Literal["MDL", "BDE"]],
        # TODO - add metrics option 

        # BN-specific
        n_parents_per_node: Iterable[list[int]] = ([1, 2, 3],),

        # repetitions / seeds
        n_repetitions: int = 1,

        simulate_trajectories_to_csv_kwargs = {
            # "sampling_frequency": 3,
            "target_attractor_ratio": 0.4,  # Approximate fraction of trajectory in attractor (0-1)
            "tolerance": 0.1,               # Allowed deviation from the calculated entrance step (0-1)
            "max_iter": 50,                 # Maximum attempts to generate a valid state per step before restarting
            "max_trajectory_restarts": 1000  # Maximum number of trajectory restarts allowed
        }


    ):
        self.data_path = Path(data_path)
        self.experiment_name = experiment_name

        self.experiment_root = self.data_path / self.experiment_name

        # define directory structure
        self.paths = {
            "ground_truth": self.experiment_root / "bn_ground_truth",
            "datasets": self.experiment_root / "datasets",
            "datasets_bnfinder": self.experiment_root / "datasets_bnfinder",
            "results": self.experiment_root / "results",
        }

        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)

        # =========================
        # build experimental design
        # =========================
        conditions = list(
            product(
                num_nodes,
                update_mode,
                trajectory_length,
                n_trajectories,
                sampling_frequency,
                score_functions,
                n_parents_per_node,
                range(n_repetitions),
            )
        )

        self.experiment_df = pd.DataFrame(
            conditions,
            columns=[
                "num_nodes",
                "update_mode",
                "trajectory_length",
                "n_trajectories",
                "sampling_frequency",
                "score_function",
                "n_parents_per_node",
                "rep_id",
            ],
        ) 
        # simulating trajectory setting
        self.simulate_trajectories_to_csv_kwargs = simulate_trajectories_to_csv_kwargs

        # stable condition identifier
        self.experiment_df["condition_id"] = (
            self.experiment_df.index.astype(str).str.zfill(4)
        )

    # =========================
    # INSPECTION
    # =========================
    def show_experiment_df(self) -> pd.DataFrame:
        """Return dataframe with all experimental conditions."""
        return self.experiment_df

    def show_paths(self) -> dict:
        """Return dictionary with all important paths."""
        return self.paths

    # =========================
    # CORE EXECUTION LOGIC
    # =========================
    def _run_single_condition(self, row: pd.Series) -> None:
        """
        Executes ONE experimental condition.
        This function is process-safe.
        """

        cid = row["condition_id"]

        # ---------- paths ----------
        gt_path = self.paths["ground_truth"] / f"{cid}.csv"
        dataset_path = self.paths["datasets"] / f"{cid}.csv"
        bnf_dataset_path = self.paths["datasets_bnfinder"] / f"{cid}.txt"
        metrics_path = self.paths["results"] / f"{cid}_metrics.csv"

        # ---------- 1. create BN ----------
        bn = BN(
            mode=row["update_mode"],
            trajectory_length=row["trajectory_length"],
            num_nodes=row["num_nodes"],
            n_parents_per_node=row["n_parents_per_node"],
        )

        # ---------- 2. save ground truth ----------
        bn.save_ground_truth(gt_path)

        # ---------- 3. generate data ----------
        simulate_trajectories_to_csv(
            bn_instance=bn,
            num_trajectories=row["n_trajectories"],
            output_file=dataset_path,
            sampling_frequency=row["sampling_frequency"],
            **self.simulate_trajectories_to_csv_kwargs
        )

        # ---------- 4. inference ----------
        run_bnfinder(
            dataset_path=dataset_path,
            ground_truth_path=gt_path,
            score_functions=[row["score_function"]],
            bnf_file_path=bnf_dataset_path,
            trained_model_name=self.paths["results"] / cid,
            metrics_file=metrics_path,
        )

    # =========================
    # PARALLEL EXECUTION
    # =========================
    def run_experiment(
        self,
        *,
        n_jobs: int = 1,
        subset: Optional[pd.DataFrame] = None,
    ):
        """
        Run experiment.
        - n_jobs = number of parallel processes
        - subset = optional subset of experiment_df
        """

        df = subset if subset is not None else self.experiment_df

        rows = [row for _, row in df.iterrows()]

        if n_jobs == 1:
            for row in rows:
                self._run_single_condition(row)
        else:
            with mp.Pool(processes=n_jobs) as pool:
                pool.map(self._run_single_condition, rows)
