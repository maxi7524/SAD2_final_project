# python libraries
from ast import Dict
from pathlib import Path
from itertools import product
import shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import Iterable, Literal, Optional
# sad2 library
from sad2_final_project.boolean_bn import BN, simulate_trajectories_to_csv
from sad2_final_project.bnfinder import manager_bnfinder



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
        analysis_metrics: Iterable[Literal["TP", "FP", "FN", "precision", "recall", "sensitivity", "AHD"]]=["TP", "FP", "FN", "precision", "recall", "sensitivity", "AHD", "SHD", "EHD", "SID"],
        # TODO LIBRARY: add analysis part fo cost functions
        analysis_score_functions: Iterable[Literal["MDL", "BDE"]] = ["MDL", "BDE"],

        # BN-specific
        n_parents_per_node: Iterable[list[int]] = ([1, 2, 3],),

        # repetitions / seeds
        n_repetitions: int = 1,

        simulate_trajectories_to_csv_kwargs = {
            # "sampling_frequency": 3,
            "target_attractor_ratio": 0.4,  # Approximate fraction of trajectory in attractor (0-1)
            "tolerance": 0.1,               # Allowed deviation from the calculated entrance step (0-1)
            # już nie używamy "max_iter": 50,                 # Maximum attempts to generate a valid state per step before restarting
            # już nie używamy "max_trajectory_restarts": 1000  # Maximum number of trajectory restarts allowed
        }


    ):
        # =========================
        # paths assignments and folder creation
        # =========================
        self.data_path = Path(data_path)
        self.experiment_name = experiment_name

        self.experiment_root = self.data_path / self.experiment_name

        # define directory structure
        self.paths = {
            "ground_truth": self.experiment_root / "bn_ground_truth",
            "datasets": self.experiment_root / "datasets",
            "datasets_bnfinder": self.experiment_root / "datasets_bnfinder",
            "results": self.experiment_root / "results"
        }
        # cleaning old experiment, and create new catalogues
        self._rm_old_experiment()
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

        # =========================
        # other parameters 
        # =========================
        # analysis attribute
        self.analysis_metrics = analysis_metrics
        self.analysis_score_functions = analysis_score_functions
        # simulating trajectory setting
        self.simulate_trajectories_to_csv_kwargs = simulate_trajectories_to_csv_kwargs

        # stable condition identifier
        self.experiment_df["condition_id_name"] = (
            self.experiment_df.index.astype(str).str.zfill(5)
        )
        self.experiment_df["condition_id_num"] = (
            self.experiment_df.index.astype(int)
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
    
    def normalize_sample(self, n: float, in_place: bool = True) -> pd.DataFrame | None:
        """
        Normalize number of trajectories based on trajectory length relative to network size.
        
        Formula: k = trajectory_length * n_trajectories / (num_nodes * n)
        Returns normalized n_trajectories as ceil(n_trajectories / k).
        
        Args:
            n: Coefficient multiplied by num_nodes
            in_place: If True, modifies self.experiment_df in place. If False, returns modified copy.
            
        Returns:
            None if in_place=True, otherwise returns modified DataFrame
        """
        # Vectorized calculation
        k = (self.experiment_df["trajectory_length"] * self.experiment_df["n_trajectories"]) / (self.experiment_df["num_nodes"] * n)
        normalized_values = np.ceil(self.experiment_df["n_trajectories"] / k).astype(int)
        
        if in_place:
            self.experiment_df["n_trajectories"] = normalized_values
        else:
            df_copy = self.experiment_df.copy()
            df_copy["n_trajectories"] = normalized_values
            return df_copy
    
    # =========================
    # File management
    # =========================

    def save_experiment_df_to_csv(self, csv_path: str | Path = None) -> None:
        """Saves experiment df (metadata)"""
        # default path
        if csv_path is None:
            csv_path = self.paths['results'] / 'metadata.csv'
        # saving results
        self.experiment_df.to_csv(csv_path, index=False)
        pass

    def _merge_csv(self, csv_dir: str | Path = None) -> None:
        """
        Inner function
        Joins all outputs from `_run_single_condition`.
        """
        # default path
        if csv_dir is None:
            csv_dir = self.paths['results']

        csv_dir = Path(csv_dir)
        rtn_csv = csv_dir / f'joined_results_{self.experiment_name}.csv'
        # collect of .csv files
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            raise RuntimeError("No CSV files to merge")
        # join pdfs
        df = pd.concat(
            (pd.read_csv(f) for f in csv_files),
            ignore_index=True
        )
        # save results
        df.to_csv(rtn_csv, index=False)
        # remove other .csvs
        for f in csv_files:
            if f != rtn_csv:
                f.unlink()
        pass

    def _rm_old_experiment(self) -> None:
        path = self.experiment_root

        if path.exists() and path.is_dir():
            shutil.rmtree(path)
        else:
            pass

    # =========================
    # CORE EXECUTION LOGIC
    # =========================
    def _run_single_condition(self, row: pd.Series) -> bool:
        """
        Executes ONE experimental condition.
        This function is process-safe.
        """

        cid = row["condition_id_name"]

        # ---------- paths ----------
        gt_path = self.paths["ground_truth"] / f"{cid}.csv"
        dataset_path = self.paths["datasets"] / f"{cid}.csv"
        bnf_dataset_path = self.paths["datasets_bnfinder"] / f"{cid}.txt"
        metrics_path = self.paths["results"] / f"{cid}_metrics.csv"

        # ---------- 1. create BN ----------
        bn = BN(
            mode=row["update_mode"],
            num_nodes=row["num_nodes"],
            n_parents_per_node=row["n_parents_per_node"],
        )

        # ---------- 2. save ground truth ----------
        bn.save_ground_truth(gt_path)

        # ---------- 3. generate data ----------
        # oznaczamy czy wygenerowany dataset powiódł sie sukcesem (potrzebne do zliczania sukcesów i jeżeli brak sukcesu nie próbujemy uruchamiać bnfindera bo brak pliku)
        success, ratio = simulate_trajectories_to_csv(
            bn_instance=bn,
            num_trajectories=row["n_trajectories"],
            output_file=dataset_path,
            sampling_frequency=row["sampling_frequency"],
            trajectory_length=row["trajectory_length"],
            **self.simulate_trajectories_to_csv_kwargs
        )

        # if not success:
        #     print(f"[INFO] Dataset {dataset_path} NOT created (requirements not met). Skipping inference.")
        #     return False

        # ---------- 4. inference ----------
        manager_bnfinder(
            ## file paths
            dataset_path=dataset_path,
            ground_truth_path=gt_path,
            trained_model_name=self.paths["results"] / cid,
            bnf_file_path=bnf_dataset_path,
            metrics_file=metrics_path,
            ## model parameters
            score_functions=[row["score_function"]],
            ## analysis settings
            analysis_metrics=self.analysis_metrics,
            analysis_score_functions=self.analysis_score_functions,
            dataset_succeeded=success,
            attractor_ratio=ratio,
        )

        return True
    # =========================
    # PARALLEL EXECUTION
    # =========================
    def run_experiment(
        self,
        *,
        n_jobs: int = 1,
        subset: Optional[pd.DataFrame] = None,
        progress_interval: int = 100,
    ):
        """
        Run experiment.
        - n_jobs = number of parallel processes
        - subset = optional subset of experiment_df
        - progress_interval = print progress every N steps
        Returns: number of successful datasets
        """

        # save metadata for experiment (common: condition_id) # -> data/experiment_name/metadata

        df = subset if subset is not None else self.experiment_df

        rows = [row for _, row in df.iterrows()]

        success_count = 0
        total = len(rows)

        if n_jobs == 1:
            for idx, row in enumerate(rows, 1):
                if self._run_single_condition(row):
                    success_count += 1
                if idx % progress_interval == 0:
                    print(f"[Progress] {idx}/{total} conditions completed ({100*idx/total:.1f}%)")
        else:
            with mp.Pool(processes=n_jobs) as pool:
                for idx, result in enumerate(pool.imap_unordered(self._run_single_condition, rows), 1):
                    if result:
                        success_count += 1
                    if idx % progress_interval == 0:
                        print(f"[Progress] {idx}/{total} conditions completed ({100*idx/total:.1f}%)")

        # TODO COMMENTED: remove prints 
        # print(f"\n[SUMMARY] Generated datasets: {success_count}/{len(rows)}")

        # merge all csv
        self._merge_csv()
        # obtain metadata
        self.save_experiment_df_to_csv()
        return success_count

