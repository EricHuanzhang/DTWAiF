"""
Anomaly-detection evaluation strategies for the TFB benchmark.

One invocation trains the model once per series and evaluates a single
hyper-parameter setting.
"""
import base64
import pickle
import time
import warnings
import os
import traceback
import json
from datetime import datetime
from typing import List, Any

import numpy as np
import pandas as pd

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_label
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_random_seed, fix_all_random_seed


class AnomalyDetect(Strategy):

    """Base strategy for anomaly-detection evaluation."""
    REQUIRED_CONFIGS = ["seed"]
    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        super().__init__(strategy_config, evaluator)
        self.model = None
        self.data_lens = None
        fix_random_seed()
        _seed = self.strategy_config.get("seed", 2021)
        fix_all_random_seed(_seed)


    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """

        Train the model on the training split, then score and label the test split.

        One invocation evaluates one hyper-parameter setting and produces one row
        per entry of `anomaly_ratio` (or a single row when POT supplies the
        threshold).

        """
        fix_random_seed()

        hyper_params = getattr(model_factory, 'model_hyper_params', {}).copy()
        _stale = sorted(k for k in hyper_params if k.startswith("search_"))
        if _stale:
            warnings.warn(f"this release evaluates a single hyper-parameter setting; "
                          f"the grid keys {_stale} are ignored. Sweep a parameter by "
                          f"invoking the benchmark once per value instead.")
            for k in _stale:
                hyper_params.pop(k)

        model = model_factory()
        try:
            self.model = model

            train_data, train_label, test_data, test_label = self.split_data(series_name)

            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(train_data, train_label)
            else:
                self.model.fit(train_data, train_label)
            end_fit_time = time.time()

            fit_duration = end_fit_time - start_fit_time

            actual_label = test_label.to_numpy().flatten()
            single_series_results_list = []

            start_inference_time = time.time()

            try:
                detect_res = self.detect(test_data, test_label, series_name)
            except TypeError:
                print("ERROR:detect_res = self.detect(test_data, test_label, series_name)")

            if isinstance(detect_res, (tuple, list)):
                predict_labels = detect_res[0]
                another = detect_res[1] if len(detect_res) > 1 else detect_res[0]
            else:
                predict_labels = detect_res
                another = detect_res

            end_inference_time = time.time()
            inference_duration = end_inference_time - start_inference_time

            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            model_params_str = json.dumps(hyper_params, sort_keys=True)

            if hasattr(self.model, 'config') and getattr(self.model.config, 'save_npz', False):
                self._save_npz_csv(
                    series_name=series_name,
                    model_name=model_factory.model_name,
                    test_data=test_data,
                    actual_label=actual_label,
                    anomaly_scores=another,
                    predict_labels_dict=predict_labels,
                )

            print(f"Evaluate Results:")
            for ratio, predict_label in predict_labels.items():
                remaining_length = len(actual_label) - len(predict_label)

                if remaining_length > 0:
                    predict_label = np.pad(predict_label, (0, remaining_length), mode="constant", constant_values=0)
                    if another is not None and getattr(another, 'ndim', 0) > 0:
                        another_pad = np.pad(another, (0, remaining_length), mode="constant", constant_values=0)
                    else:
                        another_pad = another

                evaluate_result, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float)
                )
                print(f"{evaluate_result}")

                row_result = [model_params_str] + evaluate_result + [
                    series_name,
                    fit_duration,
                    inference_duration,
                    ratio,
                    '',
                    '',
                    log_info,
                ]
                single_series_results_list.append(row_result)

        except Exception as e:
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            hyper_params_str = json.dumps(getattr(model_factory, 'model_hyper_params', {}), sort_keys=True)
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log, FieldNames.MODEL_PARAMS: hyper_params_str}
            )]

        return single_series_results_list

    def _save_npz_csv(self, series_name, model_name, test_data,
                      actual_label, anomaly_scores, predict_labels_dict, output_dir='./reconstruction_results'):
        """Persist per-run scores and labels for offline analysis."""
        dataset_name = series_name.replace(".csv", "").replace("/", "_").replace("\\", "_")
        os.makedirs(output_dir, exist_ok=True)

        time_str = datetime.now().strftime("%m%d%H%M")
        npz_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_{time_str}.npz")
        csv_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_{time_str}.csv")

        M = len(anomaly_scores) if anomaly_scores is not None else len(actual_label)

        first_key = list(predict_labels_dict.keys())[0] if predict_labels_dict else None
        pred_labels = predict_labels_dict.get(first_key) if first_key else None

        if hasattr(test_data, 'index'):
            time_index = test_data.index[:M].to_numpy()
        else:
            time_index = np.arange(M)

        inputs_np = test_data.values[:M] if hasattr(test_data, 'values') else test_data[:M]
        ground_truth_np = actual_label[:M]
        scores_np = anomaly_scores[:M] if anomaly_scores is not None else np.zeros(M)
        labels_np = pred_labels[:M] if pred_labels is not None else np.zeros(M)

        np.savez(
            npz_path,
            time_index=time_index,
            input_timeseries=inputs_np,
            ground_truth=ground_truth_np,
            anomaly_scores=scores_np,
            predicted_labels=labels_np,
        )

        data_dict = {'time_index': time_index}

        if inputs_np.ndim == 2:
            cols = test_data.columns if hasattr(test_data, "columns") else [f"Feature_{i}" for i in
                                                                            range(inputs_np.shape[1])]
            for i, col_name in enumerate(cols):
                data_dict[col_name] = inputs_np[:, i]
        else:
            col_name = test_data.name if hasattr(test_data, "name") else "Feature_0"
            data_dict[col_name] = inputs_np

        data_dict['ground_truth'] = ground_truth_np
        data_dict['anomaly_scores'] = scores_np
        data_dict['predicted_labels'] = labels_np

        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path, index=False)

        print(f"\n[Export] Results Saved Successfully!")
        print(f"   -> NPZ: {npz_path}")
        print(f"   -> CSV: {csv_path}")
        print(f"   -> Data Shape: {df.shape} (Tail truncated: {len(actual_label) - M} points)\n")

    def split_data(self, data: str):
        raise NotImplementedError

    def detect(self, test_data: pd.DataFrame, test_label: pd.Series, series_name: str) -> List[Any]:
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        return [FieldNames.MODEL_PARAMS] + self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ANOMALY_RATIO,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]

class FixedDetectScore(AnomalyDetect):
    """Score-only evaluation on a fixed train/test split."""
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class FixedDetectLabel(AnomalyDetect):
    """Label evaluation on a fixed train/test split."""
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class UnFixedDetectScore(AnomalyDetect):
    """Score-only evaluation with a per-series split."""
    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )

        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data, test_label, series_name):
        return self.model.detect_score(test_data, test_label, series_name)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(AnomalyDetect):
    """Label evaluation with a per-series split."""
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data, test_label, series_name):
        return self.model.detect_label(test_data,test_label, series_name)


    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AllDetectScore(AnomalyDetect):
    """Score-only evaluation over all series."""
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class AllDetectLabel(AnomalyDetect):
    """Label evaluation over all series."""
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__
