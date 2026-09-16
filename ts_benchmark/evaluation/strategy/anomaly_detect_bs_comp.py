# -*- coding: utf-8 -*-
import os
import base64
import pickle
import time
import traceback
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
from ts_benchmark.utils.random_utils import fix_random_seed


class AnomalyDetect(Strategy):
    """
    异常检测类，用于在时间序列数据上执行异常检测。
    """

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化子类实例。

        :param strategy_config: 模型评估配置。
        """
        super().__init__(strategy_config, evaluator)
        self.model = None
        self.data_lens = None


    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed()

        model = model_factory()
        try:
            self.model = model
            train_data, train_label, test_data, test_label = self.split_data(
                series_name
            )
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(train_data, train_label)  # 在训练数据上拟合模型
            else:
                self.model.fit(train_data, train_label)  # 在训练数据上拟合模型

            end_fit_time = time.time()
            predict_labels, another = self.detect(test_data)
            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            actual_label = test_label.to_numpy().flatten()
            end_inference_time = time.time()

            # ========== [新增] NPZ 保存 ==========
            # save_npz = self.strategy_config.get("save_npz", False)
            #if getattr(self.model.config, 'save_npz', False):
            if hasattr(self.model, 'config') and getattr(self.model.config, 'save_npz', False):
                self._save_npz_csv(
                    series_name=series_name,
                    model_name=model_factory.model_name,
                    test_data=test_data,
                    actual_label=actual_label,
                    anomaly_scores=another,
                    predict_labels_dict=predict_labels,
                )
            # ========== [新增结束] ================

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                remaining_length = len(actual_label) - len(predict_label)
                print(remaining_length)
                # Pad the predict_label array with zeros at the end
                if remaining_length > 0:
                    predict_label = np.pad(
                        predict_label,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )
                    another = np.pad(
                        another,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )

                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float)
                )
                print(single_series_results)

                inference_data = [predict_label, another]
                actual_data_pickle = pickle.dumps(test_label)
                actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")

                inference_data_pickle = pickle.dumps(inference_data)
                inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                    "utf-8"
                )
                single_series_results += [
                    series_name,
                    end_fit_time - start_fit_time,
                    end_inference_time - end_fit_time,
                    ratio,
                    '',
                    '',
                    log_info,
                ]
                single_series_results_list.append(single_series_results)
        except Exception as e:
            # log = f"{traceback.format_exc()}\n{e}"
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )]
        return single_series_results_list

    def _save_npz_csv(self, series_name, model_name, test_data,
                      actual_label, anomaly_scores, predict_labels_dict, output_dir='./reconstruction_results'):
        """
            NPZ 文件命名: {model_name}_{dataset_name}.npz
        内容:
          - input_timeseries: 测试集原始多元时序 (N, C)
          - ground_truth:     测试集真值标签 (N,)
          - anomaly_scores:   模型最终异常分数 (M,) （M 可能 ≤ N）
          - predicted_labels:  模型预测异常标签 (M,)

        将推理结果同时保存为 NPZ 和 CSV 文件。
        采用截断策略以 anomaly_scores 长度为基准，确保各列严格等长。
        强制提取 test_data.index 保证时间戳或序列号被存储。
        """
        dataset_name = series_name.replace(".csv", "").replace("/", "_").replace("\\", "_")
        os.makedirs(output_dir, exist_ok=True)

        npz_path = os.path.join(output_dir, f"{model_name}_{dataset_name}.npz")
        csv_path = os.path.join(output_dir, f"{model_name}_{dataset_name}.csv")

        # 安全防范：防止模型未返回分数导致 len() 报错
        M = len(anomaly_scores) if anomaly_scores is not None else len(actual_label)

        # 获取第一个 ratio 的预测标签 (处理可能为 None 的情况)
        first_key = list(predict_labels_dict.keys())[0] if predict_labels_dict else None
        pred_labels = predict_labels_dict.get(first_key) if first_key else None

        # 1. 提取时间戳 / Index
        if hasattr(test_data, 'index'):
            time_index = test_data.index[:M].to_numpy()
        else:
            time_index = np.arange(M)

        # 2. 安全提取并截断特征数据
        inputs_np = test_data.values[:M] if hasattr(test_data, 'values') else test_data[:M]
        ground_truth_np = actual_label[:M]
        scores_np = anomaly_scores[:M] if anomaly_scores is not None else np.zeros(M)
        labels_np = pred_labels[:M] if pred_labels is not None else np.zeros(M)

        # --- 保存 NPZ 文件 ---
        np.savez(
            npz_path,
            time_index=time_index,
            input_timeseries=inputs_np,
            ground_truth=ground_truth_np,
            anomaly_scores=scores_np,
            predicted_labels=labels_np,
        )

        # --- 保存 CSV 文件 ---
        data_dict = {'time_index': time_index}

        # 如果输入是多变量时间序列 (2D)，将每个特征解包为 CSV 的独立列
        if inputs_np.ndim == 2:
            cols = test_data.columns if hasattr(test_data, "columns") else [f"Feature_{i}" for i in
                                                                            range(inputs_np.shape[1])]
            for i, col_name in enumerate(cols):
                data_dict[col_name] = inputs_np[:, i]
        else:
            # 单变量情况
            col_name = test_data.name if hasattr(test_data, "name") else "Feature_0"
            data_dict[col_name] = inputs_np

        data_dict['ground_truth'] = ground_truth_np
        data_dict['anomaly_scores'] = scores_np
        data_dict['predicted_labels'] = labels_np

        # 利用 Pandas 高效保存对齐的 CSV
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path, index=False)

        print(f"\n[Export] Results Saved Successfully!")
        print(f"   -> NPZ: {npz_path}")
        print(f"   -> CSV: {csv_path}")
        print(f"   -> Data Shape: {df.shape} (Tail truncated: {len(actual_label) - M} points)\n")

    def split_data(self, data: str):
        raise NotImplementedError

    def detect(self, test_data: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ANOMALY_RATIO,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]


class FixedDetectScore(AnomalyDetect):
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

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(AnomalyDetect):
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

    def detect(self, test_data):
        return self.model.detect_label(test_data)


    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AllDetectScore(AnomalyDetect):
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
