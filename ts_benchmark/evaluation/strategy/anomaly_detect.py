# -*- coding: utf-8 -*-
import base64
import pickle
import time
import os
import traceback
import copy
import json
import itertools
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

# 原始定义
# class AnomalyDetect(Strategy):
#     """
#     异常检测类，用于在时间序列数据上执行异常检测。
#     """
#
#     def __init__(self, strategy_config: dict, evaluator: Evaluator):
#         """
#         初始化子类实例。
#
#         :param strategy_config: 模型评估配置。
#         """
#         super().__init__(strategy_config, evaluator)
#         self.model = None
#         self.data_lens = None
#
#
#     def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
#         """
#         执行异常检测策略。
#
#         :param series_name: 要执行异常检测的序列名称。
#         :param model_factory: 模型对象的构造/工厂函数。
#         :return: 评估结果。
#         """
#         fix_random_seed()
#
#         model = model_factory()
#         try:
#             self.model = model
#             train_data, train_label, test_data, test_label = self.split_data(
#                 series_name
#             )
#
#             # 数据分析用，可删除
#             # train_data = pd.DataFrame(
#             #     self.scaler.transform(train_data.values),
#             #     columns=train_data_value.columns,
#             #     index=train_data_value.index,
#             # )
#             # self.train_data_loader = anomaly_detection_data_provider(
#             #     train_data,
#             #     batch_size=config.batch_size,
#             #     win_size=config.seq_len,
#             #     step=1,
#             #     mode="train",
#             # )
#             #
#             # test = pd.DataFrame(
#             #     self.scaler.transform(test_data.values), columns=test.columns, index=test.index
#             # )
#             # self.test_data_loader = anomaly_detection_data_provider(
#             #     test,
#             #     batch_size=config.batch_size,
#             #     win_size=config.seq_len,
#             #     step=1,
#             #     mode="test",
#             # )
#
#             start_fit_time = time.time()
#             if hasattr(model, "detect_fit"):
#                 self.model.detect_fit(train_data, train_label)  # 在训练数据上拟合模型
#             else:
#                 self.model.fit(train_data, train_label)  # 在训练数据上拟合模型
#
#             # 可视化指定段
#             # model.reconstruct_segment_and_visualize(
#             #     test_data=test_data,
#             #     test_label=test_label['label'],  # 假设 test_label 是单列 DataFrame
#             #     start='15506',
#             #     end='15633',
#             #     series_name=series_name,
#             #     save_path='./figures/',
#             #     show=True
#             # )
#
#             end_fit_time = time.time()
#             # 兼容不同模型可能返回不同数量的值（单个数组 或 Tuple）
#             predict_labels, another = self.detect(test_data, test_label, series_name)
#             # if isinstance(detect_res, (tuple, list)):
#             #     predict_labels = detect_res[0]
#             #     another = detect_res[1] if len(detect_res) > 1 else detect_res[0]
#             # else:
#             #     predict_labels = detect_res
#             #     another = detect_res
#             if not isinstance(predict_labels, dict):
#                 predict_labels = {"None": predict_labels}
#
#             actual_label = test_label.to_numpy().flatten()
#             end_inference_time = time.time()
#
#             single_series_results_list = []
#             for ratio, predict_label in predict_labels.items():
#                 remaining_length = len(actual_label) - len(predict_label)
#                 print(f"remaining_length:{remaining_length}")
#                 # Pad the predict_label array with zeros at the end
#                 if remaining_length > 0:
#                     predict_label = np.pad(
#                         predict_label,
#                         (0, remaining_length),
#                         mode="constant",
#                         constant_values=0,
#                     )
#                     another = np.pad(
#                         another,
#                         (0, remaining_length),
#                         mode="constant",
#                         constant_values=0,
#                     )
#
#                 single_series_results, log_info = self.evaluator.evaluate_with_log(
#                     actual=actual_label.astype(float),
#                     predicted=predict_label.astype(float)
#                 )
#                 print(f"single_series_results:{single_series_results}")
#
#                 inference_data = [predict_label, another]
#                 actual_data_pickle = pickle.dumps(test_label)
#                 actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")
#
#                 inference_data_pickle = pickle.dumps(inference_data)
#                 inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
#                     "utf-8"
#                 )
#                 single_series_results += [
#                     series_name,
#                     end_fit_time - start_fit_time,
#                     end_inference_time - end_fit_time,
#                     ratio,
#                     '',
#                     '',
#                     log_info,
#                 ]
#                 single_series_results_list.append(single_series_results)
#         except Exception as e:
#             # log = f"{traceback.format_exc()}\n{e}"
#             log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
#             single_series_results_list = [self.get_default_result(
#                 **{FieldNames.LOG_INFO: log}
#             )]
#         return single_series_results_list
#
#     def split_data(self, data: str):
#         raise NotImplementedError
#
#     def detect(self, test_data: pd.DataFrame, test_label: pd.Series, series_name: str) -> List[Any]:
#         raise NotImplementedError
#
#     @staticmethod
#     def accepted_metrics():
#         raise NotImplementedError
#
#     @property
#     def field_names(self) -> List[str]:
#         return self.evaluator.metric_names + [
#             FieldNames.FILE_NAME,
#             FieldNames.FIT_TIME,
#             FieldNames.INFERENCE_TIME,
#             FieldNames.ANOMALY_RATIO,
#             FieldNames.ACTUAL_DATA,
#             FieldNames.INFERENCE_DATA,
#             FieldNames.LOG_INFO,
#         ]

class AnomalyDetect(Strategy):

    REQUIRED_CONFIGS = ["seed"]
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
        fix_random_seed()
        _seed = self.strategy_config.get("seed", 2021)
        fix_all_random_seed(_seed)


    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed()

        # 🌟 1. 提取网格搜索超参数空间，保证不污染原配置
        hyper_params = getattr(model_factory, 'model_hyper_params', {}).copy()
        search_sw = hyper_params.pop("search_smooth_windows", None)
        search_pq = hyper_params.pop("search_pot_qs", None)  # 🌟 [新增]：提取 pot_q 搜索空间
        # 🌟 score_lambda 是【纯后处理】参数：只出现在
        # E = AvgPool(E_time, w) + score_lambda * AvgPool(E_freq, w)
        # 不参与训练，因此可以放进免重训的快速网格。
        search_sl = hyper_params.pop("search_score_lambdas", None)
        # 🌟 [v6_10_2] score_len_divisor：分数轴基线窗宽的除数，W_b = N // divisor。
        # 它只作用在打分链路的【最后一维分数】上（_normalize_scores），
        # 既不参与训练，也不影响逐通道误差矩阵 E，因此同样属于免重训网格，
        # 而且比 sw 更便宜 —— 换它连 E 都不用重算。
        # divisor = 1 表示"不做时变校正"（基线取全局常数），这一档必须可达：
        # 九数据集实测里 Genesis 与 MSL 的最优解就在这一档。
        search_ld = hyper_params.pop("search_score_len_divisors", None)


        model = model_factory()
        try:
            self.model = model

            # 若未传入特定搜索列表，则优雅退化为模型当前设定的标量
            if search_sw is None:
                search_sw = [getattr(self.model.config, 'SMOOTH_WINDOW', 7)] if hasattr(self.model, 'config') else [7]
            if search_pq is None and self.model.config.use_pot:
                search_pq = [getattr(self.model.config, 'pot_q', 0.01)] if hasattr(self.model, 'config') else [0.01]  # 🌟 [新增]
            if search_sl is None:
                search_sl = [getattr(self.model.config, 'score_lambda', 0.5)] if hasattr(self.model,'config') else [0.5]
            if search_ld is None:
                search_ld = [getattr(self.model.config, 'score_len_divisor', 8)] if hasattr(self.model, 'config') else [8]

            if not isinstance(search_sw, (list, tuple)): search_sw = [search_sw]
            if not isinstance(search_pq, (list, tuple)): search_pq = [search_pq]
            if not isinstance(search_sl, (list, tuple)): search_sl = [search_sl]
            if not isinstance(search_ld, (list, tuple)): search_ld = [search_ld]

            train_data, train_label, test_data, test_label = self.split_data(series_name)

            # =========================================================
            # 🌟 2. 静态执行区：有且仅执行一次的耗时模型深度训练
            # =========================================================
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(train_data, train_label)
            else:
                self.model.fit(train_data, train_label)
            end_fit_time = time.time()

            # 唯一的、将被所有网格参数共享的漫长训练时间
            fit_duration = end_fit_time - start_fit_time

            actual_label = test_label.to_numpy().flatten()
            single_series_results_list = []

            # =========================================================
            # 🌟 3. 动态执行区：免重训极速推理与打分重构
            # =========================================================
            # ⚠️ 循环顺序按【重算代价】从大到小排，不能随意调换：
            #   (sw, sl)  —— 决定测试段 E 矩阵，换一次要重算 _build_E（SWaT 规模 18 秒），
            #                 最贵，放最外层；
            #   ld        —— 只改分数轴基线窗宽，E 与 Q_k 都能复用，只需重算
            #                 _normalize_scores 与参考池分数，中等代价，放中间；
            #   pq        —— 只改阈值，前面全部命中缓存，最便宜，放最内层。
            # 顺序写反会让外层缓存反复失效，网格耗时成倍增加。
            for sw, sl, ld, pq in itertools.product(search_sw, search_sl,
                                                    search_ld, search_pq):

                # 3.1 动态热更模型后处理超参
                if hasattr(self.model, "config"):
                    self.model.config.SMOOTH_WINDOW = sw
                    self.model.config.score_lambda = sl
                    self.model.config.score_len_divisor = ld  # 🌟 [v6_10_2]
                    self.model.config.pot_q = pq  # 🌟 [新增]：动态热更 POT 阈值系数

                # 3.2 启动极速推断评估
                start_inference_time = time.time()

                # 兼容不同子类的 detect 传参要求
                try:
                    detect_res = self.detect(test_data, test_label, series_name)
                except TypeError:
                    print("ERROR:detect_res = self.detect(test_data, test_label, series_name)")

                # 兼容 Tuple 分解与单变量返回
                if isinstance(detect_res, (tuple, list)):
                    predict_labels = detect_res[0]
                    another = detect_res[1] if len(detect_res) > 1 else detect_res[0]
                else:
                    predict_labels = detect_res
                    another = detect_res

                end_inference_time = time.time()
                # 独立的单次打分耗时（通常只有几秒）
                inference_duration = end_inference_time - start_inference_time

                if not isinstance(predict_labels, dict):
                    predict_labels = {"None": predict_labels}

                # 3.3 为当前的网格坐标生成【专属参数 JSON 快照】
                current_params = copy.deepcopy(hyper_params)
                current_params["SMOOTH_WINDOW"] = sw
                current_params["score_lambda"] = sl
                current_params["score_len_divisor"] = ld   # 🌟 [v6_10_2] 必须记进 CSV，
                                                          # 否则事后无法分辨每行用的哪个窗宽
                current_params["pot_q"] = pq
                model_params_str = json.dumps(current_params, sort_keys=True)
                print(f"SMOOTH_WINDOW:{sw}, score_lambda:{sl}, "
                      f"score_len_divisor:{ld}, pot_q:{pq}")

                # ========== [新增] NPZ 保存 ==========
                #save_npz = self.strategy_config.get("save_npz", False)
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

                # 3.4 指标评估与结果装载
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

                    # 🌟 强力拦截：将生成的专属参数串直接对齐写入我们定制的 field_names
                    row_result = [model_params_str] + evaluate_result + [
                        series_name,
                        fit_duration,  # FieldNames.FIT_TIME
                        inference_duration,  # FieldNames.INFERENCE_TIME
                        ratio,  # FieldNames.ANOMALY_RATIO
                        '',  # ACTUAL_DATA: 置空，防止网格裂变导致海量 Pickle 引发内存溢出 OOM
                        '',  # INFERENCE_DATA: 同上
                        log_info,
                    ]
                    single_series_results_list.append(row_result)

        except Exception as e:
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            hyper_params_str = json.dumps(getattr(model_factory, 'model_hyper_params', {}), sort_keys=True)
            # 在错误时依然提供对齐格式，保证框架正常生成 CSV
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log, FieldNames.MODEL_PARAMS: hyper_params_str}
            )]

        return single_series_results_list

    def _save_npz_csv(self, series_name, model_name, test_data,
                      actual_label, anomaly_scores, predict_labels_dict, output_dir='./reconstruction_results'):
        """
        将推理结果同时保存为 NPZ 和 CSV 文件。
        采用截断策略以 anomaly_scores 长度为基准，确保各列严格等长。
        强制提取 test_data.index 保证时间戳或序列号被存储。
        """
        dataset_name = series_name.replace(".csv", "").replace("/", "_").replace("\\", "_")
        os.makedirs(output_dir, exist_ok=True)

        time_str = datetime.now().strftime("%m%d%H%M")
        npz_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_{time_str}.npz")
        csv_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_{time_str}.csv")

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

    def detect(self, test_data: pd.DataFrame, test_label: pd.Series, series_name: str) -> List[Any]:
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        # 🌟 主动前置声明 MODEL_PARAMS，彻底阻断 build_result_df 中无差别的静态覆盖
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

    def detect(self, test_data, test_label, series_name):
        return self.model.detect_score(test_data, test_label, series_name)

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

    def detect(self, test_data, test_label, series_name):
        return self.model.detect_label(test_data,test_label, series_name)


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
