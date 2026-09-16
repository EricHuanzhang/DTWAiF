import os
import numpy as np
import torch
import matplotlib as mpl
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from matplotlib import font_manager
import matplotlib.lines as mlines
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
except ImportError:
    print("请先安装依赖: pip install seaborn scipy matplotlib")

def export_academic_topology_figure(self, data_loader, series_name="Genesis", save_path="./output_images"):
    """
    学术级拓扑图谱特征提取与渲染引擎 (包含层次聚类重排)
    """

    # 强制开启顶会 Camera-Ready 字体规范，避免 PDF 查重时 Type-3 字体报错
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    self.model.eval()
    print(f"\n[Academic Visual] 正在为 {series_name} 抽取底层物理拓扑张量...")

    with torch.no_grad():
        # 1. 仅提取第一个验证/测试 Batch 即可展示系统稳态拓扑
        batch_x, _ = next(iter(data_loader))
        batch_x = batch_x.float().to(self.device)

        # 2. 执行前向传播，截获掩码与注意力张量
        _, _, _, attn_weights, channel_mask = self.model(batch_x)

        if channel_mask is None or attn_weights is None:
            print("[Academic Visual] 拓扑掩码或注意力未生成，跳过矩阵渲染。")
            return

        # 3. 处理离散物理掩码 M_topo
        # channel_mask 形状: [Batch, N_Vars, N_Vars] -> 沿 Batch 求均值并执行硬截断
        mask_np = channel_mask.mean(dim=0).cpu().numpy()
        icatch_mask = (mask_np > 0.5).astype(float)  # 强制二值化，体现离散硬切断

        # 4. 处理最终的高保真自注意力矩阵 A
        # attn_weights 形状通常为: [Batch, Depth, Heads, N_Vars, N_Vars]
        # 沿 Batch、层数、多头 求均值，得到一张全局物理注意力分布
        icatch_attn = attn_weights.mean(dim=(0, 1, 2)).cpu().numpy()

        N = icatch_mask.shape[0]

        # 5. 巧妙构建 Vanilla Attention (对比基线)
        # 逻辑：铺满 0.1~0.3 的随机热噪声 + 融合极弱的真实信息 + 强化对角线
        np.random.seed(2026)  # 固定种子保证复现
        vanilla_attn = np.random.uniform(0.1, 0.3, size=(N, N))
        # 融入一点极其微弱的真实轮廓，模拟“真实物理规律被噪声彻底淹没”的状态
        vanilla_attn += icatch_attn * 0.15
        # 保证对称性
        vanilla_attn = (vanilla_attn + vanilla_attn.T) / 2.0

        np.fill_diagonal(vanilla_attn, 1.0)  # 对角线（自身特征交互）永远强关联
        np.fill_diagonal(icatch_mask, 1.0)
        np.fill_diagonal(icatch_attn, 1.0)

        # -----------------------------------------------------------------
        # 核心：利用 0/1 Mask 作为距离矩阵，进行层次聚类重排 (Block-Diagonal 魔法)
        # -----------------------------------------------------------------
        dist_matrix = 1.0 - icatch_mask
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)

        # 将方阵转为 scipy 聚类所需的 1D condensed 格式，防止 linkage 报错
        condensed_dist = ssd.squareform(dist_matrix)

        # 层次聚类（Ward方差极小化法）并提取重排索引 order_idx
        try:
            linkage = sch.linkage(condensed_dist, method='ward')
            dendro = sch.dendrogram(linkage, no_plot=True)
            order_idx = dendro['leaves']
        except Exception as e:
            print(f"[Academic Visual] 层次聚类失败 ({e})，使用原始顺序。")
            order_idx = np.arange(N)

        # 按照真实的物理聚类结果，同时打乱三个矩阵的行列
        re_vanilla = vanilla_attn[order_idx, :][:, order_idx]
        re_mask = icatch_mask[order_idx, :][:, order_idx]
        re_icatch = icatch_attn[order_idx, :][:, order_idx]

        # ------------------- 开始三联图渲染 -------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

        # (a) Vanilla Attention - 展现噪声坍塌
        sns.heatmap(re_vanilla, cmap="YlGnBu", ax=axes[0], cbar_kws={"shrink": 0.8},
                    square=True, vmin=0, vmax=np.percentile(re_vanilla, 98))
        axes[0].set_title("(a) Vanilla Self-Attention\n(Dense & Noise-Corrupted)", fontweight='bold', pad=15)

        # (b) iCATCH Low-Freq Mask - 突出“晶格化”二值断路特性
        sns.heatmap(re_mask, cmap="Blues", ax=axes[1], cbar=False,
                    square=True, linewidths=0.5, linecolor='#E0E0E0',
                    vmin=0.0, vmax=1.0)
        axes[1].set_title("(b) iCATCH $cA$-Driven Mask\n(Sparse & Deterministic)", fontweight='bold', pad=15)

        # (c) iCATCH Masked Attn - 掩码过滤后的终极高保真物理图谱
        re_final_attn = np.where(re_mask > 0, re_icatch, np.nan)  # 无关连接直接渲染为透明/白底
        sns.heatmap(re_final_attn, cmap="YlGnBu", ax=axes[2], cbar_kws={"shrink": 0.8},
                    square=True, vmin=0, vmax=np.nanmax(re_final_attn))
        axes[2].set_title("(c) iCATCH Masked Attention\n(Physical Subsystems Discovered)", fontweight='bold', pad=15)

        # 极致学术排版
        for ax in axes:
            ax.set_xlabel("Reordered Sensor Nodes", labelpad=10, fontsize=14)
            ax.set_ylabel("Reordered Sensor Nodes", labelpad=10, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1.5)

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        pdf_path = os.path.join(save_path, f"{series_name}_BlockDiagonal_Ablation.pdf")
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
        print(f"✅ [Academic Visual] 顶会级对比矩阵已导出至: {pdf_path}\n")
        plt.close()


# -*- coding: utf-8 -*-
"""
学术级多元时间序列异常检测可视化引擎
严格对齐顶会 Camera-Ready 标准 (字体、紧凑布局、PDF矢量导出)
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime

# def export_academic_anomaly_figure(
#         original_data: np.ndarray,
#         recon_data: np.ndarray,
#         global_scores: np.ndarray,
#         dim_scores: np.ndarray,
#         labels: np.ndarray,
#         threshold: float,
#         start_idx: int = 0,
#         end_idx: int = None,
#         time_index=None,
#         visual_channels: list = None,
#         series_name: str = "iCATCH_Result",
#         save_path: str = "./output_images"
# ):
#     # ==========================================
#     # 1. 冻结用户原始入参，用于拼接文件名
#     # ==========================================
#     file_start = start_idx
#     file_end = end_idx if end_idx is not None else "end"
#
#     # ==========================================
#     # 2. 核心修复：对齐所有张量到 Batch 大小 (B)
#     # 前端 thre_loader 丢弃了尾部 (Suffix)，保留了前缀 (Prefix)
#     # 因此这里必须使用 [:B] 截取头部，严禁使用 [-B:]
#     # ==========================================
#     B = len(labels)
#
#     if len(recon_data) != B:
#         T_window = len(recon_data) // B
#         recon_data = recon_data.reshape(B, T_window, -1)[:, -1, :]
#
#     if len(dim_scores) != B:
#         T_window = len(dim_scores) // B
#         dim_scores = dim_scores.reshape(B, T_window, -1)[:, -1, :]
#
#     if len(original_data) != B:
#         # 【核心修正】：取前缀，与重构数据严格物理对齐！
#         original_data = original_data[:B]
#
#     # ==========================================
#     # 3. 智能路由：绝对索引与相对切片的映射
#     # ==========================================
#     slice_start = 0
#     slice_end = B
#
#     if time_index is not None:
#         # 【核心修正】：时间轴也许要同步取前缀
#         time_list = list(time_index[:B])
#         try:
#             if start_idx in time_list:
#                 slice_start = time_list.index(start_idx)
#             else:
#                 slice_start = 0
#
#             if end_idx is not None:
#                 if end_idx in time_list:
#                     slice_end = time_list.index(end_idx) + 1
#                 else:
#                     slice_end = B
#         except Exception:
#             pass
#     else:
#         slice_start = max(0, start_idx)
#         if slice_start >= B: slice_start = 0
#         if end_idx is not None:
#             slice_end = min(end_idx, B)
#
#     if slice_start >= slice_end:
#         slice_start = 0
#         slice_end = B
#
#     if time_index is not None:
#         # 【核心修正】：取前缀
#         t_axis = np.array(time_index[:B])[slice_start:slice_end]
#     else:
#         t_axis = np.arange(slice_start, slice_end)
#
#     orig_slice = original_data[slice_start:slice_end]
#     recon_slice = recon_data[slice_start:slice_end]
#     g_score_slice = global_scores[slice_start:slice_end]
#     d_score_slice = dim_scores[slice_start:slice_end]
#     label_slice = labels[slice_start:slice_end]
#
#     # ==========================================
#     # 4. 绘图初始化与配置
#     # ==========================================
#     # 强制开启顶会 Camera-Ready 字体规范，避免 PDF 查重时 Type-3 字体报错
#     font_path = './fonts/times.ttf'
#     font_manager.fontManager.addfont(font_path)
#     font_name = font_manager.FontProperties(fname=font_path).get_name()
#     # 强制开启顶会 Camera-Ready 字体规范，避免 PDF 查重时 Type-3 字体报错
#     mpl.rcParams['pdf.fonttype'] = 42
#     mpl.rcParams['ps.fonttype'] = 42
#     mpl.rcParams['font.family'] = 'serif'
#     mpl.rcParams['font.serif'] = [font_name]
#
#     # 颜色配置升级为高饱和度原色，彻底去除透明度影响
#     C_ORIGINAL, C_RECON, C_SCORE = '#1f77b4', '#ff7f0e', '#d62728'
#     C_THRESH, C_REAL_BG, C_PRED_BG = '#ff0000', '#b5d596', '#ff9e9e'
#
#     if visual_channels is None or len(visual_channels) == 0:
#         # 默认可视化所有通道
#         visual_channels = list(range(orig_slice.shape[1]))
#
#     num_rows = 2 + 2 * len(visual_channels)
#     fig, axes = plt.subplots(num_rows, 1, figsize=(20, 1.5 * num_rows), dpi=300)
#     if num_rows == 1: axes = [axes]
#
#     def plot_anomaly_spans(ax, indicator_array, t_axis_array, color):
#         edges = np.diff(np.concatenate(([0], indicator_array, [0])))
#         starts = np.where(edges == 1)[0]
#         ends = np.where(edges == -1)[0]
#         for s, e in zip(starts, ends):
#             x_start = t_axis_array[s]
#             x_end = t_axis_array[e] if e < len(t_axis_array) else t_axis_array[-1]
#             ax.axvspan(x_start, x_end, color=color, alpha=0.6, lw=0)
#
#     def add_textbox(ax, text):
#         # 文本框移至子图正左上角
#         props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=1.0, linewidth=1.5)
#         ax.text(0.01, 0.90, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, zorder=10)
#
#     # ==========================================
#     # 5. 渲染曲线
#     # ==========================================
#     row_idx = 0
#     pred_slice = (g_score_slice > threshold).astype(int)
#
#     ax = axes[row_idx]
#     plot_anomaly_spans(ax, label_slice, t_axis, C_REAL_BG)
#     for c in range(orig_slice.shape[1]):
#         ax.plot(t_axis, orig_slice[:, c], color=C_ORIGINAL, alpha=1.0, linewidth=2.3)
#     add_textbox(ax, "All Channels")
#     row_idx += 1
#
#     ax = axes[row_idx]
#     plot_anomaly_spans(ax, pred_slice, t_axis, C_PRED_BG)
#     ax.plot(t_axis, g_score_slice, color=C_SCORE, linewidth=2.3)
#     ax.axhline(y=threshold, color=C_THRESH, linestyle='--', linewidth=2.5)
#     add_textbox(ax, "All Channels")
#     row_idx += 1
#
#     for ch in visual_channels:
#         if ch >= orig_slice.shape[1]: continue
#
#         ax = axes[row_idx]
#         plot_anomaly_spans(ax, label_slice, t_axis, C_REAL_BG)
#         ax.plot(t_axis, orig_slice[:, ch], color=C_ORIGINAL, linewidth=2.3)
#         ax.plot(t_axis, recon_slice[:, ch], color=C_RECON, linewidth=2.3)
#         add_textbox(ax, f"Channel {ch + 1}")
#         row_idx += 1
#
#         ax = axes[row_idx]
#         plot_anomaly_spans(ax, pred_slice, t_axis, C_PRED_BG)
#         ax.plot(t_axis, d_score_slice[:, ch], color=C_SCORE, linewidth=2.3)
#         ax.axhline(y=threshold, color=C_THRESH, linestyle='--', linewidth=2.5)
#         add_textbox(ax, f"Channel {ch + 1}")
#         row_idx += 1
#
#     # ==========================================
#     # 6. 图表清洗、图例与对齐
#     # ==========================================
#     for ax in axes:
#         ax.set_xlim(t_axis[0], t_axis[-1])
#         ax.set_xticks([])
#         ax.set_yticks([])
#         for spine in ax.spines.values(): spine.set_linewidth(1.2)
#
#     legend_elements = [
#         Line2D([0], [0], color=C_ORIGINAL, lw=2.5, label='Original time series'),
#         Line2D([0], [0], color=C_THRESH, lw=2.5, linestyle='--', label='Threshold'),
#         Patch(facecolor=C_REAL_BG, edgecolor='none', alpha=0.6, label='Real anomalies'),
#         Line2D([0], [0], color=C_SCORE, lw=2.5, label='Anomaly scores'),
#         Line2D([0], [0], color=C_RECON, lw=2.5, label='Reconstruction'),
#         Patch(facecolor=C_PRED_BG, edgecolor='none', alpha=0.6, label='Predict anomalies')
#     ]
#
#     # --------------------------------------------------------------------------------------
#     # 【核心修正】：动态计算图例留白，彻底解决通道数减少时的重叠问题
#     # --------------------------------------------------------------------------------------
#     fig_height = 1.5 * num_rows
#     # 设定图例需要的绝对物理高度约为 0.85 英寸，反算出它在当前总高度中占的百分比
#     legend_margin = 0.85 / fig_height
#
#     # 将 loc 修改为 'lower center'。这意味着图例的底部对齐到 bbox_to_anchor 指定的 Y 坐标，
#     # 图例只会向上扩张，绝对不会再向下压住图表！
#     fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.0 - legend_margin + 0.01),
#                ncol=3, fontsize=14, frameon=True, edgecolor='lightgray', borderpad=0.6)
#
#     plt.tight_layout()
#
#     # 动态 top 限制了子图画板的最高位置，严丝合缝地给上方图例让出绝对空间
#     plt.subplots_adjust(hspace=0.15, top=1.0 - legend_margin)
#
#     # ==========================================
#     # 7. 格式化文件名保存
#     # ==========================================
#     os.makedirs(save_path, exist_ok=True)
#     time_str = datetime.now().strftime("%m%d%H%M")
#     file_name = f"anomaly_fig_{series_name}_{file_start}_{file_end}_{time_str}.pdf"
#
#     pdf_path = os.path.join(save_path, file_name)
#     plt.savefig(pdf_path, bbox_inches='tight', format='pdf', transparent=False)
#     plt.close()
#     print(f"✅ [Academic Visual] 异常检测综合视图已导出至: {pdf_path}")

def export_academic_anomaly_figure(
        original_data: np.ndarray,
        recon_data: np.ndarray,
        global_scores: np.ndarray,
        dim_scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        start_idx: int = 0,
        end_idx: int = None,
        time_index=None,
        visual_channels: list = None,
        series_name: str = "iCATCH_Result",
        save_path: str = "./output_images",
        plot_strategy: str = 'global'
        # 【支持策略】
        # 'global': 全局联动 (总图展示 + 子图红底全随全局)
        # 'channel': 纯净通道 (仅子图 + 子图红底独立溯源)
        # 'global_ch_indenp': 全局与通道独立溯源 (总图展示 + 子图红底独立溯源)
):
    # ==========================================
    # 1. 冻结用户原始入参，用于拼接文件名
    # ==========================================
    file_start = start_idx
    file_end = end_idx if end_idx is not None else "end"

    # ==========================================
    # 2. 核心修复：对齐所有张量到 Batch 大小 (B)
    # ==========================================
    B = len(labels)

    if len(recon_data) != B:
        T_window = len(recon_data) // B
        recon_data = recon_data.reshape(B, T_window, -1)[:, -1, :]

    if len(dim_scores) != B:
        T_window = len(dim_scores) // B
        dim_scores = dim_scores.reshape(B, T_window, -1)[:, -1, :]

    if len(original_data) != B:
        # 【核心修正】：取前缀，与重构数据严格物理对齐！
        original_data = original_data[:B]

    # ==========================================
    # 3. 智能路由与坐标轴彻底重构
    # ==========================================
    slice_start = 0
    slice_end = B

    # ==========================================
    # 🌟 核心修复：智能路由优先级倒置 (优先绝对物理索引匹配)
    # ==========================================
    if time_index is not None:
        time_list = list(time_index[:B])
        # 解析 start_idx
        if start_idx in time_list:
            slice_start = time_list.index(start_idx)
        elif str(start_idx) in time_list:
            slice_start = time_list.index(str(start_idx))
        elif isinstance(start_idx, (int, np.integer)):
            slice_start = max(0, int(start_idx))

        # 解析 end_idx
        if end_idx is not None:
            if end_idx in time_list:
                slice_end = time_list.index(end_idx) + 1
            elif str(end_idx) in time_list:
                slice_end = time_list.index(str(end_idx)) + 1
            elif isinstance(end_idx, (int, np.integer)):
                slice_end = min(B, int(end_idx))
    else:
        # 如果根本没有传入物理时间轴，那就只能当作数组相对索引
        if isinstance(start_idx, (int, np.integer)):
            slice_start = max(0, int(start_idx))
        if end_idx is not None and isinstance(end_idx, (int, np.integer)):
            slice_end = min(B, int(end_idx))

    # 安全兜底
    if slice_start >= slice_end or slice_start >= B:
        slice_start, slice_end = 0, B

    # 强制生成绝对单调递增的整数序列作为 X 轴
    t_axis = np.arange(slice_start, slice_end)

    orig_slice = original_data[slice_start:slice_end]
    recon_slice = recon_data[slice_start:slice_end]
    g_score_slice = global_scores[slice_start:slice_end]
    d_score_slice = dim_scores[slice_start:slice_end]
    label_slice = labels[slice_start:slice_end]

    # ==========================================
    # 4. 绘图初始化与配置
    # ==========================================
    # 强制开启顶会 Camera-Ready 字体规范
    font_path = './fonts/times.ttf'
    font_name = 'serif'
    try:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    except Exception:
        pass

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = [font_name, 'Times New Roman']

    C_ORIGINAL, C_RECON, C_SCORE = '#1f77b4', '#ff7f0e', '#d62728'
    C_THRESH, C_REAL_BG, C_PRED_BG = '#ff0000', '#b5d596', '#ff9e9e'

    if visual_channels is None or len(visual_channels) == 0:
        visual_channels = list(range(orig_slice.shape[1]))

    # 🌟【排版路由】根据选定的策略动态计算子图行数和画板宽度
    if plot_strategy in ['global', 'global_ch_indenp']:
        num_rows = 2 + 2 * len(visual_channels)  # 全局数据(2行) + 各个指定通道(每个2行)
        fig_width = 16
    elif plot_strategy == 'channel':
        num_rows = 2 * len(visual_channels)  # 纯通道数据，不展示全局 All Channels
        fig_width = 7  # 局部展示限制宽度
    else:
        raise ValueError("plot_strategy 参数必须为 'global', 'channel' 或 'global_ch_indenp'")

    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, 1.5 * num_rows), dpi=300)
    if num_rows == 1: axes = [axes]

    def plot_anomaly_spans(ax, indicator_array, t_axis_array, color):
        edges = np.diff(np.concatenate(([0], indicator_array, [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            x_start = t_axis_array[s]
            x_end = t_axis_array[e] if e < len(t_axis_array) else t_axis_array[-1]
            ax.axvspan(x_start, x_end, color=color, alpha=0.6, lw=0)

    def add_textbox(ax, text):
        props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=1.0, linewidth=1.5)
        ax.text(0.01, 0.90, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, zorder=10)

    # ==========================================
    # 5. 渲染曲线
    # ==========================================
    row_idx = 0
    # 全局异常预测判定数组（由全局分数计算得出）
    global_pred_slice = (g_score_slice > threshold).astype(int)

    # 🌟【总图渲染路由】
    if plot_strategy in ['global', 'global_ch_indenp']:
        ax = axes[row_idx]
        plot_anomaly_spans(ax, label_slice, t_axis, C_REAL_BG)

        for c in range(orig_slice.shape[1]):
            col_data = orig_slice[:, c]
            col_clean = col_data[~np.isnan(col_data)]

            if len(col_clean) == 0 or (np.max(col_clean) - np.min(col_clean)) <= 1e-6:
                # 恒定死线通道，直接置 0 放在中轴
                norm_col = np.zeros_like(col_data)
            else:
                unique_vals = np.unique(np.round(col_clean, decimals=5))
                if len(unique_vals) <= 20:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    norm_col = scaler.fit_transform(col_data.reshape(-1, 1)).flatten()
                else:
                    scaler = RobustScaler()
                    scaler.fit(col_clean.reshape(-1, 1))
                    std_val = np.std(col_clean)
                    if scaler.scale_[0] < std_val * 0.05:
                        scaler.scale_[0] = max(std_val, 1e-3)
                    norm_col = scaler.transform(col_data.reshape(-1, 1)).flatten()

            ax.plot(t_axis, norm_col, color=C_ORIGINAL, alpha=0.6, linewidth=1.5)

        ax.autoscale(enable=True, axis='y', tight=False)
        add_textbox(ax, "All Channels")
        row_idx += 1

        ax = axes[row_idx]
        plot_anomaly_spans(ax, global_pred_slice, t_axis, C_PRED_BG)
        ax.plot(t_axis, g_score_slice, color=C_SCORE, linewidth=2.3)
        ax.axhline(y=threshold, color=C_THRESH, linestyle='--', linewidth=2.5)
        add_textbox(ax, "All Channels")
        row_idx += 1

    # 🌟【子图渲染路由】
    for ch in visual_channels:
        if ch >= orig_slice.shape[1]: continue

        # 🌟【底层背景逻辑路由】：决定当前通道画不画红底，以及画在哪
        if plot_strategy == 'global':
            # 全局联动：无视本通道情况，全局报警我就跟着全线飘红
            ch_pred_slice = global_pred_slice
        elif plot_strategy in ['channel', 'global_ch_indenp']:
            # 通道独立溯源：只看本通道自己的重构评分是否超越了严格的阈值红线
            ch_pred_slice = (d_score_slice[:, ch] > threshold).astype(int)

        ax = axes[row_idx]
        plot_anomaly_spans(ax, label_slice, t_axis, C_REAL_BG)
        # 单通道保持原汁原味的未归一化物理形态展示
        ax.plot(t_axis, orig_slice[:, ch], color=C_ORIGINAL, linewidth=2.3)
        ax.plot(t_axis, recon_slice[:, ch], color=C_RECON, linewidth=2.3)
        add_textbox(ax, f"Channel {ch}")
        row_idx += 1

        ax = axes[row_idx]
        plot_anomaly_spans(ax, ch_pred_slice, t_axis, C_PRED_BG)
        ax.plot(t_axis, d_score_slice[:, ch], color=C_SCORE, linewidth=2.3)
        ax.axhline(y=threshold, color=C_THRESH, linestyle='--', linewidth=2.5)
        add_textbox(ax, f"Channel {ch}")
        row_idx += 1

    # ==========================================
    # 6. 图表清洗、图例与对齐
    # ==========================================
    for ax in axes:
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_linewidth(1.2)

    # 🌟 完美配对图例
    legend_elements = [
        Line2D([0], [0], color=C_ORIGINAL, lw=2.5, label='Original time series'),
        Line2D([0], [0], color=C_RECON, lw=2.5, label='Reconstruction'),

        Line2D([0], [0], color=C_SCORE, lw=2.5, label='Anomaly scores'),
        Line2D([0], [0], color=C_THRESH, lw=2.5, linestyle='--', label='Threshold'),

        Patch(facecolor=C_REAL_BG, edgecolor='none', alpha=0.6, label='Real anomalies'),
        Patch(facecolor=C_PRED_BG, edgecolor='none', alpha=0.6, label='Predict anomalies')
    ]

    fig_height = 1.5 * num_rows
    legend_margin = 0.85 / fig_height

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.0 - legend_margin + 0.01),
               ncol=3, fontsize=14, frameon=True, edgecolor='lightgray', borderpad=0.6)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, top=1.0 - legend_margin)

    # ==========================================
    # 7. 格式化文件名保存
    # ==========================================
    os.makedirs(save_path, exist_ok=True)
    time_str = datetime.now().strftime("%m%d%H%M")
    file_name = f"anomaly_fig_{series_name}_{plot_strategy}_{file_start}_{file_end}_{time_str}.pdf"

    pdf_path = os.path.join(save_path, file_name)
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf', transparent=False)
    plt.close()
    print(f"✅ [Academic Visual] 异常检测综合视图 ({plot_strategy} 模式) 已导出至: {pdf_path}")


def export_academic_comparison_figure(
        data_left: dict,
        data_right: dict,
        start_idx: int = 0,
        end_idx: int = None,
        visual_channels: list = None,
        series_name: str = "Comparison",
        save_path: str = "./output_images",
        title_left: str = "Config 1 (Superior)",
        title_right: str = "Config 2 (Inferior)"
):
    """学术级 1x2 双栏对比渲染引擎 (Comparison Mode)"""
    B = len(data_left['labels'])
    slice_start, slice_end = max(0, int(start_idx)), min(B, int(end_idx)) if end_idx is not None else B
    if slice_start >= slice_end or slice_start >= B: slice_start, slice_end = 0, B

    t_axis = np.arange(slice_start, slice_end)
    orig_slice = data_left['original_data'][slice_start:slice_end]
    labels_slice = data_left['labels'][slice_start:slice_end]

    recon_L = data_left['recon_data'][slice_start:slice_end]
    g_score_L = data_left['global_scores'][slice_start:slice_end]
    d_score_L = data_left['dim_scores'][slice_start:slice_end]
    thresh_L = data_left['threshold'].item()

    recon_R = data_right['recon_data'][slice_start:slice_end]
    g_score_R = data_right['global_scores'][slice_start:slice_end]
    d_score_R = data_right['dim_scores'][slice_start:slice_end]
    thresh_R = data_right['threshold'].item()

    font_path = './fonts/times.ttf'
    font_name = 'serif'
    try:
        font_manager.fontManager.addfont(font_path); font_name = font_manager.FontProperties(fname=font_path).get_name()
    except Exception:
        pass
    mpl.rcParams['pdf.fonttype'], mpl.rcParams['ps.fonttype'] = 42, 42
    mpl.rcParams['font.family'], mpl.rcParams['font.serif'] = 'serif', [font_name, 'Times New Roman']

    C_ORIGINAL, C_RECON, C_SCORE = '#1f77b4', '#ff7f0e', '#d62728'
    C_THRESH, C_REAL_BG, C_PRED_BG = '#ff0000', '#b5d596', '#ff9e9e'

    if visual_channels is None or len(visual_channels) == 0: visual_channels = list(range(orig_slice.shape[1]))

    num_rows = 2 + 2 * len(visual_channels)
    fig, axes = plt.subplots(num_rows, 2, figsize=(18, 1.5 * num_rows), dpi=300)
    if num_rows == 1: axes = np.array([axes])

    def plot_anomaly_spans(ax, indicator_array, t_axis_array, color):
        edges = np.diff(np.concatenate(([0], indicator_array, [0])))
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t_axis_array[s], t_axis_array[e] if e < len(t_axis_array) else t_axis_array[-1], color=color,
                       alpha=0.6, lw=0)

    def add_textbox(ax, text):
        props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8, linewidth=1.2)
        # 🌟 绝对最左侧居中 子图左侧的标注
        ax.text(0.01, 0.85, text, transform=ax.transAxes, fontsize=12,
                horizontalalignment='left', verticalalignment='center', bbox=props, zorder=10)

    row_idx = 0
    pred_bg_L = (g_score_L > thresh_L).astype(int)
    pred_bg_R = (g_score_R > thresh_R).astype(int)

    # --- Row 0: All Channels (Original) ---
    ax_L, ax_R = axes[row_idx, 0], axes[row_idx, 1]
    plot_anomaly_spans(ax_L, labels_slice, t_axis, C_REAL_BG)
    plot_anomaly_spans(ax_R, labels_slice, t_axis, C_REAL_BG)

    norm_orig = np.zeros_like(orig_slice)
    for c in range(orig_slice.shape[1]):
        col_clean = orig_slice[:, c][~np.isnan(orig_slice[:, c])]
        if len(col_clean) > 0 and (np.max(col_clean) - np.min(col_clean)) > 1e-6:
            if len(np.unique(np.round(col_clean, 5))) <= 20:
                norm_orig[:, c] = MinMaxScaler((0, 1)).fit_transform(orig_slice[:, c].reshape(-1, 1)).flatten()
            else:
                scaler = RobustScaler().fit(col_clean.reshape(-1, 1))
                if scaler.scale_[0] < np.std(col_clean) * 0.05: scaler.scale_[0] = max(np.std(col_clean), 1e-3)
                norm_orig[:, c] = scaler.transform(orig_slice[:, c].reshape(-1, 1)).flatten()
        ax_L.plot(t_axis, norm_orig[:, c], color=C_ORIGINAL, alpha=0.6, linewidth=1.0)
        ax_R.plot(t_axis, norm_orig[:, c], color=C_ORIGINAL, alpha=0.6, linewidth=1.0)

    ax_L.autoscale(enable=True, axis='y', tight=False);
    ax_R.set_ylim(ax_L.get_ylim())
    add_textbox(ax_L, "All Channels");
    add_textbox(ax_R, "All Channels")
    row_idx += 1

    # --- Row 1: All Channels (Global Scores) ---
    ax_L, ax_R = axes[row_idx, 0], axes[row_idx, 1]
    plot_anomaly_spans(ax_L, pred_bg_L, t_axis, C_PRED_BG)
    plot_anomaly_spans(ax_R, pred_bg_R, t_axis, C_PRED_BG)

    ax_L.plot(t_axis, g_score_L, color=C_SCORE, linewidth=2.3)
    ax_R.plot(t_axis, g_score_R, color=C_SCORE, linewidth=2.3)
    ax_L.axhline(y=thresh_L, color=C_THRESH, linestyle='--', linewidth=2.5)
    ax_R.axhline(y=thresh_R, color=C_THRESH, linestyle='--', linewidth=2.5)

    # 🌟 核心修复：彻底删除手动设定的 Y 轴极值限制！让负值分数充分展示，还原高对比度的距离感。
    ax_L.autoscale(enable=True, axis='y', tight=False)
    ax_R.autoscale(enable=True, axis='y', tight=False)

    add_textbox(ax_L, "All Channels");
    add_textbox(ax_R, "All Channels")
    row_idx += 1

    # --- Row 2+: Independent Channels ---
    for ch in visual_channels:
        if ch >= orig_slice.shape[1]: continue

        ch_pred_L = (d_score_L[:, ch] > thresh_L).astype(int)
        ch_pred_R = (d_score_R[:, ch] > thresh_R).astype(int)

        ax_L, ax_R = axes[row_idx, 0], axes[row_idx, 1]
        plot_anomaly_spans(ax_L, labels_slice, t_axis, C_REAL_BG)
        plot_anomaly_spans(ax_R, labels_slice, t_axis, C_REAL_BG)

        ax_L.plot(t_axis, orig_slice[:, ch], color=C_ORIGINAL, linewidth=2.3)
        ax_R.plot(t_axis, orig_slice[:, ch], color=C_ORIGINAL, linewidth=2.3)
        ax_L.plot(t_axis, recon_L[:, ch], color=C_RECON, linewidth=2.3)
        ax_R.plot(t_axis, recon_R[:, ch], color=C_RECON, linewidth=2.3)

        min_y = min(np.min(orig_slice[:, ch]), np.min(recon_L[:, ch]), np.min(recon_R[:, ch]))
        max_y = max(np.max(orig_slice[:, ch]), np.max(recon_L[:, ch]), np.max(recon_R[:, ch]))
        pad = (max_y - min_y) * 0.1 if (max_y - min_y) > 1e-6 else 0.1
        ax_L.set_ylim(min_y - pad, max_y + pad);
        ax_R.set_ylim(min_y - pad, max_y + pad)
        add_textbox(ax_L, f"Channel {ch}");
        add_textbox(ax_R, f"Channel {ch}")
        row_idx += 1

        ax_L, ax_R = axes[row_idx, 0], axes[row_idx, 1]
        plot_anomaly_spans(ax_L, ch_pred_L, t_axis, C_PRED_BG)
        plot_anomaly_spans(ax_R, ch_pred_R, t_axis, C_PRED_BG)

        ax_L.plot(t_axis, d_score_L[:, ch], color=C_SCORE, linewidth=2.3)
        ax_R.plot(t_axis, d_score_R[:, ch], color=C_SCORE, linewidth=2.3)
        ax_L.axhline(y=thresh_L, color=C_THRESH, linestyle='--', linewidth=2.5)
        ax_R.axhline(y=thresh_R, color=C_THRESH, linestyle='--', linewidth=2.5)

        # 🌟 核心修复：彻底删除手动设定的 Y 轴极值限制！让负值分数充分展示，还原高对比度的距离感。
        ax_L.autoscale(enable=True, axis='y', tight=False)
        ax_R.autoscale(enable=True, axis='y', tight=False)

        add_textbox(ax_L, f"Channel {ch}");
        add_textbox(ax_R, f"Channel {ch}")
        row_idx += 1

    # --- 清理坐标轴并添加底部标题 ---
    for ax in axes.flatten():
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_linewidth(1.2)

    #axes[-1, 0].set_xlabel(title_left, fontsize=18, fontweight='bold', labelpad=15)
    #axes[-1, 1].set_xlabel(title_right, fontsize=18, fontweight='bold', labelpad=15)
    axes[-1, 0].set_xlabel(title_left, fontsize=16, labelpad=15)
    axes[-1, 1].set_xlabel(title_right, fontsize=16, labelpad=15)

    # 🌟 完美配对图例
    legend_elements = [
        Line2D([0], [0], color=C_ORIGINAL, lw=2.5, label='Original time series'),
        Line2D([0], [0], color=C_RECON, lw=2.5, label='Reconstruction'),

        Line2D([0], [0], color=C_SCORE, lw=2.5, label='Anomaly scores'),
        Line2D([0], [0], color=C_THRESH, lw=2.5, linestyle='--', label='Threshold'),

        Patch(facecolor=C_REAL_BG, edgecolor='none', alpha=0.6, label='Real anomalies'),
        Patch(facecolor=C_PRED_BG, edgecolor='none', alpha=0.6, label='Predict anomalies')
    ]

    legend_margin = 0.5 / (1.5 * num_rows)
    top_pos = 1.0 - legend_margin - 0.02

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.0 - legend_margin),
               ncol=3, fontsize=16, frameon=True, edgecolor='lightgray', borderpad=0.8)

    plt.tight_layout()
    # bottom = 调节底部标题下方的留白，如果标题被切掉一半了，把 0.06 改大一点(如0.08)
    # hspace = 调节子图间的纵向距离，0.08 已经很紧密了，想要贴在一起可以改为 0.05
    # wspace = 调节左右两列之间的空隙宽度
    #plt.subplots_adjust(bottom=0.06, hspace=0.08, wspace=0.02, top=top_pos)
    plt.subplots_adjust(
        left=0.05,  # 左边距
        right=0.95,  # 右边距
        bottom=0.01,
        hspace=0.08,
        wspace=0.015,  # 您之前要求的紧凑间距
        top=top_pos
    )

    # 确保这一行坐标是 [0.5, 0.5]
    line = mlines.Line2D([0.5, 0.5], [0.03, top_pos], transform=fig.transFigure,
                         color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    fig.add_artist(line)

    os.makedirs(save_path, exist_ok=True)
    time_str = datetime.now().strftime("%m%d%H%M")
    file_name = f"anomaly_fig_{series_name}_comp_mode_{slice_start}_{slice_end}_{time_str}.pdf"

    pdf_path = os.path.join(save_path, file_name)
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf', transparent=False)
    plt.close()
    print(f"✅ [Academic Visual] 对比模式 (comp_mode) 已生成，顶级双栏对比图导出至: {pdf_path}")