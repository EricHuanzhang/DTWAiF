'''
* @author: EmpyreanMoon
*
* @create: 2024-08-25 20:20
*
* @description: various forms of frequency loss
'''

import torch
from einops import rearrange
import numpy as np


class frequency_loss(torch.nn.Module):
    def __init__(self, configs, keep_dim=False, dim=None):
        super(frequency_loss, self).__init__()
        self.keep_dim = keep_dim
        self.dim = dim
        if configs.auxi_mode == "fft":
            self.fft = torch.fft.fft
        elif configs.auxi_mode == "rfft":
            self.fft = torch.fft.rfft
        else:
            raise NotImplementedError
        self.configs = configs
        self.use_ArcTanLoss = configs.use_ArcTanLoss

        if configs.mask:
            self._generate_mask()
        else:
            self.mask = None

    def _generate_mask(self):
        if self.configs.add_noise and self.configs.noise_amp > 0:
            seq_len = self.configs.pred_len
            cutoff_freq_percentage = self.configs.noise_freq_percentage
            if self.configs.auxi_mode == "rfft":
                cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            elif self.configs.auxi_mode == "fft":
                cutoff_freq = int((seq_len) * cutoff_freq_percentage)
                low_pass_mask = torch.ones(seq_len)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1)
        else:
            self.mask = None
    """
    作用：计算频域损失
    输入：
        outputs：重构后的频域表示
        batch_y：原始时域数据（注意，是RevIN后的数据）
    """
    def forward(self, outputs, batch_x):
        # 1. 执行 FFT (保持原逻辑)
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = self.fft(outputs, dim=1)
        # 2. 关键修改：转换为对数幅度
        epsilon = 1e-8  # 防止 log(0)

        # 原始值/目标值处理
        target_fft = self.fft(batch_x, dim=1)
        target_mag = target_fft.abs()
        target_log = torch.log(target_mag + epsilon)  # Log 变换

        # 预测值处理
        pred_mag = frequency_outputs.abs()
        pred_log = torch.log(pred_mag + epsilon)  # Log 变换

        # fft shape: [B, P, D]
        if self.configs.auxi_type == 'complex':
            # 自然地处理了相位和幅度的耦合。当幅度接近0时，损失平滑趋零。
            # 采用数值稳定的 Complex L1 Loss，避免fre loss 震荡不收敛的问题
            loss_auxi = frequency_outputs - target_fft
        elif self.configs.auxi_type == 'complex-phase':
            loss_auxi = (frequency_outputs - self.fft(batch_x, dim=1)).angle()
        elif self.configs.auxi_type == 'complex-mag-phase':
            loss_auxi_mag = (frequency_outputs - self.fft(batch_x, dim=1)).abs()
            loss_auxi_phase = (frequency_outputs - self.fft(batch_x, dim=1)).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        elif self.configs.auxi_type == 'phase':
            loss_auxi = frequency_outputs.angle() - self.fft(batch_x, dim=1).angle()
        elif self.configs.auxi_type == 'mag':
            loss_auxi = pred_mag - target_mag
        elif self.configs.auxi_type == 'mag-phase':
            loss_auxi_mag = frequency_outputs.abs() - self.fft(batch_x, dim=1).abs()
            loss_auxi_phase = frequency_outputs.angle() - self.fft(batch_x, dim=1).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        else:
            raise NotImplementedError

        if self.mask is not None:
            loss_auxi *= self.mask

        # [Fix] 核心修复：引入 ArcTan 鲁棒机制
        if self.use_ArcTanLoss:
            # 无论 loss_auxi 是复数还是实数，计算其模的平方作为误差能量
            # abs() 对于复数返回模，对于实数返回绝对值
            sq_error = loss_auxi.abs() ** 2

            # 应用 ArcTan 截断大梯度
            loss_final = torch.atan(sq_error)

            # 均值聚合
            loss_auxi = loss_final.mean(dim=self.dim,
                                        keepdim=self.keep_dim) if self.configs.module_first else loss_final.mean(
                dim=self.dim, keepdim=self.keep_dim)
        else:
            if self.configs.auxi_loss == "MAE":
                loss_auxi = loss_auxi.abs().mean(dim=self.dim,
                                                 keepdim=self.keep_dim) if self.configs.module_first else loss_auxi.mean(
                    dim=self.dim, keepdim=self.keep_dim).abs()  # check the dim of fft
            elif self.configs.auxi_loss == "MSE":
                loss_auxi = (loss_auxi.abs() ** 2).mean(dim=self.dim,
                                                        keepdim=self.keep_dim) if self.configs.module_first else (
                        loss_auxi ** 2).mean(dim=self.dim, keepdim=self.keep_dim).abs()
            else:
                raise NotImplementedError

        return loss_auxi


class frequency_criterion(torch.nn.Module):
    def __init__(self, configs):
        super(frequency_criterion, self).__init__()
        # 1. 实例化基础频域损失函数
        # keep_dim=True 表示计算 Loss 后不求平均，保留维度以便后续映射回时间轴
        self.metric = frequency_loss(configs, dim=1, keep_dim=True)
        # 2. 定义滑动窗口参数
        self.patch_size = configs.inference_patch_size
        self.patch_stride = configs.inference_patch_stride
        self.win_size = configs.seq_len
        # 3. 计算可以切分出多少个 Patch
        # 公式: N = (L - P) / S + 1
        self.patch_num = int((self.win_size - self.patch_size) / self.patch_stride + 1)
        # 4. 计算末尾剩余无法被完整窗口覆盖的长度 (Padding)
        self.padding_length = self.win_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)

    def forward(self, outputs, batch_x):
        # outputs shape: [Batch, Seq_Len, N_Vars]
        # batch_x shape: [Batch, Seq_Len, N_Vars]

        # 1. unfold: 在时间维度 (dim=1) 上进行滑动窗口切分
        # output_patch shape: [Batch, Patch_Num, N_Vars, Patch_Size]
        output_patch = outputs.unfold(dimension=1, size=self.patch_size,
                                      step=self.patch_stride)
        # 获取维度信息
        b, n, c, p = output_patch.shape
        # 2. rearrange: 将 Patch 展平到 Batch 维度
        # 变换为 [(Batch * Patch_Num), Patch_Size, N_Vars]
        # 这样做的目的是把每个 Patch 当作一个独立的短序列来计算 Loss
        output_patch = rearrange(output_patch, 'b n c p -> (b n) p c')

        # 对真实标签 batch_x 做同样的操作
        x_patch = batch_x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        x_patch = rearrange(x_patch, 'b n c p -> (b n) p c')
        #-------第二步：计算局部频域损失--------------------
        # 作用：对每个小片段打分。
        # 关键点：如果一个片段的频域误差很大，我们认为这个片段里的每一个时间点都分担这个误差。
        # 3. 计算每个 Patch 的频域损失
        # 调用 frequency_loss 对每个小片段做 FFT 并计算差异
        # 注意：main_part_loss shape: [(Batch * Patch_Num), 1, N_Vars] (因为 dim=1 被 reduce 了但 keep_dim=True)
        main_patch_loss = self.metric(output_patch, x_patch)

        # 4. 广播损失值
        # 虽然 FFT Loss(main_patch_loss) 是针对整个 Patch 算出一个数，但我们需要把它赋给 Patch 里的每一个时间点
        # repeat 后 shape: [(Batch * Patch_Num), Patch_Size, N_Vars]
        main_patch_loss = main_patch_loss.repeat(1, self.patch_size, 1)

        # 恢复 Batch 维度
        # shape: [Batch, Patch_Num, Patch_Size, N_Vars]
        main_patch_loss = rearrange(main_patch_loss, '(b n) p c -> b n p c', b=b)

        # -------第三步：Scatter & Average（重叠部分聚合）--------------------
        # 因为窗口是滑动的（通常 Stride=1），同一个时间点（比如第 50 秒）会被多个 Patch 包含。
        # 我们需要把所有包含第 50 秒的 Patch 的评分加起来取平均。
        # 重叠累加 (Overlap-Add)：如果第 50 秒在 Patch A 中误差是 0.5，在 Patch B 中误差是 0.7。
        # 平均化：那么第 50 秒的最终得分是 (0.5 + 0.7) / 2 = 0.6。
        # 这消除了单个窗口边缘效应带来的抖动，使得异常评分曲线更加平滑且准确。
        # 5. 构造索引矩阵
        # end_point: 最后一个完整 Patch 结束的位置
        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1

        # 生成每个 Patch 的起始和结束索引
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        # 构造 scatter 需要的 indices 张量
        # 这段代码生成了一个映射表，告诉程序：第 n 个 Patch 的第 p 个点，对应原序列的哪个位置。
        indices = torch.tensor([range(start_indices[i], end_indices[i]) for i in range(n)]).unsqueeze(0).unsqueeze(-1)
        indices = indices.repeat(b, 1, 1, c).to(main_patch_loss.device)

        # 6. 初始化全长损失画布 (Canvas)
        # main_loss shape: [Batch, Patch_Num, Valid_Seq_Len, N_Vars] (注意这里 dim=1 初始化时大小不对，看下一行)
        # 修正：这里的 main_loss shape 实际上是 [Batch, Patch_Num, Seq_Len_Main_Part, N_Vars]
        # 但 scatter_ 的目标是把 patch loss 填回去。
        main_loss = torch.zeros((b, n, self.win_size - self.padding_length, c)).to(main_patch_loss.device)

        # 7. Scatter 操作 (核心聚合)
        # 将每个 Patch 的 loss (src) 按照 indices 映射回 main_loss 的时间轴上
        # 这里的 dim=2 是被填充的时间维度。
        main_loss.scatter_(dim=2, index=indices, src=main_patch_loss)

        # 8. 计算重叠次数并求平均
        # non_zero_cnt: 统计每个时间点被多少个 Patch 覆盖了
        non_zero_cnt = torch.count_nonzero(main_loss, dim=1)

        # Sum / Count = Average
        # 最终得到每个时间点的平均频域重构误差
        main_loss = main_loss.sum(1) / non_zero_cnt

        # -------第四步：处理 Padding（尾部残余）---------------------
        # 9. 处理尾部不够一个完整 Patch 的数据
        if self.padding_length > 0:
            # 直接取最后 padding_length 长度的数据计算 Loss
            padding_loss = self.metric(outputs[:, -self.padding_length:, :], batch_x[:, -self.padding_length:, :])
            # 同样进行广播
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            # 拼接到主 Loss 后面
            total_loss = torch.concat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss

class ArcTanLoss(torch.nn.Module):
    """
    Robust Loss Function:
    类似于 Huber Loss，但在误差极大时提供更平滑的梯度衰减 (有界梯度)。
    """
    def __init__(self):
        super(ArcTanLoss, self).__init__()

    def forward(self, pred, target):
        # 计算平方误差
        diff = torch.square(pred - target)
        # 应用 ArcTan 限制大误差的影响
        loss = torch.mean(torch.atan(diff))
        return loss