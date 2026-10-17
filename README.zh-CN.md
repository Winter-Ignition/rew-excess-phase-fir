# REW Excess Phase FIR · v1

[English](README.md) | 简体中文

从 REW 原始测量提取剩余相位，生成接近全通的定长 FIR，并提供图表、预测和最终 WAV 系数验收。支持立体声、多分频、未知分频和环绕声中的对称配对或独立声道。

**启用剩余相位 FIR 后，必须重新测量并重新校准频响，即使已经做过频响 EQ。** 现有混合相位或剩余相位校正先关闭，必要分频、保护和已调好的系统延迟保留。

## 能做什么

- 原生 IR 文本/浮点 WAV 优先，也支持完整密集频率网格的原始幅频与总相位导出；已有剩余相位必须明确声明。
- 分析脉冲峰与时间参考，默认 1/6 倍频程复数矢量平滑，避免整圈阶跃制造虚假群延时。
- 最低不失真频率至 10 kHz 进行正则化慢变相位反演；较快的局部群延时修正限于 300 Hz 与头动保守估计频率中较低者。
- 各声道独立拟合；配对声道在 10 kHz 以上仅做平滑差分校准，保留共同相位。
- 默认 192 kHz、16384 点、float32 WAV、42.67 ms 公共参考延迟；检查幅度误差、局部带外泄漏和提前能量。
- 输出离线 HTML、PNG、CSV、声道清单和独立验收结果。

## 作为 Skill 使用

将仓库中的 `SKILL.md`、`agents/`、`references/`、`scripts/` 和 `requirements.txt` 放入你的 Skills 目录下的 `rew-excess-phase-fir/` 文件夹。向支持 Skills 的助手提出“使用 rew-excess-phase-fir 校准我的 REW 测量”，按提示补充声道、最低不失真频率、头部移动范围和延迟预算。

具体流程见 [SKILL.md](SKILL.md)，测量方法见 [测量与导出](references/measurement.md)。

## 命令行

Python 3.10+。推荐在隔离环境安装依赖：

```bash
python -m venv .venv
# 激活本地虚拟环境后执行：
python -m pip install -r requirements.txt
python scripts/phase_fir.py inspect left.txt right.txt
python scripts/phase_fir.py design request.json --out design-run
python scripts/phase_fir.py verify design-run
python scripts/test_phase_fir.py
```

请求 JSON 示例、输入限制、复测比较见 [命令与参数](references/commands.md)；算法与验收标准见 [算法说明](references/algorithm.md)。每次设计使用新输出目录，不覆盖已有 manifest。

## 如何判断结果

先看低频群延时峰谷，再看实际 FIR 的增益和提前能量。图上的“应用后”是预测，不能替代加载后的声学复测。低可信区留空，单点测量不能证明整个头部移动区域都有效。公共参考延迟不替代音箱/低音炮的真实时间对齐。

全通滤波也可能增加音乐瞬时峰值；结合报告的保守上界或实际音频峰值测试，给所有声道设置共同余量。每个逻辑输入只加载一份完整滤波，避免重复叠加旧的相位校正。低音管理的挂载位置见 Skill 说明。

## 测试与隐私

当前包含 16 项合成测试，涵盖最小相位/纯延迟、全通恢复、未知三分频、相位换圈、输入边界、声道映射、系数篡改验收、头动范围与整体/局部分工。

仓库不包含个人测量、案例报告、本机路径、凭据或记忆；脚本在本地运行，不自动上传数据或修改音频设备设置。请把自己的输入和输出放在仓库外或 `measurements/`、`outputs/` 中，不要提交。

## 许可证

[MIT](LICENSE)。参考 DRC、REW 等公开资料的稳定机制，相关链接在算法说明中；未复制或调用外部校准引擎，未声称完成外部引擎声学对照。
