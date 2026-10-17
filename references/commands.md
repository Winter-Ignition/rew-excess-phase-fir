# v2 请求与命令

Python 3.10+，依赖见 `requirements.txt`。使用已经验证的本地解释器，不改全局环境。

```text
python scripts/phase_fir.py inspect left.txt right.txt
python scripts/phase_fir.py analyze request.json --out analysis-run
python scripts/phase_fir.py design request.json --out design-run
python scripts/phase_fir.py verify design-run
python scripts/phase_fir.py compare remeasure.json --reference design-run --out remeasure-run
python -m unittest discover -s scripts -p "test_*.py"
```

每次生成用新目录。`design` 包含实际 WAV 验收；非零退出的候选不算通过。

## 新基线

相对路径均相对 JSON 所在目录。以下为通用参数示例，不代表系统测量。

```json
{
  "sample_rate": 192000,
  "taps": 16384,
  "max_latency_ms": 42.67,
  "min_usable_hz": 40,
  "measurement_valid_low_hz": 25,
  "head_half_ranges_cm": [10, 10, 10],
  "existing_phase_correction": "off",
  "channels": [
    {"name":"L", "path":"left.txt", "kind":"frd", "measurement_sample_rate":48000, "crossover_hz":[2000]},
    {"name":"R", "path":"right.txt", "kind":"frd", "measurement_sample_rate":48000, "crossover_hz":[]}
  ],
  "pairs": [["L", "R"]]
}
```

`min_usable_hz` 是实际音量下最低不失真频率；`measurement_valid_low_hz` 是非噪声主导的测量下限。实际校准起点取二者较高值，可逐声道覆盖。噪声下限不能由扫频起点或音箱标称下潜直接替代。未知可省略/null，但会明确低频参考未认证。低频等效延伸的候选与敏感度保存在本地分析摘要。

`kind=ir`：浮点 WAV 或保留头部的 REW 单列原始 IR；`frd`：频率、dB、总相位三列；`excess`：已计算的剩余相位，不再分解。WAV 可用 `input_channel` 从 0 起选声道。稀疏/对数包裹总相位会拒绝；确实未包裹才声明 `phase_unwrapped:true`。密集 FRD 保留全 FFT 高端；低端优先完整，若裁切则需明确噪声边界、首点不高于该边界且保持密集线性网格。不能只改类型标签绕过检查。

`crossover_hz` 可未知 `[]`，三分频可填两个近似点。`max_usable_hz` 限制窄带声道；`measurement_valid_high_hz` 默认 22000，标注可信测量高端。`role:lfe` 默认上限 120 Hz，独立处理，不是共享低音炮物理输出。`pairs` 支持多个不重叠配对，中置等不配对；`unprocessed_channels` 输出相同公共纯延迟。

## 已加载 FIR 后的复测重做

请求沿用声道测量和参数，改为：

```json
{
  "existing_phase_correction": "this_filter",
  "reference_directory": "previous-v2-design",
  "loaded_filters": [
    {
      "path": "previous-complete.wav",
      "sha256": "实际旧 WAV 的 SHA256",
      "channels": ["L", "R"],
      "delay_samples": 8192
    }
  ]
}
```

此片段需合并进完整请求。`loaded_filters.channels` 是该 WAV 从左到右的声道映射，必须覆盖所有设计声道且不重复。旧 WAV 采样率/长度必须匹配，近全通且只有相同的公共参考延迟。缺文件、哈希错误、映射不全或状态矛盾时停止，不猜版本。可直接输入 v1 WAV，但 `reference_directory` 仅接受 v2 配套清单；无原参考时明确新的低端参考不能与旧报告直接作绝对值比较。

`reference_directory` 冻结低端模型形状、窗口和平滑设置；只允许导出相对电平变化，不重拟合模型形状。新文件是 `H_new ≈ H_loaded × C_residual` 的一次定长重渲染，保留一份公共延迟。它是完整替换文件，播放时移除旧文件。

若只是验证已交付效果，使用 `compare`、`existing_phase_correction:this_filter` 和原设计目录，不需要提供旧 FIR 再生成。它按分频带报告相位、群延时及预测偏差，线性时间差只用于分析。

## 约束和输出

默认分析平滑 `smoothing_octaves=1/6`、`adaptive_smoothing=true`。宽带 Gaussian 基底 sigma `.35` 倍频程、间隔 `.25`、曲率惩罚 `.8`；与分析平滑不同。局部上限为 `min(local_detail_cap_hz=300,头动估计,声道上限)`，未知移动范围保守取 200 Hz。`local_depth=0..1`，`local_highpass_octaves=.8`；300/400 Hz 旧字段仅用于统计/图示，不是宽带退出点。

默认 `max_gain_error_db=.03`；局部带外 `stop_gain_error_db=.005`、`stop_phase_error_deg=.1`。局部退出新增群延时峰值 `.25 ms`、RMS `.12 ms`。提前 1 ms 总能量 `max_far_pre_energy=.005`、局部增量 `max_added_pre_energy=.0005`；替换时额外与旧实际 FIR 比较 1 ms 和 5 ms 提前能量。它们是工程预算，不是听觉无感阈值。

`pair_end_hz=20000` 是可配置高频退出上限，实际候选受可信频带限制，不固定为该频率。默认高频残余左右差 P95 ≤5°，8–10 kHz 群延时峰 ≤40 µs 或不明显恶化的原值；这些是系数预测门限，不能冒充声学复测通过。组合相位误差默认 ≤2°，300 Hz 以上 ≤.3°；有限长度不能表示的目标会拒绝。

`notch_threshold_db=-6` 标识相对 1/3 倍频程背景的陷波，降低拟合权重；低频陷波处实际新增相位默认不超过 8°。若整体平滑校准在陷波处也需要较大相位，要看图判断；不能为了过验收直接放宽阈值。

`direct_peak_s` 指定从文件首样本计的直达峰。可信且尚未应用的复麦克风校准可声明 `calibration:{"path":"mic.txt","trust_phase":true}` 与 `calibration_state:not_applied`。不重复校准，不猜只有幅度表的麦克风相位。

低音管理声明 `bass_management:true`，挂载点 `mount:before_bass_management` 或 `common_all_branches`，同一输入所有分支使用同一 FIR；不在共享低音炮输出叠多个输入各自校准。

输出完整 FLOAT WAV、`manifest.json`、`acceptance.json`、本地 `analysis.npz`、曲线、PNG 和离线 HTML。保持这些文件配套供复测。首次使用和最终交付均须提醒启用 FIR 后重新测量、重新校准频响。
