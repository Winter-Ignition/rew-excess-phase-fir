# 请求格式与命令

Python 3.10+，依赖 `numpy scipy soundfile matplotlib`。将命令的 `python` 换成已验证的解释器绝对路径；从本 Skill 的 `scripts/phase_fir.py` 运行。

```text
python scripts/phase_fir.py inspect left.txt right.txt
python scripts/phase_fir.py analyze request.json --out analysis-run
python scripts/phase_fir.py design request.json --out design-run
python scripts/phase_fir.py verify design-run
python scripts/phase_fir.py compare remeasure.json --reference design-run --out remeasure-run
python scripts/test_phase_fir.py
```

每次生成用新目录，不覆盖已有 manifest。`analyze` 不输出 WAV；`design` 包含分析、生成、实际回读验收和报告。非零退出码表示未完成或验收失败，失败目录的候选 WAV 不算已通过。

请求示例仅用合成名称，相对路径以 JSON 所在目录为准：

```json
{
  "sample_rate": 192000,
  "taps": 16384,
  "max_latency_ms": 42.67,
  "min_usable_hz": 40,
  "head_half_ranges_cm": [10, 10, 10],
  "existing_phase_correction": "off",
  "channels": [
    {"name":"L", "path":"left.txt", "kind":"frd", "measurement_sample_rate":48000, "crossover_hz":[2000]},
    {"name":"R", "path":"right.txt", "kind":"frd", "measurement_sample_rate":48000, "crossover_hz":[2000]}
  ],
  "pairs": [["L", "R"]],
  "unprocessed_channels": []
}
```

`kind=ir` 为 WAV/标准 REW 单列 IR 文本；`frd` 为原始总相位三列；`excess` 为已有剩余相位三列。WAV 用 `input_channel` 从 0 起选声道。

原始 FRD 采用密集线性频点；稀疏/对数网格仅接受真实未包裹相位并声明 `phase_unwrapped:true`，且需填写测量采样率。不要把包裹相位只改标签来绕过检查。明确声明 `frd` 时允许幅度平直的真实全通测量，自动检测会提示核对类型。

无论是否包裹，FRD 还需完整频带边界（首频点 ≤最低可用频率/16、高端 ≥采样率×.499）；10–22 kHz 的裁切导出改用原生 IR。此检查避免已在合成真值试验中观察到的带外延伸误校准。

每声道可覆盖 `min_usable_hz`、`max_usable_hz`；`crossover_hz` 为列表，未知 `[]`，三分频填两个近似点。多配对如 `[["L","R"],["SL","SR"]]`，中置不配对。真正的 LFE 逻辑输入可设置 `role:lfe`（最高可用频率默认 120 Hz，可覆盖），按单侧校准并在其上限退出；它不是共享低音炮输出。窄带声道使用其可用频点拟合，不能把带外噪声当作真实高频剩余相位。

`smoothing_octaves` 默认 1/6，`adaptive_smoothing=true`。统一慢变基底使用 `global_basis_sigma_octaves=.65`、`global_regularization=3`；这些与分析用的 1/6 平滑不同。局部细节上限为 `min(local_detail_cap_hz=300,头动保守估计频率)`，`local_depth` 为 0–1；`local_highpass_octaves=.8` 将慢变偏移从局部目标中去掉。旧的 `lf_local_high_hz=300`、`lf_stop_hz=400` 只保留作低频统计/显示范围，不再决定完整滤波或局部项的退出范围。每声道实际范围见 manifest 的 `calibration_ranges`。

`optimize=true` 按实际系数搜索。`max_gain_error_db=.03`，`stop_gain_error_db=.005`，`stop_phase_error_deg=.1`。带外检查从实际局部上限开始，对比包含完整慢变校准的基线。`transition_max_added_gd_ms=.25`、`transition_rms_added_gd_ms=.12` 限制局部退出后的额外起伏。`max_far_pre_energy=.005`、`max_added_pre_energy=.0005` 是提前 1 ms 总能量/增量预算，不是听觉无感阈值。延迟少则减弱局部结构，无法形成有效补偿会退出。

IR 可声明 `direct_peak_s`（从首样本计）、`timing_reference` 和 `time_origin_s`。已应用的麦克风校准不重复输入；确定尚未应用且有可信复校准才传 `calibration:{"path":"mic.txt","trust_phase":true}` 和 `calibration_state:"not_applied"`。只有幅度校准时不猜相位，按自洽原始链分析并注明麦克风自身剩余相位未知。

低音管理增加 `bass_management:true`、`mount:"before_bass_management"`（推荐）或 `common_all_branches`，工作记录保留分频点。不能把各输入的不同 FIR 装到共享低音炮输出。`unprocessed_channels` 导出相同公共纯延迟，不能替用户做真实到达时间对齐。

输出完整 WAV、`manifest.json` 声道清单、独立 `acceptance.json`、私有 `analysis.npz`、CSV、PNG 和离线 HTML。每份 WAV 的 `role=complete`，只加载一次。

复测请求可设 `existing_phase_correction:"this_filter"` 表示只打开本次交付滤波，其他待替换的相位音效仍关闭；这个状态仅 `compare` 接受，不能再用它生成叠加补偿。比较固定原先全部平滑混合权重，并只在分析中对齐复测的线性时间规约，不修改播放延迟。
