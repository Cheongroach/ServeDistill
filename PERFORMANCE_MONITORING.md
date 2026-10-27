# ServeDistill 性能监控与有效性认证

本文是独立的复现手册。先确定要测的量，再选工具；每一步都说明输入、参数、输出和能支持的结论。项目教学顺序见 [54 步教程](docs/一步步认识ServeDistill项目-friendly.md)。

2026-10-27 状态：本轮 none/nsys/PyTorch 副本均通过 update162 的运行门；实际 CUDA/NVTX、GPU kernel trace 和 91 个原始输出文件的 SHA256 已独立核验。[本轮导出](reports/monitoring/profile-professional-v1/PUBLIC_EXPORT_MANIFEST.json)保留版本、遥测、统计 CSV、压缩 PyTorch trace、失败记录与原始/公开哈希。NCU 2026.1.1 在 driver 535.261.03 上报告不兼容；串行兼容性任务已提交，尚无本轮有效 NCU counter。DCGM、perf、compute-sanitizer 的本轮环境验证尚未完成。Nsight Systems 的命令名是 `nsys`，不是 `nvsys`。

## 第一阶段：确定测量问题与边界

### 第 1 步：把质量、训练时间和服务时间分开

| 问题 | 主证据 | 为什么选它 |
|---|---|---|
| 一次 learner 更新是否更快？ | 无 profiler 的固定轨迹、同步重复计时 | 控制 token/shape，排除 profiler 开销 |
| 完整训练是否更快？ | 训练 elapsed/phase_seconds、真实有效 token | 计入 rollout、Teacher、DEV 等阶段 |
| 时间花在哪里？ | Nsight Systems CUDA/NVTX timeline | 可以看到 API、kernel、复制、同步与间隙 |
| 哪个框架 operator 产生这些工作？ | PyTorch profiler shape/operator trace | 将系统 kernel 关联回框架调用 |
| kernel 的硬件瓶颈是什么？ | Nsight Compute counters | 分析吞吐、内存、occupancy、launch 条件 |
| GPU 是否降频、显存紧张？ | nvidia-smi，可选 DCGM | 保留设备状态随时间的变化 |
| GPU 是否等待 CPU/IO？ | `/proc` 进程树、perf stat | 补充 host 的 CPU、fault、RSS、IO 证据 |
| 是否有非法显存访问？ | Compute Sanitizer | 检查 CUDA 执行有效性 |
| 服务吞吐是否有效？ | `http_benchmark` 请求级评分与时延 | 只完成错误答案的吞吐不能冒充有效吞吐 |
| 总成本是否改善？ | 所有 Pod 的终态分配区间 | 包含失败、准备和评测，而非只看热循环 |

停下来回答：你要证明的是“kernel 更快”还是“端到端得到正确答案更便宜”？

### 第 2 步：保留计时分母

`response_tokens` 指回答 mask 中有效动作，包含第一次 EOS。训练日志中的累计值不包含 gold anchor 的额外 CE token；anchor 是新增计算，不能把 sampled response token 数解释成全部训练动作数。使用 anchor 时另看 `anchor_ce_loss` 和新增 forward/backward 范围。

learner tokens/s 的固定轨迹比较由 `benchmark_microbatch.py` 实现；完整训练 seconds/update 包含不同方法实际执行的工作。两者都不等于服务 requests/s。

停下来回答：同 prompts/update 的 AOPD 与 SFT，是否具有相同 token、Teacher 和时间预算？

### 第 3 步：把 profiler 开销排除出速度结论

各工具在独立进程从同一 checkpoint 恢复。NCU replay、PyTorch shape/memory 记录、nsys trace 都可能改变执行。`manifest.json` 固定 `profiler_times_eligible_for_speed_claim=false`。

速度主证据使用历史 [expanded/performance.json](reports/expanded/performance.json) 及无 profiler 原始 measurements；采集副本用于解释机制。必要时比较 none 与各 profiler 的同 update 墙钟，报告观测开销，不能把 profiler 时间移回性能主表。

停下来回答：NCU 比基线慢十倍，是否说明原应用的 kernel 就慢十倍？

## 第二阶段：准备一致的输入

### 第 4 步：安装 Python 依赖

```bash
python -m pip install -e '.[data,profiling,test]'
python -m pytest -q tests
```

保留宿主 CUDA 环境提供的 PyTorch，先核对 CUDA/驱动兼容性。核心 Transformers/PEFT 版本在 `pyproject.toml` 中固定；运行时记录真实版本，安装声明不替代环境审计。

CPU 回归在当前隔离依赖环境通过 136 项，包含完整更新的 microbatch 一致性、空/缺失 counter 拒绝、真实 NVTX CSV 结构、CPU-only trace 拒绝和数值漂移检查。CPU 测试不证明 GPU 执行、计数器权限或新方法业务质量。

### 第 5 步：指定模型、数据和 checkpoint

```bash
export STUDENT=/absolute/path/student/model
export TEACHER=/absolute/path/teacher/model
export DATA=/absolute/path/training-data
export TRAIN_CONFIG=/absolute/path/original-training-config.json
export CHECKPOINT=/absolute/path/checkpoint-0160
export TOOLS_ROOT=/absolute/path/nsight-tools
```

`DATA` 目录含 `train.jsonl` 和 `dev.jsonl`。输入使用你的实际路径；不要把这些变量留为空。`--config` 是源训练的 immutable config；`--profile` 是观测配置，二者用途不同。

checkpoint 必须含 `state.json`、`training.pt`、`adapter_model.safetensors` 和 adapter 配置。模型 manifest 应验证权重、tokenizer 和配置内容。

`experiments/profile-professional-v1/` 和 `experiments/ncu-compatibility-v1/` 保存各自实际执行的 code/config archive 与逐文件 SHA256。复跑历史采集时先在新目录解压对应 `source.tar.gz`，核对 manifest，再提供外部数据、模型和 checkpoint。当前 main 的后续监控改进与冻结采集源码分别记录，不能把当前 head 当成历史实际执行版本。

### 第 6 步：固定观测窗口

默认配置假定源 checkpoint step=160：

```text
冷恢复 160 → 完整 warmup 161 → 捕获 update 162 → 保存诊断状态
```

其他起点需显式传 `--first-update N --last-update M`，其中 `N > checkpoint_step+1`、`N≤M≤training_steps`。runner 拒绝重复输出目录和没有 warmup 的窗口。

`microbatch=4` 是诊断配置覆写，发生在原 checkpoint 配置恢复验证之后；原始和观测配置均保存。它不授权更改正式训练或选择结果。

停下来回答：为什么不能直接把 step=640 的最终模型放进 step=162 的默认窗口？

## 第三阶段：用 none 做恢复和有界观测

### 第 7 步：先运行没有重型 profiler 的副本

```bash
python scripts/profile_run.py \
  --profile configs/monitoring/baseline.json \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-none
```

用 `--dry-run` 只查看最终 argv；它不生成有效性结果，也不执行恢复。工具由参数列表调用，不通过 shell 拼接模型路径。

| 参数 | 作用 | 默认/约束 |
|---|---|---|
| `--profile` | 有界观测配置 | schema_version=1 |
| `--validity` | 运行有效性门 | `configs/validity/runtime.json` |
| `--student/--teacher` | 实际模型目录 | 必填；配套 tokenizer 内容一致 |
| `--data` | 已固定训练数据 | train/dev 文件 |
| `--config` | immutable training config | 必须与保存状态一致 |
| `--resume` | 完整 checkpoint | 不接受只有 adapter 的目录 |
| `--output` | 新诊断输出目录 | 已存在则拒绝 |
| `--tools-root` | 已解压工具搜索根 | 可选 |
| `--first-update/--last-update` | 覆写观察窗口 | 必须保留 warmup |

### 第 8 步：阅读基础产物

| 文件 | 内容 | 如何判断 |
|---|---|---|
| `manifest.json` | 最终 argv、配置、代码/data/state SHA256 | 输入与实际执行必须匹配 |
| `application.log` | 真实 stdout/stderr | 保留 trace 缺失和数值错误 |
| `application/train/restoration-*.json` | adapter/Adam/RNG 恢复指纹 | `passed=true` 才有冷恢复证据 |
| `application/train/metrics.jsonl` | update/version/loss/梯度/有效 token | 目标窗口必须完整 |
| `application/observation/configuration.json` | 原配置与观测覆写 | 检查 microbatch 等诊断变更 |
| `application/observation/host-spans.json` | host 范围起止、wall seconds | CUDA 异步工作可能跨越范围 |
| `gpu-telemetry.csv` | 设备级状态 | 原始 CSV 表头含单位 |
| `host-telemetry.jsonl` | 主机总量和可见进程树 | 不归因到不可见的其他租户 |
| `tool-audit.json` | 工具命令/版本/退出码 | 每个导出单独核查 |
| `validity.json` | 窗口、finite、概率、更新门 | 不是业务质量证明 |
| `outputs.json` | 输出文件 bytes/SHA256 | 校验归档是否完整 |

产物 manifest 不对自己递归求哈希；原输入哈希与公开脱敏导出哈希也须分别保存。

## 第四阶段：安装可追溯的官方 Nsight

### 第 9 步：使用锁定包安装

```bash
python scripts/install_profile_tools.py --root "$TOOLS_ROOT" --tools nsys ncu
```

使用官方 [Ubuntu 22.04 amd64 仓库](https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/Packages.gz)，校验 SHA256 后通过 `dpkg-deb -x` 解压，无需安装到系统目录。下载/解压仍需要网络、磁盘空间和 Linux x86_64。

| CLI | 锁定版本/包 | 选择理由 |
|---|---|---|
| nsys | `NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb` | 与既有有效采集使用的官方 CLI 保持一致 |
| ncu | `nsight-compute-2026.1.1_2026.1.1.2-1_amd64.deb` | 官方版本化包，大小 329,884,926 bytes，内容 SHA256 已从官方 index 核对 |

完整 digest 见 `configs/monitoring/tools.lock.json`。`installation-audit.json` 保存实际 binary、version 和包指纹。现成工具可通过 `SERVEDISTILL_NSYS` / `SERVEDISTILL_NCU` 指定绝对可执行路径，实际版本仍必须归档；suite 显式使用它安装的锁定二进制。

当前锁文件保留首次请求的 NCU 2026.1.1，便于复现失败；它未通过本轮 R535 驱动采集。不要把这份锁文件解释为当前 worker 的已验证兼容推荐。兼容版本必须经实际 capture 后更新，并保留原失败记录。

兼容性配置 `configs/monitoring/ncu-compatibility-candidates.json` 固定官方 2025.2.1 → 2024.1.1 → 2023.3.1 的顺序、包大小和 SHA256。runner 在一张 GPU 上串行执行实际训练副本，首个完整 runtime/counter 门通过的版本才入选：

```bash
python scripts/run_ncu_compatibility.py \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --checkpoint "$CHECKPOINT" \
  --tools-root "$TOOLS_ROOT" --output results/ncu-compatibility
```

每版输出 `capture/`、`runner.log`；`cases.json` 保留所有尝试和失败，`completed.json` 保存入选版本与候选配置 SHA256。下载包通过校验只证明来源一致，query-metrics 成功只证明查询可用，都不替代实际采集。

停下来回答：包下载地址相同但 SHA256 不同，应该继续解压吗？

## 第五阶段：Nsight Systems 系统时间线

### 第 10 步：运行有界 CUDA/NVTX capture

```bash
python scripts/profile_run.py \
  --profile configs/monitoring/nsys.json --tools-root "$TOOLS_ROOT" \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-nsys
```

runner 对训练应用包装的关键参数：

```text
nsys profile --trace=cuda,nvtx,osrt
  --capture-range=cudaProfilerApi --capture-range-end=stop
  --sample=none --cpuctxsw=none --force-overwrite=true --output=<capture>
  <cold-restored training application>
```

| 参数 | 技术解释与选型理由 |
|---|---|
| `trace=cuda,nvtx,osrt` | 将 GPU/复制/API 与训练范围、host runtime 对齐 |
| `capture-range=cudaProfilerApi` | `cudaProfilerStart/Stop` 只围住预定 update，排除加载和 warmup |
| `capture-range-end=stop` | 停止采集后应用仍保存诊断状态 |
| `sample=none` | 首轮不引入 CPU call-stack 采样开销 |
| `cpuctxsw=none` | 兼容受限容器；CPU 调度深入分析再单独采集 |
| `force-overwrite=true` | 工具临时同名文件允许重建，但 runner 的根输出目录仍拒绝重用 |

### 第 11 步：阅读 timeline 和统计导出

GUI 打开 `capture.nsys-rep`。runner 自动导出：

```bash
nsys stats --report=cuda_gpu_kern_sum --format=csv capture.nsys-rep
nsys stats --report=cuda_api_sum --format=csv capture.nsys-rep
nsys stats --report=nvtx_sum --format=csv capture.nsys-rep
nsys stats --report=cuda_gpu_mem_time_sum --format=csv capture.nsys-rep
```

kernel 表提供名称、调用次数和时长；CUDA API 表提供 launch、同步、allocation 等 host 调用统计；NVTX 表提供命名 host 范围。不同 stream 的 kernel 会重叠，时长总和不能直接当成 timeline wall time。NVTX 时间和 GPU kernel 时间也不是同一单位的工作归属口径。

`scripts/analyze_profile.py` / `scripts/analyze_expanded_profiles.py` 处理项目归档 trace；历史 CSV、kernel 摘要和 timeline 图在 `reports/profiling` 与 `reports/expanded/profiles`。

先找到有代表性的长耗时或高频 kernel，再到 NCU。仅凭 kernel 数量下降，不能证明完整训练已经加速。

停下来回答：timeline gap、同步 API 和 kernel overlap 分别需要什么证据？

## 第六阶段：Nsight Compute kernel counter

### 第 12 步：从低开销、少 kernel 的采集开始

```bash
python scripts/profile_run.py \
  --profile configs/monitoring/ncu.json --tools-root "$TOOLS_ROOT" \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-ncu
```

关键包装命令：

```text
ncu --profile-from-start off --nvtx --nvtx-include learner_forward/
  --replay-mode kernel --launch-count 8
  --metrics gpu__time_duration.sum,
            sm__throughput.avg.pct_of_peak_sustained_elapsed,
            dram__throughput.avg.pct_of_peak_sustained_elapsed
  --force-overwrite --export <capture> <application>
```

| 参数/metric | 输入规则与技术解释 |
|---|---|
| `profile-from-start off` | 等待应用 cudaProfilerStart，避免采集模型加载与 warmup kernel |
| `nvtx-include learner_forward/` | `/` 表示 push/pop NVTX range；只选择 learner forward 范围 |
| `launch-count=8` | 最多采集八个匹配 launch，限制 replay 与产物体积；这是诊断样本，不是所有 kernel |
| `replay-mode=kernel` | 重放单 kernel 采集不同 counters，可能保存/恢复其访问内存并改变缓存行为 |
| `gpu__time_duration.sum` | GPU kernel duration，单位由 CSV Metric Unit 给出，不擅自假定秒 |
| `sm__throughput...pct...` | SM 吞吐相对持续峰值的比例，是相关子单元聚合量；不等于纯 FLOPs 峰值比 |
| `dram__throughput...pct...` | DRAM 吞吐相对持续峰值比例；结合 kernel 读写字节与缓存命中解释 |

runner 保存 `available-metrics.txt`。指标是否支持要看当前 H20、驱动和实际 CLI，不能只看配置名字。

### 第 13 步：只对定位到的代表 kernel 深挖

`configs/monitoring/ncu_deep.json` 用 `SpeedOfLight`、`MemoryWorkloadAnalysis`、`LaunchStats`、`Occupancy` 四个 section，默认最多一个 launch。它减少采集数量，但 counter passes 可能比基础配置更多。

复制配置到自己的诊断目录后，可以在 `options.kernel_name` 填从 nsys 得到的 regex，例如 `regex:.*gemm.*`；没有匹配 kernel 时必须记为未取得代表性 counter。不要假设第一个 kernel 就是主要瓶颈。

```bash
ncu --list-sections
ncu --query-metrics
python scripts/profile_run.py \
  --profile configs/monitoring/ncu_deep.json --tools-root "$TOOLS_ROOT" \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-ncu-deep
```

Occupancy 描述驻留线程/warp 与资源约束；高 occupancy 不保证高 Tensor Core 吞吐。MemoryWorkloadAnalysis 检查 DRAM/L2/其他层次的流量与瓶颈。LaunchStats 检查 grid、block、寄存器和共享内存；它们一起解释 kernel 受限原因。

### 第 14 步：保留原报告、CSV 和采集失败

```bash
ncu --import capture.ncu-rep --csv --page raw > ncu-metrics.csv
```

输出 `capture.ncu-rep` 可用 GUI 打开；`ncu-metrics.csv` 的 metric name、value、unit、kernel 和 launch 信息应一并保存。版本、可用指标列表、counter 权限错误和非零导出退出码都在 audit/log 中。

`validity.json.report` 要求 raw export 退出 0、至少一个真实 kernel、每个 kernel 的请求 metric 齐全、带单位的有限非负数，以及正的 kernel duration。空 CSV、`N/A`、缺失 DRAM metric 或导出失败均不能通过。Nsight Systems 同样要求 kernel/API/NVTX 三类真实统计行与 learner 范围；PyTorch trace 要求实际 GPU kernel 和 learner 范围。

出现 `ERR_NVGPUCTRPERM` 时，仅说明当前运行身份不能读取所需计数器。保留失败证据，并通过平台支持的 counter 权限配置解决；不要把 nvidia-smi utilization 填进 NCU 缺失字段，也不要在共享宿主上擅自修改驱动参数。

NCU 和 DCGM profiling counters 不应在同一采集窗口竞争；必要时关闭你自己的 DCGM counter watcher 后单独运行。涉及 NCCL/并发 stream 的 kernel replay 可能等待其他工作，应进一步选择合适 replay 模式和范围，不能把单卡配置未经验证推广到多卡。

停下来回答：一个只有报告文件但没有采集到 kernel 的结果，算有效 NCU 证据吗？

## 第七阶段：PyTorch operator、shape 和内存归因

### 第 15 步：捕获一个 microbatch 的 forward/backward

```bash
python scripts/profile_run.py \
  --profile configs/monitoring/torch.json \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-torch
```

| 配置 | 默认值 | 原因 |
|---|---|---|
| `record_shapes` | true | 区分 microbatch 改变的 shape 与 operator 调用 |
| `profile_memory` | true | 保留框架分配事件；不等于驱动全部显存 |
| `with_stack` | false | 首轮降低 Python/C++ stack 记录体积 |
| `torch_first_microbatch` | true | 只从第一个 learner_forward 到该 microbatch backward 捕获，限制 trace |

输出 `torch-trace.json` 是 Chrome/Perfetto 可读 trace；`torch-operators.txt` 是 operator 汇总。使用 `scripts/analyze_torch_profile.py` 分析项目归档文件。

PyTorch allocated/reserved、驱动 memory.used 和 peak 是不同口径。框架 peak 包含所测进程的分配边界；驱动数字还可能包括上下文、非 PyTorch allocation 和其他进程。不能把两者差值未经证据解释成泄漏。

停下来回答：为什么三种显存口径都应保留？

## 第八阶段：设备与主机遥测

### 第 16 步：有限时间收集 nvidia-smi 与 `/proc`

```bash
python scripts/system_watch.py \
  --config configs/monitoring/system.json \
  --root-pid "$TRAIN_PID" --output results/system-watch
```

默认 30 秒；GPU 500 ms 一次，host 1 秒一次。输入进程 PID 应是要观察的实际进程树根。单独启动的 watcher 不会自动知道哪个训练 PID 属于你。

GPU 字段：timestamp、index、UUID、GPU/memory utilization、memory used/total、power、temperature、SM/memory clocks。CSV 表头给出 MiB、W、MHz、°C 或百分比。

`/proc/stat` 与 diskstats 是 host 累计 counters；进程记录包含 CPU ticks、RSS bytes、minor/major faults、IO bytes。ticks 转秒需除以记录的 `clock_ticks_per_second`，IO throughput 需计算相邻采样的 counter 差/实际时间差。进程结束、PID 重用、无权限和采样缺口必须保留。

短 kernel 无法通过低频 utilization 归因。观察到降频或大 IO 也需要和 nsys 时间范围对齐，不能仅凭同时出现就宣布根因。

### 第 17 步：可用 DCGM 时补充硬件 profiling fields

```bash
dcgmi discovery -l
dcgmi dmon -l
python scripts/system_watch.py \
  --config configs/monitoring/system.json --enable-dcgm \
  --root-pid "$TRAIN_PID" --output results/dcgm-watch
```

要求已有可访问的 DCGM hostengine，服务生命周期由环境管理。配置默认关闭，显式启用后若 binary 不存在或采集非零退出，runner 失败并保存原始输出。

| DCGM field ID | 含义 |
|---|---|
| 1001 | graphics/compute engine active |
| 1002 | SM active |
| 1003 | SM occupancy |
| 1004 | tensor pipe active |
| 1005 | DRAM active |
| 1009/1010 | PCIe TX/RX bytes |

`dcgmi dmon -i 0 -e <IDs> -d 1000 -c 30` 使用本地 DCGM GPU ID，不能假定容器的 CUDA ordinal 与 host ID 一致；先 discovery 核查 UUID。`N/A` 不等于零。profiling fields、ratio/byte 的单位以及支持情况以本机 field list 为准。

理由：nvidia-smi 只能给粗粒度 utilization，DCGM profiling fields 能补充 tensor/SM 等长期状态；NCU 用于短范围 kernel 精查。两者互补，但需要避免 counter 冲突。

### 第 18 步：perf stat 核查 CPU 行为

```bash
python scripts/profile_run.py \
  --profile configs/monitoring/perf.json \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-perf
```

包装 `perf stat -x , -o perf-stat.csv -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions -- <application>`。

`perf-stat.csv` 的 task-clock 通常是 ms，cycles/instructions 是计数；IPC= instructions/cycles。它覆盖整个应用进程生命周期，包含加载/warmup，不能和单 update 的 GPU range 混算。权限不足、event 不支持或 multiplexing 必须记录；百分比运行时间很低时，缩放 counters 不能被解释成无误差的精确计数。

理由：GPU API 等待可能来自 host 调度、页面缺失或 CPU 指令执行；perf 提供硬件/软件 counter 验证。`/proc` 已能回答的问题无需重复昂贵采样。

## 第九阶段：数值与执行认证

### 第 19 步：检查运行时合同

`configs/validity/runtime.json` 声明冷恢复、概率阈值、finite loss/gradient/parameters、Teacher 冻结、真实非零更新、rollout version 与 mask 边界。

`validity.json` 的目标窗口必须包含所有指定 step；default behavior error≤0.15；delta>0；loss/gradient/delta 有限。Teacher 冻结与 finite parameters 由 learner 原生门执行，失败会导致非零退出。Teacher 完整训练结束时另检查已声明的有限参数投影指纹；这不是每次 update 对完整 Teacher 权重逐字节校验。

`run_profiler_suite.py` 还比较各诊断进程最终 adapter/Adam/RNG 指纹，输出 `exact_final_state_equivalence`。该字段 false 时不能声称数值轨迹等价，即使各自的 runtime gate 都通过。数值近似相容需要另行定义容差和任务检验，不能改名为精确一致。

本轮 nsys/PyTorch 与 none 的 RNG、update、有效 token 数相同，精确指纹不同。[nsys 状态比较](reports/monitoring/PROFILE_NUMERICS_NSYS.json)和[PyTorch 状态比较](reports/monitoring/PROFILE_NUMERICS_TORCH.json)量化了全部 300 个 adapter 与 900 个 optimizer tensor：adapter 最大绝对差分别为 `3.73e-9` / `1.86e-9`，optimizer 为 `4.75e-9` / `5.76e-9`。两者通过检查差值前冻结的容差，但这仅支持诊断状态近似相容，不证明生成文本或业务质量等价。

```bash
python scripts/compare_profile_states.py \
  --reference results/profiler-suite/none/application/train/checkpoint-0162 \
  --candidate results/profiler-suite/nsys/application/train/checkpoint-0162 \
  --gates configs/validity/profile-numerics.json \
  --output results/nsys-numerics.json
```

输入必须是已核验的本地训练 checkpoint；`training.pt` 含 pickle 状态，不能加载不可信来源。adapter 容差为 `rtol=1e-5, atol=1e-6`，optimizer 为 `rtol=1e-4, atol=1e-8`；要求相同 key/shape/dtype、有限值、相同 param groups/step/token 和 RNG。输出逐 tensor 的最大绝对差、L2 差、参考范数、相对 L2、精确一致和容差判定。

```bash
python scripts/run_profiler_suite.py \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --checkpoint "$CHECKPOINT" \
  --tools-root "$TOOLS_ROOT" --output results/profiler-suite
```

每个 backend 在单独冷进程中执行。suite 保留失败 case，继续收集其他工具，并在末尾报告失败，不自动把 missing tool 填成 passed。

### 第 20 步：Compute Sanitizer 检查 CUDA 访问

```bash
python scripts/profile_run.py \
  --profile configs/monitoring/sanitizer.json \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-sanitizer
```

默认 `compute-sanitizer --tool memcheck --error-exitcode 86 --log-file sanitizer.log <application>`。自定义配置可以选择 racecheck、initcheck、synccheck；它们分别处理特定共享内存竞争、未初始化访问或同步问题，不能互相替代。

默认覆盖整个副本，包括加载和 warmup，microbatch=1、上限 1800 秒。大型训练的检查开销可能很大，优先用于最小可复现工作量和自定义 kernel。没有错误仅证明该次执行被该工具覆盖的检查通过；它不证明概率合同、业务答案、分布式逻辑或无性能退化。

## 第十阶段：服务级性能与有效吞吐

### 第 21 步：运行固定请求级 benchmark

先部署通过质量选择规则的模型，另起 benchmark 进程：

```bash
python scripts/serving_monitor.py \
  --config configs/monitoring/serving.json \
  --data /absolute/path/frozen-evaluation.jsonl \
  --output results/serving
```

该配置映射至 `python -m servedistill.http_benchmark`，分别执行并发 1/4/16；传入端点、模型名、每域数量、128 输出上限、120 秒 request timeout 和 900 秒 sweep timeout。`--dry-run` 展示真实 argv。

输入须有可评分 target；首次性能调试用 DEV/workload bank，正式 test 不能因调配置被反复观察。保留 `requests.jsonl` 的实际输入 ID、响应、分数、错误、latency、TTFT 和 first visible answer。thinking 已显式关闭；协议 fallback 另列，不能混为原生路径。

有效吞吐=通过预先评分规则的完成答案数/实际测量时间；P50/P95/P99 说明分位数样本量和 timeout 处理。`--gpus` 用于该次服务成本分母，应填写实际占用 GPU 数，不能把并发数当作卡数。

### 第 22 步：复查质量与成本门后才使用“优势”

冷恢复、finite 或 profiler trace 都不认证业务质量。独立确认仍需要所有配对种子、两个强 SFT、query 配对区间、各域非劣效、schema 和逐类 recall，以及不相交复验。新 anchor 轮是 DEV-only。

成本使用 `GPU·h = Σ(actual allocation seconds × allocated GPUs)/3600`。GPU power 遥测积分可作为有采样误差的设备能量估计，不是账单或数据中心总能耗。费用需要你明确提供单价。

历史 [expanded/report](reports/expanded/REPORT.md) 的 learner 2.98× 与完整训练 1.98× 基于已完成测量；常数 substitute 的失败也必须一起报告。本轮监控、DEV、后续确认和历史成本分别入账，不将提交成功算成验收成功。

## 配置与输出定位

| 文件/入口 | 直接运行方式 |
|---|---|
| `baseline/nsys/ncu/ncu_deep/torch/perf/sanitizer.json` | `scripts/profile_run.py --profile <file> ...` |
| `system.json` | `scripts/system_watch.py --config <file> ...` |
| `serving.json` | `scripts/serving_monitor.py --config <file> ...` |
| `tools.lock.json` | `scripts/install_profile_tools.py --lock <file> ...` |
| `ncu-compatibility-candidates.json` | `scripts/run_ncu_compatibility.py --candidates <file> ...` |
| `validity/runtime.json` | `scripts/profile_run.py --validity <file> ...` |
| `validity/profile-numerics.json` | `scripts/compare_profile_states.py --gates <file> ...` |
| `anchor-development-v1.json` | `scripts/run_anchor_development.py --protocol <file> ...`；须使用绑定输入和源码 |

默认配置都需要实际输入路径和对应 GPU/工具环境，不提供伪造 trace、占位测量值或自动通过的缺失字段。

公开采集导出使用 `scripts/export_profile_evidence.py --evidence <原始目录> --output <新目录>`，先核验每个 backend 的 `outputs.json`，再脱敏文本并压缩完整 PyTorch trace。`gzip -dc torch-trace.json.gz > torch-trace.json` 后可用 Perfetto 打开。原始 `.nsys-rep`、SQLite 和训练 tensor 保留在独立原始证据目录，公开 manifest 记录其字节数/哈希；文本脱敏后的哈希与原始文件不能混用。首次失败套件的实际 Pod 分配成本见 `reports/monitoring/PROFILE_PROFESSIONAL_V1_COST.json`，新兼容性任务成本另计。

## 官方技术参考

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：capture range、trace、统计与 GUI。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：NVTX filter、replay、metric、section 与 export/import。
- [GPU counter permission](https://developer.nvidia.com/ERR_NVGPUCTRPERM)：计数器权限错误。
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)：activities、shape、memory、stack。
- [DCGM user guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html)：field watches、profiling 与 hostengine。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：各检查器及 error-exitcode。
- [perf stat](https://man7.org/linux/man-pages/man1/perf-stat.1.html)：event、multiplexing、计数与输出格式。
