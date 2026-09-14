# Merlin / Arnold 训练适配

这里的 `mlx` 是 Merlin 任务与 GPU 管理工具。训练仍使用 PyTorch、Transformers Trainer 和可选 DeepSpeed。

## 统一训练入口

让 Merlin worker 或 Arnold 任务在每个节点运行一次 `bash script/train.sh`。脚本通过 torchrun 在调度器分配的可见 GPU 上启动进程，不再硬编码 GPU `0,3` 或个人服务器路径。

日常配置集中在 `script/train.sh` 顶部的“配置区”。修改一次后无需在终端逐项 export。例如把三条路径配置改成：

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/models/your-model}"
DATA_PATH="${DATA_PATH:-/mnt/data/sft_train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/output/debug-run}"
TRAIN_MODE="${TRAIN_MODE:-debug}"
```

然后在 **worker/job 内部**直接运行：

```bash
bash script/train.sh
```

也可以保持 `TRAIN_MODE=train`，临时用 `bash script/train.sh --debug` 调试。正式训练时把配置区的 `TRAIN_MODE` 改回 `train`，并切换到正式输出目录。GPU 数量、batch、学习率、epoch、日志、DeepSpeed、调试步数等运行参数均在同一配置区；`EXTRA_TRAIN_ARGS` 可添加其他 Trainer 参数。已有环境变量覆盖方式保持兼容。

调试默认训练 5 个 optimizer step，禁用 checkpoint 和最终模型/state 导出，跳过旧 checkpoint 自动恢复。日志由 Merlin/Arnold 收集，失败退出码向上传递。

**注意：此脚本目前仍是资源分配后的训练入口，不会在开发机上自动申请 worker。** 用户提供的 [MLX 使用文档](https://bytedance.larkoffice.com/wiki/IjyLwRw6PidWpSkxHGwcz7hpnyd) 在当前环境跳转到登录页，尚未核实文档里的 sh 资源配置格式和 worker launch 示例。自动调度的脚本封装需待这些字段确认后接入。

常用配置项（也支持环境变量覆盖）：

| 参数 | 默认值 / 用途 |
| --- | --- |
| `MODEL_PATH` / `DATA_PATH` / `OUTPUT_DIR` | 必填；模型、数据、持久化输出 |
| `MICRO_BATCH_SIZE` | 1，每 GPU batch |
| `GRADIENT_ACCUMULATION_STEPS` | 1 |
| `MAX_LENGTH` | 4096 |
| `EPOCHS` | 2 |
| `LEARNING_RATE` | 2e-5 |
| `BF16` | True，取决于所分配 GPU 是否支持 |
| `NPROC_PER_NODE` | gpu，自动使用可见 GPU 数量 |
| `PYTHON_BIN` | python3，使用任务镜像内的解释器 |
| `DEEPSPEED_CONFIG` | 未设置则用普通 DDP；可指定 `configs/ds_config_zero3.json` |

其他 Trainer 参数可以追加到脚本后，例如 `--seed 42`。调试模式的 max_steps/save 参数优先，避免被追加参数改成正式训练。

单节点默认 `--standalone`。多节点时，由任务启动配置显式传入 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`，不要将 GPU 进程总数 WORLD_SIZE 当成节点数。只有当平台已经按每 GPU 启动一个进程并提供 rank 环境时才设置 `LAUNCH_MODE=process`，避免重复启动。脚本不猜测 Arnold 特定环境变量映射，需根据所用任务模板配置。

## worker 调试与正式提交

1. 修改代码后，用开发机上已有的 `mlx worker launch` 配置启动调试，训练命令指定为上述 `bash script/train.sh --debug`，配置模型、数据挂载和所需 GPU。
2. 确认日志正常完成指定 step、loss 有限、退出码为 0，再提交代码；不要把数据、模型、日志或输出权重加入 git。
3. 正式任务 YAML 使用相同镜像、数据和模型挂载，启动命令改为 `bash script/train.sh`，输出指向正式持久化目录。
4. 在代码已提交且远端仓库可取得该 commit 后提交 Arnold：

```bash
mlx job submit --yaml /path/to/existing-job.yaml --commit "$(git rev-parse HEAD)"
```

`--yaml` 和 `--commit` 已核对本机 CLI 帮助。仓库尚无 Merlin/Arnold YAML，资源队列、镜像、挂载及 worker launch 参数应复用你的实际任务模板；不要将这里的占位路径直接作为提交配置。当前执行会话无法通过 Merlin 开发机上下文检查，因此不能在这里声称已完成 worker GPU 调试或 Arnold 提交。

## 训练代码修复范围

修复了 dataclass 字段误写成 tuple、未初始化进程组就调用 rank/barrier、各 rank 的数据顺序可能不一致、shuffle 返回值丢失、正式输出未保存 tokenizer，以及把普通模型错误标记为 model_parallel 的问题。默认保持原检索任务的特殊 token 注册和 SFT 数据格式（`instruction` / `output`，支持原 JSON、JSONL、Excel）。

当前入口明确支持 SFT 全参数训练。旧 LoRA 分支大部分实现被注释，旧 DPO 分支也不完整，现改为提前报错，避免误以为已使用这些算法。此改动未迁移它们。正式训练默认自动恢复输出目录中的最新 checkpoint；传入 `--overwrite_output_dir True` 可从头开始。正式保存使用 Trainer 的 checkpoint/模型保存流程，同时保存 tokenizer。使用 `--do_eval True --eval_path /path/to/valid.json --eval_strategy steps --eval_steps 100` 可启用验证集。

## 环境与本地检查

复用与你的 GPU/CUDA 匹配的任务镜像。已预装匹配的 PyTorch 后，可用 `python -m pip install -r requirements-training.txt` 安装训练依赖；原依赖以 Transformers 4.51.3、Accelerate 1.6.0、Datasets 3.5.1 为基线，训练代码还导入 PEFT（可使用 0.15.2）、pandas 和 tqdm。启用 DeepSpeed 时镜像需要安装兼容其 Torch/CUDA 的 DeepSpeed；默认 DDP 无需安装 DeepSpeed。不应安装 Apple 的 mlx-lm。

```bash
bash -n script/train.sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

测试含无需 GPU 的启动参数检查，以及有训练依赖时运行的离线微型 Llama CPU smoke test（调试步数、禁止保存、正式模型与 tokenizer 导出）。CPU 测试不代表已经验证 NCCL、DeepSpeed 或集群 GPU。
