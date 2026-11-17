# colab-use：用 Colab 跑 MiniMax-H3-Image 生图

全程在终端里完成，不用打开 notebook。底层用的是谷歌官方的 [Colab CLI](https://github.com/googlecolab/google-colab-cli)（`colab` 命令，2026 年 6 月发布），本仓库的 `./h3` 在它外面包了一层，负责开机、装环境、生成、下载和关机。

```bash
uv tool install google-colab-cli      # 一次性：安装官方 Colab CLI
colab usage                           # 一次性：浏览器授权 Google 账号，顺便看 CU 余额

./h3 up                               # 开 A100 80GB → 装 ComfyUI → 下载 ~49GB 权重 → 启动（约 10 分钟）
./h3 gen "A detailed English prompt ..." -n 2         # 文生图，PNG 下载到 ./outputs/
./h3 gen -f prompts.txt --mp 4                        # 批量，用 4MP 先试提示词
./h3 edit "Keep <Picture 1> ..., but ..." photo.png   # 改图 / 多图参考
./h3 down                             # 关机，停止扣 CU（用完一定要执行）
```

## 结论

- **生图要用的是 `MiniMax-H3-Image`**（2026-11-10 开源）。`MiniMax-H3` 本体是**视频+音频**模型，不出静态图。
- **推荐的组合是 ComfyUI ≥ 0.45 加 [Comfy-Org 重新打包的权重](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)**，DiT 和文本编码器都用 int8-ConvRot，一共约 49 GB。官方 diffusers 仓库需要下载约 135 GB，在 Colab 上每次开机都得重新下载，所以不划算。
- **GPU 首选 A100 80GB**（约 7 CU/小时）。Pro+ 用户用 G4（RTX PRO 6000 96GB）更快。L4 和 T4 不推荐。
- **省 CU 的办法**：Colab 按 VM 开着的时长计费，不管 GPU 有没有在算。所以要攒一批提示词一次跑完，再立刻 `./h3 down`。批量任务可以加 `--down`，跑完自动关机。
- ⚠️ **许可证不覆盖美国、欧盟、英国和韩国**，而 Colab 的 VM 通常分配在美国。详见下文「许可证」一节。

## H3 和 H3-Image

| | MiniMax-H3 | MiniMax-H3-Image |
|---|---|---|
| 输出 | 4–15 秒视频，带立体声音频，768p（2K 超分模块未开源） | 静态图，原生约 16MP（4096²），文生图与编辑共用一个 checkpoint |
| DiT | 33B（FL2VA / Ref2VA 两个变体） | 33.1B，去掉了音频层 |
| 文本编码器 | Qwen3-VL-32B，取第 50 层 hidden states | 同一个 |
| 采样 | — | 6 步，蒸馏掉了 CFG（不支持 negative prompt，也不能改步数） |
| 编辑 | — | 最多 16 张参考图，提示词里用 `<Picture 1>`、`<Picture 2>`… 指代 |

## 方案对比

| 方案 | 每次开机下载量 | 单卡显存 | 在 Colab 上的问题 |
|---|---|---|---|
| **ComfyUI + Comfy-Org int8（本仓库采用）** | **~49 GB** | 80 GB 卡上能全部常驻 | 无明显问题；还附带网页界面和官方模板 |
| ComfyUI + Comfy-Org bf16 DiT | ~68 GB | 需要 ≥70 GB | 与原版数值一致，但下载更大，速度未必更快 |
| 官方 diffusers（`ModularPipeline`） | ~135 GB | 峰值约 65 GB，需要 auto CPU offload | 下载和加载都慢，系统内存也吃紧 |
| vLLM-Omni / SGLang | ~135 GB | 单卡需要 DLO 逐层 offload | 主要面向多卡服务器；要装 vLLM 0.33 + cu130，装起来麻烦 |
| DiffSynth-Studio 低显存 | — | 24 GB（4090）也能跑 | 针对小显存卡；Colab 上没必要用 |

Comfy-Org 打包的权重为什么这么小：

- **pruned**：H3 的 DiT 里约 13B 参数在 AdaLN 分支上，而 6 步采样的时间步是固定的，这部分可以提前算好再删掉。DiT 因此从 66 GB 降到 40 GB（bf16），不损失精度。
- **int8-ConvRot**：一种基于旋转的 int8 量化，再把 DiT 压到 21 GB。DiffSynth 的文档称它“几乎无损”。
- **文本编码器**：截断到模型实际用到的第 50 层，bf16 是 51.5 GB，int8 是 27 GB。

## GPU 选择与花费（粗估）

| 运行时 | 显存 | 约 CU/小时 | 说明 |
|---|---|---|---|
| A100 80GB（High-RAM） | 80 GB | ~7 | **首选**。int8 权重（约 46 GiB）可以全部留在显存里 |
| G4 RTX PRO 6000 | 96 GB | ~9 | 仅 Pro+。Blackwell 架构，比 A100 快；bf16 DiT 也放得下 |
| H100 | 80 GB | 5～18（社区实测差异大） | 仅 Pro+。最快，先在资源面板确认费率 |
| A100 40GB | 40 GB | ~5.3 | 能跑，但文本编码器要在显存和内存之间来回换，较慢 |
| L4 / T4 | 24 / 16 GB | — | 不推荐（T4 不支持 bf16） |

CU 费率来自社区测量，Google 不公布，以 Colab 资源面板显示为准。

**每张图的耗时**：根据 vLLM-Omni 官方给出的 H200 数据推算，A100 上 4MP 大约十几秒，16MP 大约一分钟上下，H100 和 G4 更快。这些都是推算值，以 `./h3 gen` 打印的实际耗时为准。第一张图要先把权重读进显存，会额外多 1～2 分钟。

**固定开销**：每次开机安装加下载约 10 分钟，在 A100 上约 1～1.5 CU。所以 Colab 适合攒一批提示词一次跑完。按 Pro 的价格（$9.99/100 CU）算，A100 上一张 16MP 图大约 0.1 CU，约 $0.01。作为对比，MiniMax 官方 API 的 4K 图价格约 $0.069 一张。

## 用法

### 准备（只做一次）

1. 安装官方 Colab CLI：`uv tool install google-colab-cli`（或 `pip install google-colab-cli`）。它只支持 macOS 和 Linux。
2. 运行任意 `colab` 命令（比如 `colab usage`），按提示在浏览器里授权 Google 账号。消耗的是这个账号的 Colab 计算单元（CU）。
3. 如果要用 `./h3 ui` 或 `./h3 shell`，需要 `~/.ssh/id_ed25519`（Colab 不接受 RSA 密钥）。没有的话运行 `ssh-keygen -t ed25519` 生成。

### 命令

| 命令 | 作用 |
|---|---|
| `./h3 up [--gpu A100\|G4\|H100] [--dit int8\|bf16] [--enhancer]` | 开机（默认 A100 加 `--high-mem`）、装 ComfyUI、下载权重、启动服务。可以重复执行，已有的步骤会跳过。会话已存在时直接复用，不会再开一台 |
| `./h3 gen "prompt" [-n 2] [--aspect 16:9] [--mp 16] [--seed 42] [--enhance] [--down]` | 文生图。`-f prompts.txt` 为批量模式，每行一条，`#` 开头的行忽略 |
| `./h3 edit "prompt" a.png [b.png ...]` | 图像编辑或多图参考（最多 16 张），提示词里用 `<Picture 1>`、`<Picture 2>`… 指代 |
| `./h3 ui` | 通过 SSH 端口转发，在本机浏览器打开 `http://127.0.0.1:8188` 使用 ComfyUI 网页界面，只有你自己能访问 |
| `./h3 status` | 会话信息、GPU 显存和利用率、ComfyUI 队列、CU 余额和费率 |
| `./h3 logs [-n 200]` | ComfyUI 日志末尾（排查 OOM 或报错用） |
| `./h3 restart` | 重启 ComfyUI，清空显存和队列 |
| `./h3 shell` | SSH 登录到 VM |
| `./h3 down` | 关机，停止计费 |

- **输出**：每张图存成 `outputs/t2i_<时间>_<序号>_s<seed>.png`，旁边有同名 `.json`，记录提示词、seed、尺寸和耗时。
- **多个会话**：`./h3 -s 名字 up` 或者设环境变量 `H3_SESSION`。
- **画幅和分辨率**：`--aspect` 可选 1:1、4:3、3:4、3:2、2:3、16:9、9:16、21:9。`--mp 16` 对应官方推荐尺寸（如 4096×4096）。有效范围约 4～20MP，4MP 大约快 4 倍。

### 提示词

模型最适合**详细的英文描述**：主体、构图、光线、材质，以及画面里要出现的文字（用引号括起来）。它不支持 negative prompt，不想要的东西直接写进正向提示词。

习惯写中文短句的话，用 `./h3 up --enhancer` 额外下载 7.3 GB 的 Qwen3.5-9B，之后 `./h3 gen "一只在雪地里的橘猫" --enhance --aspect auto` 会先把中文短句扩写成英文长描述，并由它推荐画幅。

### 原理

`./h3` 只在本机调用 `colab` 命令：

- `colab new` 开机。
- `colab exec` 把 `remote/setup.py` 和 `remote/generate.py` 发到 VM 的内核里执行。
- `colab upload` / `colab download` 传参考图和结果。
- `colab ssh --proxy-mode` 用来做端口转发。

ComfyUI 作为后台进程跑在 VM 上，`remote/h3_comfy.py` 通过 `127.0.0.1:8188` 的 API 提交工作流。工作流参数和官方模板一致：euler / simple / 6 步 / cfg 1。

## 许可证

MiniMax-H3-Image Community License（[原文](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)）的要点：

- 授权地域为全球，**但排除欧盟、英国、韩国和美国**；在排除地域使用模型或其输出不在授权范围内。排除地域的用户可以联系 MiniMax 单独申请授权。
- 年收入超过 100 万美元，或者对第三方提供托管服务（API、网页等），需要事先申请书面授权。

Colab 不能选择机房，VM 通常分配在美国。`./h3 up` 会通过 ipinfo.io 显示 VM 所在地（用 `--no-location-check` 关掉）。运行前请自己确认用法是否合规。

## 文件说明

- `h3`：本机命令行入口，Python 3 标准库，不需要装任何依赖。
- `remote/setup.py`：在 VM 上执行的初始化脚本，负责检查硬件和所在地、安装 ComfyUI、并行下载权重、启动服务，最后写出 `/content/h3_config.json`。
- `remote/generate.py`：在 VM 上执行的生成任务，每完成一张输出一行 `H3_RESULT`，本机据此下载。
- `remote/h3_comfy.py`：只用标准库的 ComfyUI API 客户端，负责拼工作流、提交任务、轮询结果、上传参考图。
- `tools/validate_graphs.py`：用本地（CPU）ComfyUI 的校验逻辑检查工作流，确认节点名、参数名和取值都合法，不需要 GPU，也不需要真实权重。

## 参考

- 模型卡：[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)、[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- ComfyUI 打包权重：[Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)；ComfyUI 教程：[docs.comfy.org](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image)
- diffusers 文档：[MiniMax-H3-Image pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image)
- vLLM-Omni 部署配方（含 H200 实测延迟）：[recipes.vllm.ai](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image)
- DiffSynth-Studio 低显存方案：[文档](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/MiniMax-H3-Image.html)
- 官方 Colab CLI：[googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
- Colab GPU 与 CU 费率（社区整理）：[fastero.com](https://fastero.com/blog/google-colab-gpu)
