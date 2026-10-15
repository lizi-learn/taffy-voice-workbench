# Taffy Voice Workbench · 语音与情绪对照工作台

用同一文本、同一参考音频，比对可本地运行的 TTS。公开的是自编适配器、校验下载器、测试方法和结果；**不是音色权重、录音数据集或第三方引擎的重新授权**。

试听与资源：<https://塔菲.com/voice/>（非官方、非商业粉丝研究，所有试听明确标注 AI 合成）。

## 已实现

- GPT‑SoVITS v2ProPlus：明确选择 GPT / SoVITS 权重、准确主参考转写、多参考融合、temperature / top-p / top-k；作为当前音色优化主线。
- Bert‑VITS2：固定公开音色、受控噪声参数和逐句固定种子；检查全部活动参数，不静默随机初始化缺失层。
- IndexTTS‑2.5：8 维情绪向量、强度、语速，使用参考音频克隆。
- CosyVoice3‑0.5B‑2512：开心、伤心、生气等独立指令，参考音色与朗读文本分开传递。
- 相同文案与固定种子对照；记录原始 WAV 校验、RTF、峰值显存、内存、采样率、削波率。
- Index 声码器输出在进入上游 PCM 削波前保留峰值余量；保留未保护原版的对照记录，不靠事后调小音量掩盖已经产生的削波。
- 只读、无 GPU 的公共试听页；没有开放任意人物克隆接口。
- 单元测试不要求 GPU；依赖隔离，不修改其他项目的 Python 环境。

“已实现适配器”不代表所有情绪都听感准确。请阅读 `reports/` 中的实测状态；没有人工盲听验收，不宣称 MOS 或音色相似度排名。CER 是 ASR 文本错误率，不是好听程度。

## 本地使用

Python 3.10/3.11、CUDA GPU；各引擎请按其官方锁定环境分别安装。每个进程只加载一个引擎，不要混装 requirements，更不要升级生产推理环境里的 Torch。

1. 在独立目录检出 `engines.lock.json` 指定的上游版本，保留各自许可。
2. 下载模型：安装 `requests`，运行 `python tools/prepare_assets.py --out assets.lock.json` 生成新快照计划；如需复现实验，直接使用仓库已有计划，不重新生成。
3. `python tools/download_assets.py assets.lock.json --root /path/to/private/models`。镜像文件必须与固定官方 SHA‑256 一致；小配置使用不可变 revision。
4. 准备自己有权使用的参考录音，**不要上传他人录音或账号密钥到仓库**。

```bash
python cli.py --engine index \
  --source /path/to/index-tts --models /path/to/models/index \
  --reference /path/to/permitted-reference.wav \
  --text '这是一段人工智能合成的语音，今天也要开心呀。' \
  --mood happy --strength 0.8 --speed 1.0 --output /tmp/happy-AI.wav
```

CosyVoice 用 `--engine cosy`，`--strength` 保持 1。它没有经过校准的连续情绪强度接口，不能伪造滑块。Index 的 `gentle` 是低强度平静向量近似，不是训练时独立的“温柔”类别。

GPT‑SoVITS 需先按其上游方法备齐预训练目录（含 BERT、HuBERT、SV、G2PW、fastText 等）。`--models` 指向 `GPT_SoVITS/pretrained_models`；自己训练的 GPT / SoVITS 权重分别传入，不从目录里猜“最新”文件。

```bash
python cli.py --engine gsv --source /path/to/GPT-SoVITS \
  --models /path/to/GPT-SoVITS/GPT_SoVITS/pretrained_models \
  --gpt /path/to/trusted-gpt.ckpt --sovits /path/to/trusted-sovits.pth \
  --reference /path/to/permitted-reference.wav --reference-text '这条参考录音的准确文本。' \
  --aux-reference /path/to/permitted-reference-2.wav \
  --temperature 0.7 --top-p 0.9 --top-k 15 \
  --text '这是一段人工智能合成的语音。' --output /tmp/gsv-AI.wav

python cli.py --engine bert --source /path/to/pinned-taffy-bert-space \
  --bert-config /path/to/config.json --bert-weights /path/to/trusted-G_15800.pth \
  --noise-scale 0.35 --noise-scale-w 0.6 \
  --text '这是一段人工智能合成的语音。' --output /tmp/bert-AI.wav
```

两者只支持 `--mood neutral --strength 1`，否则明确报错；“低噪声”不冒充“温柔/开心”。Bert 不需要参考音频。可选 `--nltk-data /path/to/private-nltk-cache` 避免使用损坏的全局缓存。Bert 适配器针对锁定 Space 的 G_15800 架构，不承诺兼容任意版本 checkpoint。

```bash
python -m unittest discover -s tests -v
python cli.py --help
```

`tools/benchmark_emotions.py --help` 提供同文本批量对照。输出目录不可覆盖，断点重跑应新建目录；固定种子降低随机性，但不承诺跨 GPU、软件版本逐字节一致。

公共试听前端也在 `web/`：运行 `python -m http.server 8080 --directory web`，打开 `http://127.0.0.1:8080/voice/`。音频按需从公开静态站加载，不把音色权重、参考录音或 GPU API 放到浏览器里。A/B 偏好只存在 localStorage；网站不收集偏好或训练数据。没有线上自由输入文本合成，勿将预录试听按钮称为实时生成。

## 许可与边界

本仓库原创调度/适配器代码为 MIT。依赖与权重另计：

| 部件 | 上游许可 / 限制 |
| --- | --- |
| CosyVoice3 代码及本次主模型 | Apache‑2.0；外部依赖分别核对 |
| IndexTTS‑2.5 代码及主模型 | bilibili Model Use License Agreement，非 MIT |
| F5‑TTS 代码 / 基座权重 | MIT / CC‑BY‑NC‑4.0 |
| GPT‑SoVITS / 公共 Bert‑VITS2 音色 | 代码与音色权重许可分别核对，不在此仓库再分发 |
| 声音录音、人物音色、表情与 MMD | 不由本仓库 MIT 授权，版权及人格权益属于相应权利人 |

`assets.lock.json` 中记录辅助模型许可，包括可选 MaskGCT 的 CC‑BY‑NC‑4.0；IndexTTS‑2.5 本次走自己的 codec.pth，不依赖该可选语义 codec。请勿把公开可下载误解成获得商用授权。

本项目不代表塔菲本人发言，不提供身份认证、私信冒充、电话或交易用途。公开音频仅作 AI 研究演示。权利人可通过仓库 issue 申请更正或移除。参考录音与私有训练权重不随代码发布。

## 硬件选型

NVIDIA：先比较我们实测的 F5、GPT‑SoVITS、IndexTTS、CosyVoice；带情绪优先听后两者。Qwen3、VoxCPM2、Step‑Audio‑EditX、Fish S2‑Pro 等见研究清单，未跑的不填实测值。

RK3588：已在 16GB 实板跑通 Bert‑VITS2 的纯 CPU / ONNX 路线，见 [板端代码、实测与许可](edge/README.md) 和 [独立机器可读报告](reports/rk3588-20261015.json)。推荐先保留 FP32 音色条件，再优化局部声码器输出；低内存才考虑只量化 BERT。GPT‑SoVITS 板端、RKNN/NPU 尚未验收，不把导出成功当作部署成功。旧 GPU 报告中的 `rk3588_measured:false` 仅指那一轮，板端结果单独列出。

新增板端导出器包含 AGPL-3.0-only 改编代码，本目录原有 MIT LICENSE 不覆盖这些文件；组合运行需遵守对应 copyleft 义务，详见 `edge/README.md`。不再笼统把整套运行环境称为 MIT。

## 本轮实测摘要（2026-10-15）

累计 **5 种引擎、21 套配置、201 条对照生成**，其中网页精选 17 套配置、159 条音频。本轮新增 11 组、88 条音色参数对照，另有 2 条接口冒烟生成不计入对照。22 条官方模型与部署路线分别标记实测/资料核验/暂缓。

| 方案 | 基础 8 句 CER ↓ | 情绪组 CER ↓ | 热推理 RTF ↓ |
| --- | --- | --- | --- |
| F5 自训精修 | 5.08% | 无独立控制 | 0.083 |
| GPT‑SoVITS 混合 | 17.51% | 无独立控制 | 0.109 |
| CosyVoice3 指令 | 5.65% | 4.02% | 0.218 |
| IndexTTS‑2.5 保护版 | 3.39% | 8.62% | 0.226 |

这些是 RTX 4090 48GiB 上 batch 1 的短句结果，不是主观音色排名。参考响度/采样处理有差异，公共音色训练集未知，详见 `reports/all-configurations.json`。原版 Index 出现微量削波；峰值保护版 21 条重跑均为 0。语速 0.8/1/1.25× 对同一句话分别输出约 7.49/5.99/4.78 秒。

当前**音色优先于情绪**：根据站长试听反馈，GPT‑SoVITS 为主，Bert‑VITS2 重点备选。CosyVoice3 音色不够符合偏好，降为情绪实验；这不是公开 MOS 排名。原版全部保留，不用 ASR 排名推翻听感。

调优覆盖日常主参考、三参考融合、低温采样、GPT e2/e4、SoVITS e2/e4，以及 Bert 两种噪声参数。固定 8 句、逐句种子 20261015，不后期变调。**本轮未证明 GPT 候选优于原混合版，因此网站默认仍用原版**。Bert 低时长噪声版 CER 为 26.55%，匹配种子对照 28.81%；仅 177 个规范化字符的短句结果，不宣称声音更像。所有候选供同题 A/B，听感等待站长确认。

Index 的强情绪和控制组有更多 ASR 错字，不能仅凭基础 CER 更低就宣布全面胜出。原始 WAV 保留，统一线性响度试听不改变语速、音高。没有人工盲听验收或情绪识别达标的承诺。
