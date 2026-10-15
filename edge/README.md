# RK3588：保留音色，而不只是把模型缩小

实板：RK3588 / 16GB / Ubuntu 22.04 / ARM64。仅 CPU ONNX Runtime 1.23.2，**没有使用 NPU、GPU 或联网合成**。不包含可公开调用的接口。

## 当前结论

- 内存够：先用 **FP32 BERT + FP32 声学生成器**。同外部随机噪声下，板端与 x86 FP32 的 8 句输出长度一致，PCM RMSE 为 1.88e-6～4.89e-6。这是部署数值保真，不是证明原模型像本人。
- 内存紧：只量化 BERT 的常量 MatMul 权重，保留声学部分 FP32。128 帧分块峰值 RSS 约 1071MiB，但会改变特征、时长及波形，**不能称为无损**；等待主观试听验收。
- 不要删除 BERT 来省内存。消融组虽然 ASR CER 更低，却改变了训练时的条件输入和多数句子的时长；“字读准”不是“音色还原”。
- 固定播报：`phrase_cache.py` 可缓存已生成的 GPT‑SoVITS 原始 WAV，逐字、逐版本、逐文件哈希匹配，不改一个音频字节。**这是播放缓存，不是板端任意文本 TTS**。
- GPT‑SoVITS 的 ARM/ONNX 模型已完成初步导出，但未完成板端运行与一致性验收，不能和已经跑通的 Bert 路线混写成“都支持 RK3588”。RKNN/NPU 仍未实测。

## 为什么这样优化

计时发现瓶颈主要在声学生成器，BERT 量化只省了约十分之一秒量级的单句计算，不能指望它让整句快数倍。完整 flow 含全局上下文，不能任意切碎；只切局部卷积声码器，左右各保留 16 帧，输出后裁去重叠区。

4 / 8 帧上下文仍有边界误差；16 帧在两种长度、64 帧分块的导出验证中达到约 1e-8 的浮点 RMSE。实板 128 帧分块的 3 条匹配文案与整句输出相比，PCM 最大差异不超过 1 个 16-bit LSB。没有靠变速、变调或后处理掩盖误差。

| 同一组三句，重复两次后取第二轮中位数；4 个 A76 核 | RTF ↓ | 首个 PCM 数组（秒）↓ | 进程峰值 RSS |
| --- | --- | --- | --- |
| FP32 整句 | 0.737 | 5.069 | 2664MiB |
| FP32，64 帧 / 重叠 16 | 1.049 | 1.385 | 1928MiB |
| FP32，128 帧 / 重叠 16 | 0.897 | 1.826 | 2102MiB |
| FP32，256 帧 / 重叠 16 | 0.816 | 2.830 | 2002MiB |
| 仅 BERT INT8，128 帧 / 重叠 16 | 0.881 | 1.716 | 1070MiB |

RTF=生成耗时/音频时长，低于 1 表示本次生成快于播放。首个 PCM 不是扬声器首声、网络首包或冷启动。64 帧虽然更早出首段，但重叠重复计算使总 RTF 超过 1，不适合据此宣称持续流畅。128 帧是当前短句折中，长文本、并发、多小时负载还未验收。RSS 是整个运行的峰值，受 ORT arena 和请求形状影响；不能用不同文案的占用作严格量化收益。

## 导出与离线运行

首先自己合法取得上游固定版本和权重，不从本仓库下载人物音色权重。导出只在独立环境运行，不升级其他项目依赖。

```bash
python edge/export_bert.py --source /path/to/pinned-Bert-Space --out /path/to/private-onnx
python edge/export_bert_stream.py --source /path/to/pinned-Bert-Space --models /path/to/private-onnx
```

板端安装 Python、numpy==1.26.4、onnxruntime==1.23.2、tokenizers==0.21.4、pypinyin==0.55.0、jieba==0.42.1、cn2an==0.5.24、soundfile==0.13.1。模型包包含哈希清单、固定 tokenizer、中文 frontend 和其许可；不要省掉许可文件。

```bash
taskset -c 4-7 python edge/runtime_bert.py \
  --models /path/to/private-onnx --context fp32 --threads 4 \
  --prompts /path/to/chinese-prompts.json --ids welcome,story,numbers \
  --stream-chunk 128 --repeats 2 --out /path/to/new-results
```

`prompts.json` 是包含 `id`、`text` 的对象数组。默认拒绝中文 frontend 会静默丢弃的拉丁字母；含品牌、型号、英文缩写时需先显式转成你希望读出的中文。`--allow-latin-loss` 仅用于复现本次暴露该缺陷的研究消融，绝不能作为产品默认值。长度上限、哈希验证及拒绝覆盖既有输出均保留。

分块方法通过 `BertCPU.stream_chunks()` 迭代返回 NumPy PCM；CLI 为便于复核最终拼成 WAV，不是一个低延迟音频网络服务。不要把 CLI 完成后的播放叫端到端流式。

## 固定话术缓存

```bash
python edge/phrase_cache.py put --root /private/cache --profile gsv-reference-v1 \
  --text '需要缓存的准确文本' --source /private/permitted-AI.wav
python edge/phrase_cache.py get --root /private/cache --profile gsv-reference-v1 \
  --text '需要缓存的准确文本'
```

改权重、参考、种子、参数时必须改 profile 版本；未命中直接报错，不拿近似文案冒充命中。合成声音需明确标识，不用于身份验证或冒充。

## 公平比较的边界

GPT‑SoVITS 用了专门的 SoVITS 训练，CosyVoice 是通用模型参考克隆；Bert 公共音色的训练集未知。因此用户认为前两者更像，不足以推出“某架构一定胜过另一架构”。专门训练、参考风格、数据带宽和推理模式都有影响。

CosyVoice 官方 `frontend_instruct2` 会移除 LLM 参考语音 token；`zero_shot` 则保留参考语音 token，并提供准确转写。新增 `tools/benchmark_cosy_modes.py` 用同权重、同参考、同逐句种子分别验证两种模式。模式差异是一个可验证的混杂因素，**不是已经证明音色好坏的原因**。

CPU/ONNX 比较使用外置 NumPy PCG64 噪声，旧 PyTorch 音色组使用 Torch 随机数。只有本轮相同外部噪声的结果可作后端数值一致性对照，不跨组宣称逐波形相同。ASR CER 和波形相关性都不能代替音色盲听。

## 许可

`export_bert.py`、`export_bert_stream.py` 含对上游推理结构的改编，按 **AGPL-3.0-only** 发布（见本目录 LICENSE）。原创运行编排、缓存和测试文件标注 MIT；与 AGPL 中文 frontend / 模型代码组合分发或提供网络服务时仍需遵守 AGPL 的适用义务，不能把整个组合说成纯 MIT。上游 `tone_sandhi.py` 另保留 Apache-2.0 通知。模型权重、人物音色和参考录音未被本仓库重新授权，也不在公开仓库分发。

来源：

- [固定 Bert‑VITS2 Space](https://huggingface.co/spaces/songdaooi/Taffy-Bert-VITS2/tree/8d3a82bc3a94242cf91cee4f32902f13357653e9)
- [CosyVoice 官方调用示例](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/example.py)
- [CosyVoice 官方 frontend](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/cli/frontend.py)
- [GPT‑SoVITS 社区 ONNX 导出路线](https://github.com/bailu-obj/gpt-sovits-onnx-rs/tree/204ae4abcd218113aad86f6f9a66bdba92adcfd7)
- [RKNN Toolkit2 动态形状示例](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/dynamic_shape)
