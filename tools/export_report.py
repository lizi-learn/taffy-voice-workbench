"""Sanitize local measurements for publication; never export private inventory/data."""
import argparse
import json
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--timbre',type=Path)
    args=p.parse_args()
    out=Path(__file__).resolve().parents[1]/'reports'
    out.mkdir(exist_ok=True)
    result=[]
    inputs=[(n,args.previous/(n+'-metrics.json')) for n in ['f5-refined','gsv-base','gsv-trained','public-baseline','gsv-base-prepared-ref','gsv-trained-prepared-ref','gsv-sovits-only-prepared-ref']]+[
        ('cosy-instruct',args.new/'samples-cosy/cosy-metrics.json'),
        ('index-upstream-pcm',args.new/'samples-index/index-metrics.json'),
        ('index-peak-safe',args.new/'samples-index-safe/index-metrics.json')]
    if args.timbre:
        inputs.extend((p.parent.name,p) for p in sorted(args.timbre.glob('*/*-metrics.json')))
    for label,path in inputs:
        m=json.loads(path.read_text())
        rows=m['samples']
        baseline=[s for s in rows if s.get('group','baseline')=='baseline']
        def cer(group):
            data=[s for s in rows if s.get('group','baseline')==group]
            if not data:return None
            return sum(s['writing_normalized_edits'] for s in data)/sum(s['writing_normalized_characters'] for s in data)
        result.append({'configuration':label,'utterances':len(rows),'baseline_cer':cer('baseline'),
            'parameters':m.get('configuration',{}),
            'emotion_cer':cer('emotion'),'showcase_cer':cer('showcase'), 'control_cer':cer('control'),
            'baseline_warm_median_rtf':statistics.median(s['rtf'] for s in baseline[1:]),
            'max_clipped_fraction':max(s['clipped_fraction'] for s in rows),
            'samples':[{k:s[k] for k in ['id','prompt','group','text','file','mood','strength','speed','seconds','compute_seconds','rtf','sample_rate','peak','clipped_fraction','sha256','asr_text','writing_normalized_cer','acoustics'] if k in s} for s in rows]})
    report={'date':'2026-10-15','synthetic':True,'model_families':5,'configurations':len(result),
        'utterances':sum(x['utterances'] for x in result),'human_listening_accepted':False,
        'emotion_recognition_validated':False,'rk3588_measured':False,
        'method':'RTX 4090 48 GiB. Batch 1, 4 CPU threads. Original baselines retained. Timbre round varies only documented train-only references/sampling/checkpoint stages, not a strict architecture or timing ranking. Public Bert training set unknown; loader matches source Space live-key rules, with no missing live tensors. New Bert resets seed per prompt and has a matched control. Same Whisper large-v3-turbo; OpenCC t2s+cn2an+NFKC. CER is not a timbre/MOS score. Equal-loudness listening copies; raw WAV hashes retained.',
        'results':result}
    (out/'all-configurations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['Taffy Voice Workbench — 实测摘要（2026-10-15）','所有音频为 AI 合成。非本人录音、非官方、非商业。',
        f'共 {report["model_families"]} 种引擎、{report["configurations"]} 套配置、{report["utterances"]} 条对照生成。网页精选 17 套配置、159 条（含本轮 88 条调优音频）。',
        '','配置 | 基础 CER | 情绪组 CER | 基础热 RTF | 最大原始削波率']
    pct=lambda n:'未测' if n is None else f'{n*100:.2f}%'
    for r in result:lines.append(f'{r["configuration"]} | {pct(r["baseline_cer"])} | {pct(r["emotion_cer"])} | {r["baseline_warm_median_rtf"]:.3f} | {r["max_clipped_fraction"]*100:.5f}%')
    lines.extend(['','选型：遵循站长音色反馈，GPT-SoVITS 为主、Bert-VITS2 重点备选。CosyVoice3 仅作情绪控制实验，不作主音色；CER 不能替代音色听感。',
        '本轮 11 组音色调优没有证明 GPT 候选比原混合版更好，默认保留原版，候选均待试听。Bert 低时长噪声版 CER 26.55%，匹配种子对照 28.81%；仅 8 句，不据此宣布音色胜出。',
        'Index 原始版有轻微削波；发布适配器在进入上游 PCM clamp 前增加线性余量保护，21 条重跑均无削波。没有修改模型权重。',
        '语速实测（同一条 Index 开心文案）：0.8× 7.488 秒，1× 5.991 秒，1.25× 4.783 秒。',
        'CosyVoice3 的开心/温柔等条件改变音高、时长，不代表已通过情绪识别或主观盲听。未发现本轮控制指令被 ASR 转写为正文。',
        '基础组只有 8 句；参考处理并不完全一致；公共模型训练集未知。不是严格架构排名，不是商业权利证明。',
        'RK3588 尚未实测；不能拿 x86 或作者宣传速度冒充实板结果。','',report['method']])
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'model_families':5,'configurations':report['configurations'],'utterances':report['utterances']}))

if __name__=='__main__':main()
