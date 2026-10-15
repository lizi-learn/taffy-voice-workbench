"""Sanitize local measurements for publication; never export private inventory/data."""
import argparse
import json
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    args=p.parse_args()
    out=Path(__file__).resolve().parents[1]/'reports'
    out.mkdir(exist_ok=True)
    result=[]
    for label,path in [(n,args.previous/(n+'-metrics.json')) for n in ['f5-refined','gsv-base','gsv-trained','public-baseline','gsv-base-prepared-ref','gsv-trained-prepared-ref','gsv-sovits-only-prepared-ref']]+[
        ('cosy-instruct',args.new/'samples-cosy/cosy-metrics.json'),
        ('index-upstream-pcm',args.new/'samples-index/index-metrics.json'),
        ('index-peak-safe',args.new/'samples-index-safe/index-metrics.json')]:
        m=json.loads(path.read_text())
        rows=m['samples']
        baseline=[s for s in rows if s.get('group','baseline')=='baseline']
        def cer(group):
            data=[s for s in rows if s.get('group','baseline')==group]
            if not data:return None
            return sum(s['writing_normalized_edits'] for s in data)/sum(s['writing_normalized_characters'] for s in data)
        result.append({'configuration':label,'utterances':len(rows),'baseline_cer':cer('baseline'),
            'emotion_cer':cer('emotion'),'showcase_cer':cer('showcase'), 'control_cer':cer('control'),
            'baseline_warm_median_rtf':statistics.median(s['rtf'] for s in baseline[1:]),
            'max_clipped_fraction':max(s['clipped_fraction'] for s in rows),
            'samples':[{k:s[k] for k in ['id','prompt','group','text','file','mood','strength','speed','seconds','compute_seconds','rtf','sample_rate','peak','clipped_fraction','sha256','asr_text','writing_normalized_cer','acoustics'] if k in s} for s in rows]})
    report={'date':'2026-10-15','synthetic':True,'model_families':5,'configurations':len(result),
        'utterances':sum(x['utterances'] for x in result),'human_listening_accepted':False,
        'emotion_recognition_validated':False,'rk3588_measured':False,
        'method':'RTX 4090 48 GiB. Batch 1, 4 CPU threads. Shared train-only reference contents; normalized 32 kHz reference for GSV proper/new models, original gain for F5: a confound. Public Bert training set unknown and legacy condition-layer mismatch. Same Whisper large-v3-turbo; OpenCC t2s+cn2an+NFKC. Do not call CER a timbre/MOS score. Equal-loudness listening copies; raw WAV hashes retained.',
        'results':result}
    (out/'all-configurations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['Taffy Voice Workbench — 实测摘要（2026-10-15）','所有音频为 AI 合成。非本人录音、非官方、非商业。',
        f'共 {report["model_families"]} 种引擎、{report["configurations"]} 套配置、{report["utterances"]} 条原始生成。网页精选 6 套配置、71 条。',
        '','配置 | 基础 CER | 情绪组 CER | 基础热 RTF | 最大原始削波率']
    pct=lambda n:'未测' if n is None else f'{n*100:.2f}%'
    for r in result:lines.append(f'{r["configuration"]} | {pct(r["baseline_cer"])} | {pct(r["emotion_cer"])} | {r["baseline_warm_median_rtf"]:.3f} | {r["max_clipped_fraction"]*100:.5f}%')
    lines.extend(['','选型：情绪对照先听 CosyVoice3；Index 提供更多明确控制，基础转写更好，但本次强情绪/语速控制下文本错误增加。F5 保留为自训固定音色基线。',
        'Index 原始版有轻微削波；发布适配器在进入上游 PCM clamp 前增加线性余量保护，21 条重跑均无削波。没有修改模型权重。',
        '语速实测（同一条 Index 开心文案）：0.8× 7.488 秒，1× 5.991 秒，1.25× 4.783 秒。',
        'CosyVoice3 的开心/温柔等条件改变音高、时长，不代表已通过情绪识别或主观盲听。未发现本轮控制指令被 ASR 转写为正文。',
        '基础组只有 8 句；参考处理并不完全一致；公共模型训练集未知。不是严格架构排名，不是商业权利证明。',
        'RK3588 尚未实测；不能拿 x86 或作者宣传速度冒充实板结果。','',report['method']])
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'model_families':5,'configurations':report['configurations'],'utterances':report['utterances']}))

if __name__=='__main__':main()
