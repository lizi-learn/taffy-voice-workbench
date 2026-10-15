# SPDX-License-Identifier: MIT
"""Allow-list edge measurements; never copy hostnames, addresses or references."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import wave

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def pcm(path):
    with wave.open(str(path),'rb') as f:
        assert f.getnchannels()==1 and f.getsampwidth()==2
        values=array('h',f.readframes(f.getnframes()))
        if sys.byteorder!='little':values.byteswap()
        return values,f.getparams()

def listening_copy(source,target):
    values,params=pcm(source)
    peak=max(abs(x) for x in values)/32768
    rms=math.sqrt(sum((x/32768)**2 for x in values)/len(values))
    gain=min(.9/max(peak,1e-6),.1/max(rms,1e-6))
    output=array('h',(int(max(-32768,min(32767,x*gain))) for x in values))
    if sys.byteorder!='little':output.byteswap()
    with wave.open(str(target),'wb') as f:f.setparams(params);f.writeframes(output.tobytes())
    return gain

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--site-dist',type=Path,required=True);a=p.parse_args()
    root=a.inputs.resolve();dest=a.site_dist/'voice/edge';audio=dest/'audio'
    audio.mkdir(parents=True,exist_ok=True)
    records=[]
    for group in ['matrix-results','followup-results','chunk-results']:
        for path in sorted((root/group).glob('*/metrics.json')):
            m=json.loads(path.read_text());allrows=m['samples']
            warm=([r for r in allrows if r['repeat']==1] if max(r['repeat'] for r in allrows)>0 else allrows[1:])
            key=path.parent.name
            records.append(dict(id=group+'/'+key,context=m['context'],
                cpu_threads=allrows[0]['threads'],warm_n=len(warm),total_n=len(allrows),
                median_rtf=statistics.median(r['rtf'] for r in warm),
                median_first_pcm_seconds=statistics.median(r.get('first_pcm_seconds',r['compute_seconds']) for r in warm),
                first_pcm_basis='Explicit chunk events' if 'first_pcm_seconds' in warm[0] else 'Whole-utterance completion; no incremental PCM path',
                median_gapless_start_seconds=statistics.median(r.get('minimum_gapless_start_seconds',r['compute_seconds']) for r in warm),
                peak_rss_mib=max(r['rss_peak_mib'] for r in allrows),
                process_load_seconds=m['cold_load_seconds'],
                stage_medians={k:statistics.median(r['stages'][k] for r in warm)
                    for k in warm[0]['stages'] if k.endswith('_seconds')},
                samples=[{k:r[k] for k in ['id','text','seconds','compute_seconds','rtf','repeat',
                    'sha256','sample_rate','first_pcm_seconds','minimum_gapless_start_seconds',
                    'rss_peak_mib','stages','chunk_events'] if k in r} for r in allrows]))
    variants=[
        ('x86-fp32','x86 · FP32 数值参考',root/'reference-x86-fp32',0),
        ('rk-fp32','RK · FP32 整句',root/'followup-results/fp32-c0',1),
        ('rk-stream','RK · FP32 / 128 帧',root/'chunk-results/fp32-c128',1),
        ('rk-int8','RK · BERT INT8 / 128 帧',root/'chunk-results/int8-c128',1),
        ('rk-zero','RK · 删除 BERT 的消融',root/'matrix-results/zero-t4',0),
        ('cosy-instruct','Cosy3 · 原指令模式',root/'cosy-mode-results2/instruct2',None),
        ('cosy-zero','Cosy3 · 保留参考 token',root/'cosy-mode-results2/zero_shot',None)]
    samples=[];mode_results=[]
    for ident,name,folder,repeat in variants:
        m=json.loads((folder/'metrics.json').read_text())
        if ident.startswith('cosy'):
            assert 'evaluation' in m,'Finish evaluation before publishing'
            mode_results.append(dict(id=ident,**m['evaluation']['groups']['baseline'],
                method=m['method'],reference_sha256=m['reference_sha256'],
                samples=[{k:s[k] for k in ['id','text','seconds','compute_seconds','rtf',
                    'sha256','asr_text','writing_normalized_cer','clipped_fraction'] if k in s}
                    for s in m['samples']]))
        for row in m['samples']:
            if repeat is not None and row['repeat']!=repeat:continue
            prompt=row['id'].split('/')[-1]
            # Three matched showcase texts, no private reference recordings.
            if prompt not in {'welcome','story','numbers'}:continue
            source=folder/row['file']
            assert sha(source)==row['sha256']
            filename=f'{ident}-{prompt}-AI.wav';raw=audio/filename
            shutil.copy2(source,raw)
            listen=audio/(raw.stem+'-listen.wav');gain=listening_copy(raw,listen)
            samples.append(dict(variant=ident,name=name,phrase=prompt,text=row['text'],synthetic=True,
                raw='audio/'+raw.name,listen='audio/'+listen.name,listen_gain=gain,
                sha256=row['sha256'],seconds=row['seconds'],sample_rate=row['sample_rate'],
                asr_text=row.get('asr_text'),cer=row.get('writing_normalized_cer')))
    # Numerical output preservation on the real board, not a voice identity score.
    stream_parity=[]
    for context in ['fp32','int8']:
        stream=json.loads((root/f'chunk-results/{context}-c128/metrics.json').read_text())
        full=json.loads((root/f'followup-results/{context}-c0/metrics.json').read_text())
        for row in stream['samples']:
            if row['repeat']!=1:continue
            other=next(r for r in full['samples'] if r['id']==row['id'] and r['repeat']==1)
            x,_=pcm(root/f'chunk-results/{context}-c128'/row['file'])
            y,_=pcm(root/f'followup-results/{context}-c0'/other['file'])
            assert len(x)==len(y)
            errors=[i-j for i,j in zip(x,y)]
            stream_parity.append(dict(context=context,id=row['id'],equal_length=True,
                max_pcm_lsb=max(abs(v) for v in errors),
                waveform_rmse=math.sqrt(sum(v*v for v in errors)/len(errors))/32768))
    manifest=json.loads((root/'export-manifest.json').read_text())
    report=dict(date='2026-10-15',synthetic=True,rk3588_measured=True,npu_measured=False,
        gpu_on_board=False,human_timbre_accepted=False,
        board='RK3588 / 16GB / Ubuntu 22.04 / aarch64',runtime='ONNX Runtime 1.23.2 CPUExecutionProvider',
        conditions='Business workloads stopped by owner permission. Existing performance governor retained. 4 A76 cores for chunk controls; matrix explicitly records thread counts. No microphone recording. No thermal/firmware/network changes.',
        method='RTF=compute/audio duration. Warm matched chunk controls: three texts, two repeats, median of second repeat. Peak RSS over whole process. First PCM array excludes model load, playback and network. Hash checks included in load; page cache may be warm. Fixed NumPy PCG64 external noise, not old Torch RNG. No subjective timbre/MOS score. All generated WAVs are AI.',
        warnings=['Limited short-text test, not long-run or concurrent production validation.',
            'Pinned Chinese frontend drops Latin letters; runtime now rejects them by default. Research comparison retains original CER and explicit loss flags.',
            'INT8 BERT changes conditioning and is not lossless. Zero BERT is a destructive ablation, not a recommended optimization.',
            'GPU GPT was specially trained; Cosy is zero-shot and public Bert training data unknown. No architecture-only ranking.',
            'GPT-SoVITS ONNX preliminary export only; not yet verified on RK3588. No NPU claim.'],
        records=records,quality=json.loads((root/'quality-summary.json').read_text()),
        source_revision=manifest['source_revision'],source_weight_sha256=manifest['source_weight_sha256'],
        noise_policy=manifest['noise_policy'],parameters=manifest['parameters'],
        phrase_cache=json.loads((root/'phrase-cache-verification.json').read_text()),
        export_parity=manifest['parity'],vocoder_overlap_parity=manifest['streaming_split']['parity'],
        board_stream_parity=stream_parity,cosy_modes=mode_results,samples=samples)
    encoded=json.dumps(report,ensure_ascii=False,indent=2)
    # Fail closed if private inventory ever enters this allow-listed export.
    for forbidden in ['172.16.','/home/wj','/home/cat','P2-1010','password','serial_number']:
        assert forbidden not in encoded,forbidden
    (dest/'report.json').write_text(encoded)
    (Path(__file__).resolve().parents[1]/'reports/rk3588-20261015.json').write_text(encoded)
    print(json.dumps(dict(profiles=len(records),public_new_audio=len(samples),private_inventory=False)))

if __name__=='__main__':main()
