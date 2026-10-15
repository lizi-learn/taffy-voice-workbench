# SPDX-License-Identifier: MIT
"""Separate text accuracy, frontend losses, and deployment waveform fidelity."""
import argparse
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asr-model',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from evaluate_audio import main as evaluate,normalized,edit_distance
    import numpy as np
    import soundfile as sf
    root=a.root.resolve();inputs=[]
    for name in ['fp32-t4','int8-t4','zero-t4']:
        d=root/'matrix-results'/name
        inputs.append((name,d,json.loads((d/'metrics.json').read_text())))
    combined=dict(synthetic=True,samples=[])
    for name,d,m in inputs:
        for row in m['samples']:
            combined['samples'].append({**row,'id':name+'/'+row['id'],
                'file':str((d/row['file']).relative_to(root))})
    path=root/'quality-all.json';path.write_text(json.dumps(combined,ensure_ascii=False,indent=2))
    sys.argv=['evaluate_audio','--metrics',str(path),'--asr-model',str(a.asr_model)]
    evaluate()
    data=json.loads(path.read_text());summary=[]
    for name,d,m in inputs:
        rows=[]
        for row in data['samples']:
            if not row['id'].startswith(name+'/'):continue
            original=normalized(row['text']);effective=normalized(row['stages']['normalized_text'])
            hyp=normalized(row['asr_text'])
            row.update(frontend_dropped_latin=__import__('re').findall('[A-Za-z]+',row['text']),
                effective_target_cer=edit_distance(effective,hyp)/len(effective),
                effective_edits=edit_distance(effective,hyp),effective_characters=len(effective))
            rows.append(row)
        ref=json.loads((root/'reference-x86-fp32/metrics.json').read_text())
        waveform=[]
        for row in rows:
            key=row['id'].split('/',1)[1]
            counterpart=next(s for s in ref['samples'] if s['id']==key)
            x,sr=sf.read(root/row['file']);y,sr2=sf.read(root/'reference-x86-fp32'/counterpart['file'])
            assert sr==sr2
            metric=dict(id=key,equal_length=len(x)==len(y),duration_ratio=len(x)/len(y))
            if len(x)==len(y):
                metric.update(waveform_rmse=float(np.sqrt(np.mean((x-y)**2))),
                    correlation=float(np.corrcoef(x,y)[0,1]))
            waveform.append(metric)
        m['samples']=rows
        m['evaluation']=dict(original_text_cer=sum(r['writing_normalized_edits'] for r in rows)/sum(r['writing_normalized_characters'] for r in rows),
            effective_target_cer=sum(r['effective_edits'] for r in rows)/sum(r['effective_characters'] for r in rows),
            warning='Effective-target CER does not excuse dropped content. Original input CER and Latin-loss flags remain visible. No subjective timbre score.',
            against_x86_fp32=waveform)
        (d/'evaluated.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
        summary.append(dict(profile=name,**m['evaluation']))
    (root/'quality-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
