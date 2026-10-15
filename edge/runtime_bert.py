# SPDX-License-Identifier: MIT
"""Offline ONNX CPU runtime; no Torch/GPU or network API required on the board.

The separately supplied, pinned Chinese frontend is AGPL/Apache licensed. Keep
its notices. Zero-context mode is an intentionally degraded ablation, not a
recommended optimization. Never call cached features arbitrary-text synthesis.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import re

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def frame_windows(frames,chunk,overlap=16):
    if frames<=0 or chunk<=0 or overlap<16:raise ValueError('Positive lengths and at least 16 context frames required')
    for start in range(0,frames,chunk):
        end=min(frames,start+chunk)
        yield start,end,max(0,start-overlap),min(frames,end+overlap)

def check_supported_text(text,allow_latin_loss=False):
    latin=re.findall('[A-Za-z]+',text)
    if latin and not allow_latin_loss:
        raise ValueError('Pinned Chinese frontend drops Latin letters. Spell them in Chinese or explicitly allow this research ablation.')
    return latin

class BertCPU:
    def __init__(self, models, context='fp32', threads=4, split=False, allow_latin_loss=False):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np=np; self.root=Path(models).resolve(); self.context=context
        self.allow_latin_loss=allow_latin_loss
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        sys.path.insert(0,str(self.root/'frontend'))
        from text.chinese import text_normalize,g2p
        from text import cleaned_text_to_sequence
        self.normalize,self.g2p,self.sequence=text_normalize,g2p,cleaned_text_to_sequence
        options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        options.add_session_config_entry('session.inter_op.allow_spinning','0')
        self.tokenizer=Tokenizer.from_file(str(self.root/'tokenizer.json'))
        # Load and validate only the selected models. File hashes are included in cold start.
        self.sessions={}
        names=['encoder']+(['flow','vocoder'] if split else ['decoder'])+([] if context=='zero' else ['bert-'+context])
        expected={r['name']:r['sha256'] for r in self.manifest['files']}
        for name in names:
            path=self.root/(name+'.onnx')
            if digest(path)!=expected[path.name]:raise ValueError('Model hash mismatch: '+path.name)
            self.sessions[name]=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def prepare(self,text,seed=20261015):
        np=self.np; start=time.perf_counter()
        latin=check_supported_text(text,self.allow_latin_loss)
        norm=self.normalize(text)
        if not norm:raise ValueError('No supported Chinese text after upstream normalization')
        phones,tones,w2p=self.g2p(norm)
        phones,tones,languages=self.sequence(phones,tones,'ZH')
        def blanks(seq):
            out=[0]*(len(seq)*2+1);out[1::2]=seq;return out
        phones,tones,languages=[blanks(s) for s in [phones,tones,languages]]
        w2p=[x*2 for x in w2p];w2p[0]+=1
        ids=self.tokenizer.encode(norm)
        assert len(ids.ids)==len(w2p),'Tokenizer/phoneme alignment mismatch'
        frontend=time.perf_counter()-start
        start=time.perf_counter()
        if self.context=='zero':word=np.zeros((len(w2p),1024),dtype=np.float32)
        else:
            word=self.sessions['bert-'+self.context].run(None,{
                'input_ids':np.array([ids.ids],dtype=np.int64),
                'attention_mask':np.array([ids.attention_mask],dtype=np.int64),
                'token_type_ids':np.array([ids.type_ids],dtype=np.int64)})[0][0]
        bert_seconds=time.perf_counter()-start
        feature=np.repeat(word,w2p,axis=0).T[None].astype(np.float32)
        rng=np.random.default_rng(seed)
        start=time.perf_counter()
        mean,logs,duration,g=self.sessions['encoder'].run(None,{
            'phone':np.array([phones],dtype=np.int64),'tone':np.array([tones],dtype=np.int64),
            'language':np.array([languages],dtype=np.int64),'bert':feature,
            'duration_noise':rng.standard_normal((1,2,len(phones))).astype(np.float32)})
        durations=duration.astype(np.int64).reshape(-1)
        if not 0<int(durations.sum())<=3000:raise ValueError('Unsafe generated duration')
        mean=np.repeat(mean,durations,axis=2);logs=np.repeat(logs,durations,axis=2)
        z=(mean+rng.standard_normal(mean.shape).astype(np.float32)*np.exp(logs)*.35).astype(np.float32)
        acoustic_seconds=time.perf_counter()-start
        return z,g,dict(frontend_seconds=frontend,bert_seconds=bert_seconds,
            acoustic_seconds=acoustic_seconds,
            phonemes=len(phones),frames=int(durations.sum()),normalized_text=norm,
            unsupported_latin=latin,
            bert_fingerprint=hashlib.sha256(feature.tobytes()).hexdigest())

    def synthesize(self,text,seed=20261015):
        z,g,stages=self.prepare(text,seed)
        start=time.perf_counter()
        audio=self.sessions['decoder'].run(None,{'z':z,'g':g})[0].reshape(-1)
        stages['decoder_seconds']=time.perf_counter()-start
        if not self.np.isfinite(audio).all():raise RuntimeError('Invalid PCM')
        return audio,stages

    def stream_chunks(self,text,seed=20261015,chunk=64,overlap=16):
        z,g,stages=self.prepare(text,seed)
        start=time.perf_counter()
        latent=self.sessions['flow'].run(None,{'z':z,'g':g})[0]
        stages['flow_seconds']=time.perf_counter()-start
        stages['vocoder_seconds']=0
        self.last_stream_stages=stages
        hop=self.manifest['streaming_split']['hop_length']
        for first,last,left,right in frame_windows(latent.shape[-1],chunk,overlap):
            start=time.perf_counter()
            clip=self.sessions['vocoder'].run(None,{'z':latent[:,:,left:right],'g':g})[0].reshape(-1)
            stages['vocoder_seconds']+=time.perf_counter()-start
            clip=clip[(first-left)*hop:(last-left)*hop]
            if not self.np.isfinite(clip).all():raise RuntimeError('Invalid PCM chunk')
            yield clip

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--context',choices=['fp32','int8','zero'],default='fp32')
    p.add_argument('--allow-latin-loss',action='store_true',help='Research-only: explicitly allow the pinned frontend to drop Latin letters')
    p.add_argument('--threads',type=int,choices=[1,2,4,8],default=4)
    p.add_argument('--ids',help='Comma-separated prompt IDs for matched performance sweeps')
    p.add_argument('--stream-chunk',type=int,choices=[0,32,64,128,256],default=0)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--repeats',type=int,default=1)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a fresh output directory')
    a.out.mkdir(parents=True,mode=0o700)
    import numpy as np
    import soundfile as sf
    began=time.perf_counter();engine=BertCPU(a.models,a.context,a.threads,split=bool(a.stream_chunk),allow_latin_loss=a.allow_latin_loss)
    cold=time.perf_counter()-began
    prompts=json.loads(a.prompts.read_text())
    if isinstance(prompts,dict):prompts=prompts['prompts']
    if a.ids:prompts=[r for r in prompts if r['id'] in a.ids.split(',')]
    if not prompts:raise ValueError('No prompts selected')
    if a.limit:prompts=prompts[:a.limit]
    rows=[]
    import platform
    for repeat in range(a.repeats):
        for row in prompts:
            start=time.perf_counter();sr=engine.manifest['sample_rate'];events=[]
            if a.stream_chunk:
                chunks=[];before=0
                for clip in engine.stream_chunks(row['text'],chunk=a.stream_chunk):
                    events.append(dict(ready_seconds=time.perf_counter()-start,audio_before_seconds=before,
                        chunk_seconds=len(clip)/sr))
                    before+=len(clip)/sr;chunks.append(clip)
                audio=np.concatenate(chunks);stages=engine.last_stream_stages
            else:audio,stages=engine.synthesize(row['text'])
            elapsed=time.perf_counter()-start
            tag=f'-c{a.stream_chunk}' if a.stream_chunk else ''
            path=a.out/f'bert-edge-{a.context}-t{a.threads}{tag}-{row["id"]}-r{repeat}-AI.wav'
            sf.write(path,audio,sr,subtype='PCM_16')
            item=dict(id=row['id'],group='baseline',text=row['text'],file=path.name,sha256=digest(path),
                seconds=len(audio)/sr,compute_seconds=elapsed,rtf=elapsed/(len(audio)/sr),
                sample_rate=sr,peak=float(abs(audio).max()),clipped_fraction=float(np.mean(abs(audio)>=.999)),
                rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                repeat=repeat,context=a.context,threads=a.threads,stages=stages,
                first_pcm_seconds=events[0]['ready_seconds'] if events else elapsed,
                minimum_gapless_start_seconds=max(e['ready_seconds']-e['audio_before_seconds'] for e in events) if events else elapsed,
                chunk_events=events)
            rows.append(item)
            report=dict(synthetic=True,device=platform.machine(),hostname=platform.node(),context=a.context,
                source_weight_sha256=engine.manifest['source_weight_sha256'],cold_load_seconds=cold,
                wall_clock_semantics='Completion includes Chinese frontend and BERT. first_pcm is first produced array, not speaker/network latency. minimum_gapless_start assumes immediate consumption at sample rate with measured production timings. Model hash checks included in process cold load; OS page cache may be warm.',
                samples=rows)
            (a.out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print('edge_sample',row['id'],a.context,a.threads,round(elapsed,3),'s','RTF',round(item['rtf'],3),
                'RSS',round(item['rss_peak_mib'],1),'MiB',flush=True)

if __name__=='__main__':main()
