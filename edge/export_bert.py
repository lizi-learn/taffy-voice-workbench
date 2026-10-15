# SPDX-License-Identifier: AGPL-3.0-only
# Split inference derives from the pinned Bert-VITS2 Space's models.py (AGPL).
"""Export the SAME fixed voice, with external randomness for parity testing.

Frontend BERT, acoustic encoder/duration and flow/vocoder are separate. No zero
BERT shortcut. No quantization of the vocoder. Models/recordings stay private.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    source,out=a.source.resolve(),a.out.resolve()
    if out.exists():raise FileExistsError('Use a fresh export directory')
    os.umask(0o077); out.mkdir(parents=True)
    os.chdir(source); sys.path.insert(0,str(source))
    import torch
    from torch import nn
    import numpy as np
    import onnx
    import onnxruntime as ort
    import utils
    from models import SynthesizerTrn
    from text.symbols import symbols
    from text.cleaner import clean_text
    from text import cleaned_text_to_sequence
    import commons
    torch.set_num_threads(4)
    h=utils.get_hparams_from_file(str(source/'logs/Taffy/config.json'))
    m=SynthesizerTrn(len(symbols),h.data.filter_length//2+1,h.train.segment_size//h.data.hop_length,
        n_speakers=h.data.n_speakers,**h.model).eval()
    weights=source/'logs/Taffy/G_15800.pth'
    ckpt=torch.load(weights,map_location='cpu',weights_only=True)
    check=m.load_state_dict(ckpt['model'],strict=False)
    assert not check.missing_keys
    assert all('.cond_pre.' in k or '.cond_layer.' in k for k in check.unexpected_keys)
    del ckpt
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__(); self.enc=m.enc_p; self.emb=m.emb_g; self.sdp=m.sdp; self.dp=m.dp
        def forward(self,phone,tone,language,bert,duration_noise):
            g=self.emb(torch.zeros(1,dtype=torch.long,device=phone.device)).unsqueeze(-1)
            lengths=torch._shape_as_tensor(phone)[1].reshape(1).to(phone.device)
            x,mean,logs,mask=self.enc(phone,lengths,tone,language,bert,g=g)
            s=self.sdp
            hidden=s.pre(x.detach())+s.cond(g.detach())
            hidden=s.proj(s.convs(hidden,mask))*mask
            flows=list(reversed(s.flows)); flows=flows[:-2]+[flows[-1]]
            z=duration_noise*.6
            for flow in flows:z=flow(z,mask,g=hidden,reverse=True)
            logw=z[:,:1,:]*.2+self.dp(x,mask,g=g)*.8
            duration=torch.ceil(torch.exp(logw)*mask)
            return mean,logs,duration,g
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__(); self.flow=m.flow; self.dec=m.dec
        def forward(self,z,g):
            mask=torch.ones_like(z[:,:1,:])
            return self.dec(self.flow(z,mask,g=g,reverse=True)*mask,g=g)
    enc,dec=Encoder().eval(),Decoder().eval()
    # Exact text frontend is also shipped to the private board bundle, with its license.
    front=out/'frontend/text'; front.mkdir(parents=True)
    for name in ['__init__.py','symbols.py','chinese.py','tone_sandhi.py','opencpop-strict.txt']:
        shutil.copy2(source/'text'/name,front/name)
    shutil.copy2(source/'LICENSE',out/'BERT-VITS2-AGPL-LICENSE.txt')
    text='这是一段人工智能合成的语音。'
    normalized,phones,tones,word2ph=clean_text(text,'ZH')
    phones,tones,langs=cleaned_text_to_sequence(phones,tones,'ZH')
    phones,tones,langs=[commons.intersperse(v,0) for v in [phones,tones,langs]]
    n=len(phones)
    inputs=(torch.tensor([phones]),torch.tensor([tones]),torch.tensor([langs]),torch.randn(1,1024,n),torch.randn(1,2,n))
    with torch.inference_mode():
        torch.onnx.export(enc,inputs,str(out/'encoder.onnx'),opset_version=17,dynamo=False,
            input_names=['phone','tone','language','bert','duration_noise'],output_names=['mean','logs','duration','g'],
            dynamic_axes={k:{1:'phones'} for k in ['phone','tone','language']}|
                {k:{2:'phones'} for k in ['bert','duration_noise','mean','logs','duration']})
        torch.onnx.export(dec,(torch.randn(1,h.model.inter_channels,100),torch.randn(1,h.model.gin_channels,1)),
            str(out/'decoder.onnx'),opset_version=17,dynamo=False,input_names=['z','g'],output_names=['audio'],
            dynamic_axes={'z':{2:'frames'},'audio':{2:'samples'}})
    print('exported acoustic graphs',flush=True)
    # A checkpoint can contain optimizer/posterior/discriminator tensors never used in inference.
    # Preserve the trained voice by pruning unused graph branches, NOT by retraining a generic voice.
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    modeldir=source/'bert/chinese-roberta-wwm-ext-large'
    token=AutoTokenizer.from_pretrained(modeldir)
    b=AutoModelForMaskedLM.from_pretrained(modeldir).eval()
    class Context(nn.Module):
        def __init__(self):super().__init__(); self.bert=b.bert
        def forward(self,input_ids,attention_mask,token_type_ids):
            return self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,
                output_hidden_states=True).hidden_states[-3]
    context=Context().eval()
    ids=token(normalized,return_tensors='pt')
    with torch.inference_mode():
        torch.onnx.export(context,tuple(ids[k] for k in ['input_ids','attention_mask','token_type_ids']),
            str(out/'bert-fp32.onnx'),opset_version=17,dynamo=False,
            input_names=['input_ids','attention_mask','token_type_ids'],output_names=['context'],
            dynamic_axes={k:{1:'characters'} for k in ['input_ids','attention_mask','token_type_ids','context']})
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']:
        if (modeldir/name).is_file():shutil.copy2(modeldir/name,out/name)
    print('exported BERT context; MLM head and unused final layers pruned',flush=True)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(out/'bert-fp32.onnx'),str(out/'bert-int8.onnx'),
        op_types_to_quantize=['MatMul'],weight_type=QuantType.QInt8,per_channel=True,
        extra_options={'MatMulConstBOnly':True})
    print('quantized only constant BERT MatMul weights',flush=True)
    # Validate both shapes and numerical parity with externally supplied randomness.
    opt=ort.SessionOptions(); opt.intra_op_num_threads=4; opt.inter_op_num_threads=1
    sessions={name:ort.InferenceSession(str(out/(name+'.onnx')),sess_options=opt,providers=['CPUExecutionProvider'])
        for name in ['encoder','decoder','bert-fp32','bert-int8']}
    corpus=[text,'今天的阳光暖暖的，小猫在窗边看云。','现在是十月十五日，下午三点二十分。请记得喝水，也要记得休息喵。']
    results=[]
    for raw in corpus:
        norm,ph,to,w2p=clean_text(raw,'ZH'); ph,to,la=cleaned_text_to_sequence(ph,to,'ZH')
        ph,to,la=[commons.intersperse(v,0) for v in [ph,to,la]]
        w2p=[v*2 for v in w2p]; w2p[0]+=1
        ids=token(norm,return_tensors='pt')
        with torch.inference_mode():expected=context(**ids).numpy()
        feed={k:v.numpy() for k,v in ids.items()}
        contexts={name:sessions[name].run(None,feed)[0] for name in ['bert-fp32','bert-int8']}
        feature=np.repeat(expected[0],w2p,axis=0).T[None].astype(np.float32)
        rng=np.random.default_rng(20261015)
        feed={'phone':np.array([ph],dtype=np.int64),'tone':np.array([to],dtype=np.int64),
              'language':np.array([la],dtype=np.int64),'bert':feature,
              'duration_noise':rng.standard_normal((1,2,len(ph))).astype(np.float32)}
        with torch.inference_mode():pt=[v.numpy() for v in enc(*[torch.from_numpy(feed[k]) for k in ['phone','tone','language','bert','duration_noise']])]
        actual=sessions['encoder'].run(None,feed)
        assert np.array_equal(pt[2],actual[2]),'FP32 export changed phoneme durations'
        durations=pt[2].astype(np.int64).reshape(-1)
        assert 0<int(durations.sum())<2400
        mean=np.repeat(pt[0],durations,axis=2); logs=np.repeat(pt[1],durations,axis=2)
        z=(mean+rng.standard_normal(mean.shape).astype(np.float32)*np.exp(logs)*.35).astype(np.float32)
        with torch.inference_mode():audio=dec(torch.from_numpy(z),torch.from_numpy(pt[3])).numpy()
        ort_audio=sessions['decoder'].run(None,{'z':z,'g':pt[3]})[0]
        delta=float(np.sqrt(np.mean((audio-ort_audio)**2)))
        assert delta<1e-3,('waveform parity failure',delta)
        results.append(dict(text=raw,phones=len(ph),frames=int(durations.sum()),waveform_rmse=delta,
            waveform_correlation=float(np.corrcoef(audio.reshape(-1),ort_audio.reshape(-1))[0,1]),
            bert_fp32_rmse=float(np.sqrt(np.mean((expected-contexts['bert-fp32'])**2))),
            bert_int8_cosine=float(np.dot(expected.ravel(),contexts['bert-int8'].ravel())/
                (np.linalg.norm(expected)*np.linalg.norm(contexts['bert-int8'])))))
    import importlib.metadata as meta
    manifest=dict(synthetic=True,family='bert',source_revision='8d3a82bc3a94242cf91cee4f32902f13357653e9',
        source_weight_sha256=sha(weights),sample_rate=h.data.sampling_rate,inter_channels=h.model.inter_channels,
        parameters=dict(sdp_ratio=.2,noise_scale=.35,noise_scale_w=.6),
        noise_policy='External NumPy PCG64 noises for backend parity, not waveform-identical to the previous Torch seeded baseline.',
        language='ZH; retains upstream Chinese normalization/tone sandhi; not a universal multilingual frontend',
        held_out_device_test=False,parity=results,dependencies={k:meta.version(k) for k in ['onnx','onnxruntime','tokenizers','pypinyin','jieba','cn2an','numpy']},
        files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.glob('*.onnx'))])
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
