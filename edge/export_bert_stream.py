# SPDX-License-Identifier: AGPL-3.0-only
# Uses the pinned Bert-VITS2 model (AGPL); no model weights redistributed.
"""Split global-context flow from local-convolution vocoder for faithful chunks."""
import argparse
import json
from pathlib import Path
import os
import sys
from export_bert import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);a=p.parse_args()
    source,folder=a.source.resolve(),a.models.resolve()
    if (folder/'flow.onnx').exists() or (folder/'vocoder.onnx').exists():raise FileExistsError('Preserve previous exports')
    os.chdir(source);sys.path.insert(0,str(source))
    import torch
    from torch import nn
    import utils
    from models import SynthesizerTrn
    from text.symbols import symbols
    import numpy as np
    import onnxruntime as ort
    torch.set_num_threads(4)
    h=utils.get_hparams_from_file(str(source/'logs/Taffy/config.json'))
    model=SynthesizerTrn(len(symbols),h.data.filter_length//2+1,h.train.segment_size//h.data.hop_length,
        n_speakers=h.data.n_speakers,**h.model).eval()
    d=torch.load(source/'logs/Taffy/G_15800.pth',map_location='cpu',weights_only=True)
    check=model.load_state_dict(d['model'],strict=False)
    assert not check.missing_keys
    assert all('.cond_pre.' in k or '.cond_layer.' in k for k in check.unexpected_keys)
    del d
    class Flow(nn.Module):
        def __init__(self):super().__init__();self.flow=model.flow
        def forward(self,z,g):return self.flow(z,torch.ones_like(z[:,:1,:]),g=g,reverse=True)
    class Vocoder(nn.Module):
        def __init__(self):super().__init__();self.dec=model.dec
        def forward(self,z,g):return self.dec(z,g=g)
    flow,vocoder=Flow().eval(),Vocoder().eval()
    for name,module in [('flow',flow),('vocoder',vocoder)]:
        with torch.inference_mode():
            torch.onnx.export(module,(torch.randn(1,h.model.inter_channels,128),torch.randn(1,h.model.gin_channels,1)),
                str(folder/(name+'.onnx')),opset_version=17,dynamo=False,input_names=['z','g'],output_names=['audio' if name=='vocoder' else 'z_out'],
                dynamic_axes={'z':{2:'frames'},'audio' if name=='vocoder' else 'z_out':{2:'samples' if name=='vocoder' else 'frames'}})
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    sessions={n:ort.InferenceSession(str(folder/(n+'.onnx')),sess_options=options,providers=['CPUExecutionProvider']) for n in ['decoder','flow','vocoder']}
    rng=np.random.default_rng(20261015)
    tests=[]
    for frames in [173,593]:
        z=rng.standard_normal((1,h.model.inter_channels,frames)).astype(np.float32)
        g=rng.standard_normal((1,h.model.gin_channels,1)).astype(np.float32)
        full=sessions['decoder'].run(None,{'z':z,'g':g})[0].reshape(-1)
        latent=sessions['flow'].run(None,{'z':z,'g':g})[0]
        split=sessions['vocoder'].run(None,{'z':latent,'g':g})[0].reshape(-1)
        assert np.max(abs(split-full))<1e-4,'Graph split changed the waveform'
        for overlap in [4,8,16,32]:
            chunks=[]
            for start in range(0,frames,64):
                end=min(frames,start+64);left=max(0,start-overlap);right=min(frames,end+overlap)
                clip=sessions['vocoder'].run(None,{'z':latent[:,:,left:right],'g':g})[0].reshape(-1)
                chunks.append(clip[(start-left)*h.data.hop_length:(end-left)*h.data.hop_length])
            joined=np.concatenate(chunks)
            assert len(joined)==len(full)
            rmse=float(np.sqrt(np.mean((joined-full)**2)))
            tests.append(dict(frames=frames,chunk_frames=64,overlap_frames=overlap,
                waveform_rmse=rmse,max_abs_error=float(abs(joined-full).max()),
                correlation=float(np.corrcoef(joined,full)[0,1])))
    manifest=json.loads((folder/'manifest.json').read_text())
    manifest['streaming_split']=dict(full_context_flow=True,hop_length=h.data.hop_length,
        chunking='Run flow on the entire sequence; overlap/crop ONLY the local-convolution vocoder. Do not chunk attention flow.',parity=tests)
    for name in ['flow','vocoder']:
        file=folder/(name+'.onnx');manifest['files'].append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file)))
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest['streaming_split'],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
