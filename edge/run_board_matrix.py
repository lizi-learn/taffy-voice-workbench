# SPDX-License-Identifier: MIT
"""Sequential matched RK3588 CPU experiments; no background API or microphone."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve(); out=root/'matrix-results'
    if out.exists():raise FileExistsError('Preserve previous measurements; choose a fresh test root')
    out.mkdir(mode=0o700)
    env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'4',
         'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'}
    metadata=dict(start=datetime.now().astimezone().isoformat(),
        board=Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode(),
        exclusive_business_workloads_stopped=True,network_ssh_audio_retained=True,
        cpu_policy='Four A76 cores (4–7); 1/2-core sweeps stay on A76; 8-thread sweep uses both clusters. Existing performance governor unchanged.',
        results=[])
    cases=[('fp32',4,'4-7',None,1),('int8',4,'4-7',None,1),('zero',4,'4-7',None,1),
           ('int8',1,'4','welcome,story,numbers',2),('int8',2,'4,5','welcome,story,numbers',2),
           ('int8',8,'0-7','welcome,story,numbers',2)]
    for context,threads,cpus,ids,repeats in cases:
        temperature=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
        if temperature>85:raise RuntimeError('Board too hot for a valid benchmark; no thermal policy changed')
        name=f'{context}-t{threads}'
        command=['taskset','-c',cpus,sys.executable,str(root/'runtime_bert.py'),
            '--models',str(root/'bert-onnx'),'--context',context,'--threads',str(threads),
            '--prompts',str(root/'prompts.json'),'--out',str(out/name),'--repeats',str(repeats),
            '--allow-latin-loss']
        if ids:command+=['--ids',ids]
        print('starting',name,'temperature',temperature,flush=True)
        subprocess.run(command,env=env,check=True)
        metadata['results'].append(dict(profile=name,temperature_before_c=temperature,
            temperature_after_c=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000,
            cpus=cpus,metrics=str(Path(name)/'metrics.json')))
        (out/'hardware.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print('board_matrix_complete',len(cases),'profiles',flush=True)

if __name__=='__main__':main()
