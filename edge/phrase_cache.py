# SPDX-License-Identifier: MIT
"""Versioned exact-byte cache for permitted, pre-generated AI WAVs, NOT TTS.

No text normalization, pitch shifting or transcoding. Profile revision must
change whenever weights/reference/seed/settings change. Cache misses are errors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import wave

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()

def key(text,profile):
    if not text or not profile: raise ValueError('Text and versioned profile required')
    return hashlib.sha256(json.dumps([text,profile],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def put(root,text,profile,source):
    folder=Path(root)/key(text,profile)
    with wave.open(str(source),'rb') as f:
        if f.getnframes()==0 or f.getcomptype()!='NONE': raise ValueError('A nonempty PCM WAV is required')
        metadata=dict(sample_rate=f.getframerate(),channels=f.getnchannels(),frames=f.getnframes())
    folder.mkdir(parents=True,exist_ok=False,mode=0o700)
    shutil.copyfile(source,folder/'AI.wav')
    metadata.update(text=text,profile=profile,synthetic=True,sha256=digest(folder/'AI.wav'))
    (folder/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    return folder/'AI.wav'

def get(root,text,profile):
    folder=Path(root)/key(text,profile)
    m=json.loads((folder/'metadata.json').read_text())
    path=folder/'AI.wav'
    if m['text']!=text or m['profile']!=profile or digest(path)!=m['sha256']:
        raise ValueError('Cache integrity mismatch')
    return path

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['put','get'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--text',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--source',type=Path)
    a=p.parse_args()
    if a.action=='put' and not a.source: p.error('--source required for put')
    print(put(a.root,a.text,a.profile,a.source) if a.action=='put' else get(a.root,a.text,a.profile))

if __name__=='__main__': main()
