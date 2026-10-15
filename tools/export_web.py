"""Export only the project's own static UI, never media, accounts or model data."""
import argparse
from pathlib import Path
import shutil

def main():
    p=argparse.ArgumentParser()
    p.add_argument('site_dist',type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    dest=root/'web'
    (dest/'voice').mkdir(parents=True,exist_ok=True)
    for name in ['style.css','favicon.svg']:
        shutil.copy2(args.site_dist/name,dest/name)
    for name in ['index.html','voice.css','voice.js','catalog.json','licenses.txt','index-license.txt','index-license-zh.txt']:
        text=(args.site_dist/'voice'/name).read_text()
        if name=='index.html':
            text=text.replace('href="../#gallery"','href="https://塔菲.com/#gallery"').replace('href="../resources/"','href="https://塔菲.com/resources/"').replace('href="../resources/nekomimi-taffy/#showcase"','href="https://塔菲.com/resources/nekomimi-taffy/#showcase"')
        (dest/'voice'/name).write_text(text)
    # Audio stays on the static public site, so the source repository remains small.
    import json
    c=json.loads((dest/'voice/catalog.json').read_text())
    for row in c['samples']:
        for field in ['raw','listen','download']:
            if field in row:row[field]='https://xn--omsy82h.com/voice/'+row[field]
    (dest/'voice/catalog.json').write_text(json.dumps(c,ensure_ascii=False,indent=2))
    print('Exported static UI; no weights, reference recordings, or credentials.')

if __name__=='__main__':main()
