"""Optional official GitHub metadata snapshot, never a quality score."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

def main():
    path=Path(__file__).resolve().parents[1]/'research.json'
    rows=json.loads(path.read_text())
    def fetch(row):
        u=urlsplit(row['url'])
        if u.netloc!='github.com':return row
        repo='/'.join(u.path.strip('/').split('/')[:2])
        r=subprocess.run(['gh','api','repos/'+repo,'--jq','{stars:.stargazers_count,archived:.archived,pushed_at:.pushed_at,default_branch:.default_branch}'],capture_output=True,text=True)
        if r.returncode==0:
            row['github']=json.loads(r.stdout)
            row['github']['checked_at']=datetime.now(timezone.utc).isoformat()
        return row
    with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(fetch,rows))
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print('Official GitHub metadata checked:',sum('github'in r for r in rows))

if __name__=='__main__':main()
