const $ = id => document.getElementById(id);
const moodInfo = {neutral:['◌','平静'],happy:['✧','开心'],sad:['☂','难过'],angry:['ϟ','生气'],surprised:['!','惊讶'],gentle:['♡','温柔']};
const labels = {emotion:'同一句话 · 情绪对照',welcome:'欢迎来到小展台',actions:'互动提示',story:'窗边的小猫',numbers:'喝水提醒'};
const E = s => String(s??'').replace(/[&<>"']/g, x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const pct = n => Number.isFinite(n) ? (n*100).toFixed(2)+'%' : '未测';
let catalog, current, mood='neutral', pair=[], revealed=false;
const player=$('audio');
function wave(target,count){target.replaceChildren(...Array.from({length:count},(_,i)=>{const b=document.createElement('i');b.style.height=(14+Math.abs(Math.sin(i*1.7)*Math.cos(i*.32))*70)+'%';b.style.animationDelay=-(i%7)*.12+'s';return b}));}
wave(document.querySelector('.wave-bars'),30);wave($('waveform'),48);
function safeStore(key,value){try{localStorage.setItem(key,JSON.stringify(value))}catch{}}
function readStore(key,fallback){try{return JSON.parse(localStorage.getItem(key))??fallback}catch{return fallback}}
function candidates(){return catalog.samples.filter(s=>s.engine===$('engine').value && s.phrase===$('phrase').value);}
function renderSample(){
  if(!catalog)return;
  player.pause();document.body.classList.remove('playing');
  $('play').textContent='▶';$('play').setAttribute('aria-label','播放 AI 试听');
  const available=catalog.samples.filter(s=>s.engine===$('engine').value);
  for(const option of $('phrase').options)option.disabled=!available.some(s=>s.phrase===option.value);
  if(!available.some(s=>s.phrase===$('phrase').value))$('phrase').value='welcome';
  const options=candidates();
  if(!options.some(s=>s.mood===mood))mood=options[0]?.mood??'neutral';
  $('moods').replaceChildren(...Object.entries(moodInfo).map(([key,[icon,name]])=>{
    const b=document.createElement('button');b.textContent=icon+' '+name;b.type='button';b.dataset.mood=key;
    b.disabled=!options.some(s=>s.mood===key);b.setAttribute('aria-pressed',String(key===mood));
    b.onclick=()=>{mood=key;renderSample();updateURL()};return b;
  }));
  const advanced=$('engine').value==='index'&&$('phrase').value==='emotion'&&mood==='happy';
  $('variantWrap').hidden=!advanced;
  const selectedVariant=advanced?$('variant').value:'standard';
  current=options.find(s=>s.mood===mood&&(selectedVariant==='standard'?s.group!=='control':s.id==='emotion-happy-'+selectedVariant));
  const engine=catalog.engines.find(e=>e.id===$('engine').value);
  $('engineNote').textContent=engine.note;
  $('play').disabled=!current;
  if(!current){$('spokenText').textContent='这个引擎还没有这组文案的实测音频。';player.removeAttribute('src');$('downloadAudio').hidden=true;$('sampleInfo').textContent='暂未提供';$('sampleDetails').replaceChildren();return}
  $('downloadAudio').hidden=false;
  $('spokenText').textContent=current.text;
  $('sampleInfo').textContent=current.seconds.toFixed(1)+' 秒 · '+(current.sample_rate/1000)+'kHz';
  player.src=current.listen;
  $('downloadAudio').href=current.download??current.raw;$('downloadAudio').download=current.download?current.file.replace(/\.wav$/,'.zip'):current.file;
  $('downloadAudio').textContent=current.download_label??'↓ 下载 AI WAV';
  $('playStatus').textContent='已选 '+engine.name+' · '+moodInfo[mood][1]+'。点击播放 AI 试听。';
  $('sampleDetails').innerHTML=`<dl><dt>原始 WAV</dt><dd>${E(current.file)}</dd><dt>条件</dt><dd>${E(current.control??'默认参考语气')}</dd><dt>生成耗时</dt><dd>${current.compute_seconds.toFixed(3)} 秒 / RTF ${current.rtf.toFixed(3)}</dd><dt>ASR 转写</dt><dd>${E(current.asr_text??'未转写')}</dd><dt>原始削波率</dt><dd>${pct(current.clipped_fraction)}</dd><dt>原始校验</dt><dd>${E(current.sha256)}</dd><dt>试听处理</dt><dd>仅线性增益，RMS 目标 0.10、峰值上限 0.90；试听处理不变速、不变调。</dd></dl>${engine.tuning?`<p>本轮候选参数（未作主观音色排名）</p><pre>${E(JSON.stringify(engine.configuration,null,2))}</pre>`:''}`;
}
function updateURL(){const u=new URL(location.href);u.searchParams.set('engine',$('engine').value);u.searchParams.set('phrase',$('phrase').value);u.searchParams.set('mood',mood);u.searchParams.set('variant',$('variant').value);u.hash='listen';history.replaceState(null,'',u);}
async function playSample(s){
  if(!s)return;
  if(player.src!==new URL(s.listen,location.href).href){player.pause();player.src=s.listen}
  try{await player.play()}catch{$('playStatus').textContent='暂时没能播放，请重试，或点击下载 WAV。'}
}
$('play').onclick=()=>player.paused?playSample(current):player.pause();
player.addEventListener('play',()=>{document.body.classList.add('playing');$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','暂停 AI 试听');$('playStatus').textContent='正在播放 AI 合成语音 · 非本人录音。'});
for(const event of ['pause','ended'])player.addEventListener(event,()=>{document.body.classList.remove('playing');$('play').textContent='▶';$('play').setAttribute('aria-label','播放 AI 试听');});
player.addEventListener('error',()=>{$('playStatus').textContent='音频暂时打不开，请检查网络后重试，或下载 WAV。';document.body.classList.remove('playing')});
$('engine').onchange=()=>{renderSample();updateURL()};$('phrase').onchange=()=>{renderSample();updateURL()};$('variant').onchange=()=>{renderSample();updateURL()};
$('share').onclick=async()=>{updateURL();try{await navigator.clipboard.writeText(location.href);$('playStatus').textContent='试听链接已复制，包含当前引擎、文案和心情。'}catch{$('playStatus').textContent='未能写入剪贴板，可直接复制地址栏。'}};
function shuffle(){
  $('blindAudio').pause();
  const scope=$('blindScope').value, phrase=$('blindPhrase').value;
  const pool=catalog.samples.filter(s=>s.phrase===phrase&&s.mood==='neutral'&&(scope==='all'||catalog.engines.find(e=>e.id===s.engine)?.family===scope));
  // Randomize a small array without relying on unstable sort comparators.
  for(let i=pool.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[pool[i],pool[j]]=[pool[j],pool[i]]}
  const control=pool.find(s=>s.engine===(scope==='gsv'?'gsv':'bertcontrol'));
  if(scope!=='all'&&control){
    const variant=pool.find(s=>s!==control&&catalog.engines.find(e=>e.id===s.engine)?.tuning);
    pair=variant?[control,variant]:[];
    if(Math.random()<.5)pair.reverse();
  }else pair=pool.slice(0,2);
  revealed=false;$('blindAudio').removeAttribute('src');$('blindAudio').load();$('blindAudio').hidden=true;
  $('blindResult').textContent='新的一组 A / B 已准备好。两条都是 AI 合成。';$('voteStatus').textContent='偏好只保存在本浏览器。';
  $('blindA').disabled=$('blindB').disabled=pair.length<2;
}
function engineName(s){return catalog.engines.find(e=>e.id===s.engine)?.name??s.engine}
async function blindPlay(index){if(!pair[index])return;player.pause();const b=$('blindAudio');b.hidden=false;b.src=pair[index].listen;try{await b.play()}catch{$('blindResult').textContent='暂时没能播放，请重试。'}}
$('blindA').onclick=()=>{blindPlay(0);$('blindResult').textContent=revealed?'A · '+engineName(pair[0]):'正在试听 A（AI 合成）'};
$('blindB').onclick=()=>{blindPlay(1);$('blindResult').textContent=revealed?'B · '+engineName(pair[1]):'正在试听 B（AI 合成）'};
player.addEventListener('play',()=>$('blindAudio').pause());
$('reveal').onclick=()=>{if(pair.length<2)return;revealed=true;$('blindResult').textContent='A · '+engineName(pair[0])+'　/　B · '+engineName(pair[1])};
$('reshuffle').onclick=shuffle;
$('blindScope').onchange=shuffle;$('blindPhrase').onchange=shuffle;
$('blindAudio').addEventListener('error',()=>{$('blindResult').textContent='A/B 音频暂时无法加载，请检查网络后重试。'});
document.querySelectorAll('[data-vote]').forEach(b=>b.onclick=()=>{if(pair.length<2)return;const votes=readStore('taffy-voice-votes',[]);votes.push({time:new Date().toISOString(),a:pair[0].file,b:pair[1].file,pick:b.dataset.vote});safeStore('taffy-voice-votes',votes.slice(-100));$('voteStatus').textContent='已在本机记下：更喜欢 '+b.dataset.vote+'。没有上传服务器。'});
function research(){
  if(!catalog)return;
  const filter=$('researchFilter').value;
  const rows=catalog.research.filter(r=>filter==='all'||(filter==='measured'?r.status==='measured':r.tags.includes(filter)));
  $('researchCards').innerHTML=rows.map(r=>`<article class="research-card"><span class="status ${r.status==='measured'?'measured':''}">${r.status==='measured'?'已实测':r.status==='deferred'?'暂缓 / 条件限制':'官方文档核验 · 未实测'}${r.github?' · ★ '+(r.github.stars/1000).toFixed(1)+'k':''}</span><h3>${E(r.name)}</h3><p>${E(r.note)}</p><span class="tag-line">${E(r.license)} · ${E(r.focus)}${r.github?' · 热度快照 '+r.github.checked_at.slice(0,10):''}</span><a href="${E(r.url)}" target="_blank" rel="noopener">官方仓库 ↗</a></article>`).join('');
}
$('researchFilter').onchange=research;
try{
  const r=await fetch('./catalog.json',{cache:'no-cache'});if(!r.ok)throw Error('HTTP '+r.status);catalog=await r.json();
  $('counts').innerHTML=[[''+catalog.raw_utterances,'条真实生成样本 · 原版完整保留'],[''+catalog.tuning_configurations,'套新增音色调优对照'],[''+catalog.model_families,'种引擎 · '+catalog.engines.length+' 套试听方案'],['本地推理','只公开静态试听，GPU 不外借']].map(([n,t])=>`<div><strong>${E(n)}</strong><span>${E(t)}</span></div>`).join('');
  $('engine').innerHTML=[['gsv','主线 · GPT‑SoVITS'],['bert','重点备选 · Bert‑VITS2'],['other','其他研究 / 情绪实验']].map(([family,title])=>`<optgroup label="${E(title)}">${catalog.engines.filter(e=>family==='other'?!['gsv','bert'].includes(e.family):e.family===family).map(e=>`<option value="${E(e.id)}">${E(e.name)}${e.tuning?' · 待试听':''}</option>`).join('')}</optgroup>`).join('');
  const params=new URLSearchParams(location.search);
  $('engine').value=catalog.engines.some(e=>e.id===params.get('engine'))?params.get('engine'):'gsv';
  $('phrase').value=params.has('phrase')&&labels[params.get('phrase')]?params.get('phrase'):'welcome';mood=moodInfo[params.get('mood')]?params.get('mood'):'neutral';
  $('variant').value=['standard','soft','slow','fast'].includes(params.get('variant'))?params.get('variant'):'standard';
  $('measurements').innerHTML=catalog.engines.map(e=>`<tr><td>${E(e.name)}<small>${E(e.version)}${e.tuning?' · 调优候选，非已验收升级':''}</small></td><td>${pct(e.cer)}</td><td>${Number.isFinite(e.emotion_cer)?pct(e.emotion_cer):'未测 / 不支持'}</td><td>${Number.isFinite(e.rtf)?e.rtf.toFixed(3):'未测'}</td><td>${Number.isFinite(e.vram_mib)?(e.vram_mib/1024).toFixed(2)+' GiB':'未测'}</td><td>${E(e.emotion)}</td></tr>`).join('');
  renderSample();shuffle();research();
}catch(error){$('counts').textContent='实验清单暂时未加载成功，请刷新重试。';$('playStatus').textContent='清单读取失败，请检查网络后重试。';$('spokenText').textContent='暂时没能把声音装进来。';console.error('Voice catalog unavailable',error.message)}
