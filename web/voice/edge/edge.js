const $=s=>document.querySelector(s);
const notes={
  'x86-fp32':'部署数值参考。与板端使用相同 NumPy 外部噪声，不是旧 GPU / Torch 随机数版本。',
  'rk-fp32':'完整 FP32 条件与声学生成器，整句输出。作为 RK 分块方案的匹配对照。',
  'rk-stream':'当前优先：完整 flow + 128 帧声码器块，左右保留 16 帧，再裁去重叠。',
  'rk-int8':'只量化 BERT 常量 MatMul 权重，声学部分 FP32。省内存，不宣称音色无损。',
  'rk-zero':'故意删除 BERT 的破坏性消融。用于说明 CER 不能决定音色，非推荐方案。',
  'cosy-instruct':'原来使用的 instruct2 情绪指令路径；不保留 LLM 参考语音 token。GPU 生成。',
  'cosy-zero':'官方 zero_shot 路径：保留 LLM 参考语音 token，提供同一参考的准确转写。GPU 生成。',
  'gsv':'你喜欢的原混合版。SoVITS 专门训练，GPT 基座；与 Bert / Cosy 训练条件不同。'};
let current=[];
function element(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function card(s){
  const c=element('article',undefined,'audio-card'+(s.variant==='gsv'?' ai-original':''));
  c.append(element('span','AI 合成 · 非本人录音','kind'),element('h3',s.name),element('p',notes[s.variant]));
  const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=s.listen;audio.setAttribute('aria-label',s.name+'试听');
  audio.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause();}));
  c.append(audio);const a=element('a','↓ 原始 AI WAV · '+s.seconds.toFixed(2)+' 秒');a.href=s.raw;a.download='';c.append(a);return c;
}
function render(samples){
  current.forEach(a=>{a.pause();a.removeAttribute('src');a.load();});
  const phrase=$('#edgePhrase').value,rows=samples.filter(s=>s.phrase===phrase);
  $('#edgeText').textContent=rows[0]?.text||'没有匹配文案';
  $('#edgePlayers').replaceChildren(...rows.map(card));current=[...document.querySelectorAll('audio')];
}
async function start(){
  const response=await fetch('./report.json');if(!response.ok)throw Error('实测清单暂不可用');
  const r=await response.json();
  const rows=[['followup-results/fp32-c0','FP32 整句','不分块，等待整句'],['followup-results/fp32-c64','FP32 / 64 帧','更早首段，但总 RTF > 1'],['chunk-results/fp32-c128','FP32 / 128 帧','当前短句折中'],['chunk-results/fp32-c256','FP32 / 256 帧','吞吐更好，首段更晚'],['chunk-results/int8-c128','BERT INT8 / 128 帧','内存少，但不是无损']];
  $('#performance').replaceChildren(...rows.map(([id,name,note])=>{const m=r.records.find(x=>x.id===id);if(!m)throw Error('缺少配置 '+id);const tr=document.createElement('tr');if(id.includes('fp32-c128'))tr.className='preferred';[name,m.median_rtf.toFixed(3),m.median_first_pcm_seconds.toFixed(3)+' 秒',Math.round(m.peak_rss_mib)+' MiB',note].forEach(v=>tr.append(element('td',v)));return tr;}));
  $('#cosyModes').replaceChildren(...r.cosy_modes.map(m=>{const tr=document.createElement('tr');[m.id==='cosy-zero'?'zero_shot / 参考 token':'instruct2 / 指令',(100*m.writing_normalized_cer).toFixed(2)+'%',m.warm_median_rtf.toFixed(3),'调用条件已对齐；音色待试听'].forEach(v=>tr.append(element('td',v)));return tr;}));
  const samples=r.samples.map(s=>({...s}));
  // Existing GPT samples stay in their original catalog; never duplicate weights or references.
  try{const response=await fetch('../catalog.json');if(response.ok){const catalog=await response.json();for(const s of catalog.samples.filter(s=>s.engine==='gsv'&&['welcome','story','numbers'].includes(s.phrase)))samples.unshift({...s,variant:'gsv',name:'GPT‑SoVITS · 原混合版',listen:'../'+s.listen,raw:'../'+s.raw});}}catch{ /* edge evidence remains independently readable */ }
  $('#edgePhrase').addEventListener('change',()=>render(samples));render(samples);
}
start().catch(error=>{const p=element('p','读取失败：'+error.message+'。可刷新重试或下载报告。');p.setAttribute('role','alert');$('#edgePlayers').replaceChildren(p);});
