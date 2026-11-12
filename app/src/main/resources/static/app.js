'use strict';
const $ = id => document.getElementById(id);
const S = {token:localStorage.getItem('pw_token')||'', workspace:JSON.parse(localStorage.getItem('pw_workspace')||'null'),study:null,hits:[],saved:[],selected:new Set(),job:null,stream:null,tab:'results',history:[],busy:false,searching:false,deleting:false,deleteTarget:null};
const labels={completed:'已完成',running:'分析中',auditing:'核验中',queued:'排队中',failed:'失败',cancelled:'已取消',timed_out:'已超时',interrupted:'重启前未完成'};
function node(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function button(text,cls,fn){const e=node('button',cls,text);e.type='button';e.addEventListener('click',()=>Promise.resolve(fn()).catch(notice));return e;}
function headers(extra={}){return {'Authorization':`Bearer ${S.token}`,'X-Workspace-ID':S.workspace?.id||'','X-Workspace-Key':S.workspace?.key||'','Content-Type':'application/json',...extra};}
async function api(path,method='GET',body){const response=await fetch(`/api${path}`,{method,headers:headers(),body:body===undefined?undefined:JSON.stringify(body)});if(!response.ok){let error;try{error=await response.json();}catch{error={message:`请求失败 ${response.status}`};}throw Error(error.message||error.detail||`请求失败 ${response.status}`);}return response.json();}
function notice(error){$('notice').textContent=error instanceof Error?error.message:String(error);$('notice').classList.remove('hidden');}
function clearNotice(){$('notice').classList.add('hidden');}
function filterValues(){return {year_from:Number($('year-from').value)||0,year_to:Number($('year-to').value)||0,ipc_section:$('ipc').value,ipc_subclass:$('subclass').value.trim().toUpperCase(),city:$('city').value.trim(),patent_type:$('patent-type').value};}
function applyFilters(f={}){for(const [id,key] of Object.entries({'year-from':'year_from','year-to':'year_to',ipc:'ipc_section',subclass:'ipc_subclass',city:'city','patent-type':'patent_type'}))$(id).value=f[key]||'';}
function tab(name){S.tab=name;document.querySelectorAll('[data-tab]').forEach(e=>{e.classList.toggle('active',e.dataset.tab===name);e.setAttribute('aria-selected',e.dataset.tab===name);});document.querySelectorAll('.tab-panel').forEach(e=>e.classList.toggle('hidden',e.id!==name));if(name==='history')loadHistory().catch(notice);}
function empty(container,title,message){container.replaceChildren();const e=node('div','empty');e.append(node('div','empty-icon','⌕'),node('h3','',title),node('p','',message));container.append(e);}
async function connect(){
  if(!S.token){$('access-dialog').showModal();return;}
  if(!S.workspace){S.workspace=await api('/workspaces','POST',{title:'我的调研工作区'});localStorage.setItem('pw_workspace',JSON.stringify(S.workspace));}
  const info=await api('/info');$('connection').textContent=info.retrieval?.ready?'本地服务已连接':'检索服务准备中';$('scope-text').textContent=info.scope;
  $('access-dialog').close();await loadStudies();
  if(!info.retrieval?.ready)notice('检索服务尚未就绪。调研记录仍可查看，请稍后重试检索。');
}
function studyControls(enabled){for(const id of ['study-title','query','year-from','year-to','ipc','subclass','patent-type','city','follow-up','analysis-question','search','export-md','export-csv'])$(id).disabled=!enabled;}
function clearStudy(){
  S.study=null;S.hits=[];S.saved=[];S.history=[];S.selected.clear();localStorage.removeItem('pw_study');
  $('study-title').value='';$('study-title').placeholder='新建调研后开始';$('query').value='';applyFilters();$('follow-up').checked=false;
  $('saved-status').textContent='尚未创建调研';$('search-meta').textContent='筛选条件不会自动放宽';$('job-status').classList.add('hidden');$('cancel').classList.add('hidden');
  studyControls(false);renderResults();renderSaved();renderSelection();
  empty($('results'),'暂无调研','点击左侧“新建技术调研”开始，或创建新的调研主题。');
  empty($('analysis'),'暂无技术分析','新建调研并选择专利后，可生成技术对照。');empty($('history'),'暂无分析记录','新建调研后开始检索与分析。');tab('results');
}
function askDeleteStudy(study){
  if(S.busy||S.job||S.searching||S.deleting){notice('请先完成当前检索或停止分析，再删除调研。');return;}
  const title=document.querySelector(`.study-link[data-id="${study.id}"]`)?.textContent||study.title;
  S.deleteTarget={...study,title};$('delete-study-name').textContent=title;$('delete-study-error').textContent='';$('delete-study-dialog').showModal();
}
async function deleteStudy(){
  const target=S.deleteTarget;if(!target||S.deleting)return;
  S.deleting=true;$('confirm-delete-study').disabled=true;$('cancel-delete-study').disabled=true;$('delete-study-error').textContent='';
  try{
    await api(`/studies/${target.id}`,'DELETE');
    const preserveCurrent=S.study?.id!==target.id;
    $('delete-study-dialog').close();S.deleteTarget=null;clearNotice();await loadStudies(undefined,preserveCurrent);notice(`已删除调研「${target.title}」。`);
  }catch(e){if($('delete-study-dialog').open)$('delete-study-error').textContent=e.message;else notice(e);}
  finally{S.deleting=false;$('confirm-delete-study').disabled=false;$('cancel-delete-study').disabled=false;}
}
async function loadStudies(preferred,preserveCurrent=false){
  const studies=await api('/studies');$('studies').replaceChildren();
  if(!studies.length){$('studies').append(node('p','sidebar-empty','暂无调研，点击上方按钮新建。'));clearStudy();return;}
  const id=preferred||localStorage.getItem('pw_study')||studies[0].id;
  for(const study of studies){
    const row=node('div','study-row'),e=button(study.title,'study-link',()=>{if(!S.deleting)return openStudy(study.id);});row.dataset.id=study.id;row.classList.toggle('active',S.study?.id===study.id);e.dataset.id=study.id;e.title=study.title;
    const remove=button('','study-delete',()=>askDeleteStudy(study));remove.title='删除调研';remove.setAttribute('aria-label',`删除调研：${study.title}`);
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('aria-hidden','true');
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d','M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 10v7M14 10v7');svg.append(path);remove.append(svg);row.append(e,remove);$('studies').append(row);
  }
  if(preserveCurrent&&S.study&&studies.some(s=>s.id===S.study.id))return;
  await openStudy(studies.some(s=>s.id===id)?id:studies[0].id);
}
async function openStudy(id){
  if(S.job||S.searching){notice('请先完成当前检索或停止分析，再切换调研。');return;}
  S.study=await api(`/studies/${id}`);localStorage.setItem('pw_study',id);$('study-title').value=S.study.title;
  studyControls(true);document.querySelectorAll('.study-row').forEach(e=>e.classList.toggle('active',e.dataset.id===id));
  const state=S.study.state||{};S.hits=state.last_search?.hits||[];$('query').value=state.last_query||'';applyFilters(state.filters);
  S.saved=await api(`/studies/${id}/documents`);S.selected=new Set(S.saved.filter(d=>d.selected).map(d=>d.doc_id));
  $('follow-up').checked=false;$('job-status').classList.add('hidden');$('cancel').classList.add('hidden');$('search-meta').textContent='筛选条件不会自动放宽';$('saved-status').textContent='本地保存';
  renderResults();renderSaved();renderSelection();empty($('analysis'),'结论与证据会显示在这里','选择候选专利，生成技术对照或方案归纳。');await loadHistory();tab('results');
}
function getDocument(id){return S.hits.find(d=>d.doc_id===id)||S.saved.find(d=>d.doc_id===id)?.document;}
async function saveDocument(doc,selected,note){const existing=S.saved.find(d=>d.doc_id===doc.doc_id);await api(`/studies/${S.study.id}/documents`,'PUT',{doc_id:doc.doc_id,selected,note:note===undefined?(existing?.note||''):note});S.saved=await api(`/studies/${S.study.id}/documents`);$('saved-status').textContent='已保存至本机';renderSaved();}
async function selectDocument(doc,checked){
  if(checked&&S.selected.size>=6){renderResults();throw Error('每次最多选择 6 件专利；技术比较最多 4 件。');}
  if(checked)S.selected.add(doc.doc_id);else S.selected.delete(doc.doc_id);
  try{await saveDocument(doc,checked);}catch(e){if(checked)S.selected.delete(doc.doc_id);else S.selected.add(doc.doc_id);throw e;}
  finally{renderSelection();renderResults();}
}
function card(doc,savedEntry){
  const el=node('article','patent-card'),top=node('div','card-top'),check=node('input');check.type='checkbox';check.checked=S.selected.has(doc.doc_id);check.setAttribute('aria-label',`选择 ${doc.title}`);check.addEventListener('change',()=>selectDocument(doc,check.checked).catch(notice));
  const titleBox=node('div');titleBox.append(button(doc.title,'patent-title',()=>showDocument(doc)));
  const meta=node('div','card-meta');for(const t of [doc.doc_id,`${doc.year} 年`,doc.ipc_main||'IPC 缺失',doc.patent_type,doc.city].filter(Boolean))meta.append(node('span','tag',t));titleBox.append(meta);top.append(check,titleBox);el.append(top,node('p','card-abstract',doc.abstract.slice(0,190)+(doc.abstract.length>190?'…':'')));
  const bottom=node('div','card-bottom');bottom.append(node('span','',S.selected.has(doc.doc_id)?'已加入待分析清单':'勾选后保存到候选清单'),button('查看完整摘要与证据 →','link-button',()=>showDocument(doc)));el.append(bottom);
  if(savedEntry){const input=node('textarea','note-input');input.rows=2;input.value=savedEntry.note;input.placeholder='记录技术关注点、排除原因或后续核查事项';input.setAttribute('aria-label',`${doc.title} 的调研备注`);el.append(input);const actions=node('div','card-bottom');actions.append(button('保存备注','link-button',()=>saveDocument(doc,S.selected.has(doc.doc_id),input.value)),button('移出候选清单','link-button',async()=>{await api(`/studies/${S.study.id}/documents/${encodeURIComponent(doc.doc_id)}`,'DELETE');S.selected.delete(doc.doc_id);S.saved=await api(`/studies/${S.study.id}/documents`);renderSaved();renderSelection();renderResults();}));el.append(actions);}
  return el;
}
function renderResults(){
  $('result-count').textContent=S.hits.length;$('results').replaceChildren();if(!S.hits.length){empty($('results'),'从一个具体的技术问题开始','输入需求并设置条件，检索结果将保留完整摘要和原文证据。');return;}
  const groups=new Map();for(const h of S.hits){const g=h.similar_group||h.doc_id;if(!groups.has(g))groups.set(g,[]);groups.get(g).push(h);}
  for(const group of groups.values()){$('results').append(card(group[0]));if(group.length>1){const details=node('details','similar');details.append(node('summary','',`另有 ${group.length-1} 件摘要高度相似的独立申请`));for(const d of group.slice(1))details.append(card(d));$('results').append(details);}}
}
function renderSaved(){$('saved-count').textContent=S.saved.length;$('saved').replaceChildren();if(!S.saved.length){empty($('saved'),'尚未保存候选专利','勾选检索结果即可保存，并添加你的调研备注。');return;}for(const entry of S.saved)$('saved').append(card(entry.document,entry));}
function renderSelection(){
  $('selection-count').textContent=S.selected.size;$('selection').replaceChildren();
  if(!S.selected.size)$('selection').append(node('p','small muted','从检索结果中勾选需要分析的专利。'));
  for(const id of S.selected){const d=getDocument(id);if(!d)continue;const e=node('div','selection-item');e.append(node('span','',d.title),button('×','',()=>selectDocument(d,false)));$('selection').append(e);}
  $('compare').disabled=!S.study||S.busy||S.selected.size<2||S.selected.size>4;$('overview').disabled=!S.study||S.busy||!S.selected.size;
}
async function search(){
  if(!S.study)return;const query=$('query').value.trim();if(!query)throw Error('请输入具体的技术问题。');const filters=filterValues();if(filters.year_from&&filters.year_to&&filters.year_from>filters.year_to)throw Error('起始年份不得晚于结束年份。');clearNotice();
  if($('follow-up').checked){
    // Persist explicit filters before a model rewrite; the model cannot override them.
    await api(`/studies/${S.study.id}/filters`,'PUT',{filters});
    return startJob('follow_up',query);
  }
  S.searching=true;$('search').disabled=true;$('search-meta').textContent='正在检索并重排候选专利…';
  try{const found=await api(`/studies/${S.study.id}/search`,'POST',{query,filters});acceptSearch(found);}
  finally{S.searching=false;$('search').disabled=!S.study;}
}
function acceptSearch(found){S.hits=found.hits||[];renderResults();tab('results');$('search-meta').textContent=`${S.hits.length} 件候选 · ${found.group_count||0} 组 · ${(found.elapsed_ms/1000).toFixed(2)} 秒`;$('query').value=found.query;applyFilters(found.filters);if(found.no_results)notice(found.notice);$('saved-status').textContent='检索记录已保存';}
async function startJob(kind,question){
  if(S.job)throw Error('当前已有分析任务，请先完成或停止。');clearNotice();
  const created=await api(`/studies/${S.study.id}/jobs`,'POST',{kind,question:question||$('analysis-question').value.trim(),doc_ids:[...S.selected]});
  S.job=created.id;S.busy=true;renderSelection();$('search').disabled=true;$('job-status').classList.remove('hidden');$('job-status').textContent='任务已提交，等待处理…';$('cancel').classList.remove('hidden');
  S.stream=new AbortController();
  try{
    const response=await fetch(`/api/jobs/${S.job}/events`,{headers:headers(),signal:S.stream.signal});if(!response.ok)throw Error('无法订阅分析进度');
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true}).replace(/\r\n/g,'\n');let at;while((at=buffer.indexOf('\n\n'))>=0){const block=buffer.slice(0,at);buffer=buffer.slice(at+2);let event='',data='';for(const line of block.split('\n')){if(line.startsWith('event:'))event=line.slice(6).trim();if(line.startsWith('data:'))data+=line.slice(5).trim();}if(data)handleEvent(event,JSON.parse(data));}}
    const final=await api(`/jobs/${S.job}`);if(final.status==='completed')handleResult(final);else if(!['queued','running','auditing'].includes(final.status))notice(final.result?.message||labels[final.status]||final.status);
  }catch(e){if(e.name!=='AbortError'){notice(e);try{await api(`/jobs/${S.job}/cancel`,'POST',{});}catch{}}}
  finally{S.job=null;S.busy=false;S.stream=null;$('search').disabled=false;$('cancel').classList.add('hidden');renderSelection();await loadHistory();}
}
function handleEvent(type,data){if(type==='phase')$('job-status').textContent=data.label;if(type==='result')handleResult(data);}
function handleResult(job){$('job-status').textContent=labels[job.status]||job.status;if(job.status!=='completed'){notice(job.result?.message||labels[job.status]);return;}const result=job.result;if(result.kind==='follow_up'){acceptSearch(result.search);notice(`已改写为：${result.rewritten_query}`);}else{renderAnalysis(result);tab('analysis');}}
function renderAnalysis(result){
  const container=$('analysis');container.replaceChildren();const head=node('div','analysis-header');head.append(node('strong','',result.question),node('span','',`${result.claims?.length||0} 项附原文结论 · 原文定位与数值检查不等于语义支持判定`));container.append(head);
  const documents=result.documents||[],claims=result.claims||[];
  if(!claims.length)empty(container,'没有形成通过检查的技术结论','原始摘要仍可查看。请缩小问题范围，或选择更相关的候选专利。');
  else{
    const wrap=node('div','analysis-table-wrap'),table=node('table','analysis-table'),thead=node('thead'),tr=node('tr');tr.append(node('th','','比较维度'));for(const d of documents)tr.append(node('th','',d.title));thead.append(tr);table.append(thead);const tbody=node('tbody');
    const dimensions=[...new Set(claims.map(c=>c.dimension))];for(const dimension of dimensions){const row=node('tr');row.append(node('th','',dimension));for(const doc of documents){const cell=node('td'),matches=claims.filter(c=>c.dimension===dimension&&c.doc_id===doc.doc_id);if(!matches.length)cell.append(node('p','muted','未形成该维度的有效结论，需查阅摘要或后续资料。'));for(const c of matches){cell.append(node('p','',c.text));for(const evidence of c.evidence||[])cell.append(button('↗ 查看原文证据','evidence-link',()=>showDocument(doc,[evidence.id])));}row.append(cell);}tbody.append(row);}table.append(tbody);wrap.append(table);container.append(wrap);
  }
  if(result.review?.length){const details=node('details','review');details.append(node('summary','',`${result.review.length} 项待复核内容，未纳入正式结论`));for(const c of result.review){const p=node('p','',c.text);p.append(node('br'),node('small','',c.status==='numeric_review'?`数值或单位未找到：${c.missing_numbers.join('、')}`:'引用无效或对应专利不一致'));details.append(p);}container.append(details);}
  for(const line of result.limitations||[])container.append(node('p','small muted',line));
}
async function showDocument(doc,evidenceIds=[]){
  if(!doc.abstract){const found=await api(`/documents/${encodeURIComponent(doc.doc_id)}`);doc=found.documents[0];}
  const box=$('document-content');box.replaceChildren(node('h2','',doc.title),node('p','',`${doc.doc_id} · ${doc.year} · ${doc.ipc_main||'IPC 缺失'} · ${doc.patent_type}`));const original=node('div','original-abstract');
  const points=Array.from(doc.abstract);const highlights=(doc.evidence||[]).filter(e=>evidenceIds.includes(e.id)).sort((a,b)=>a.start-b.start);let start=0;for(const e of highlights){original.append(document.createTextNode(points.slice(start,e.start).join('')),node('mark','',points.slice(e.start,e.end).join('')));start=e.end;}original.append(document.createTextNode(points.slice(start).join('')));box.append(original);
  for(const e of (highlights.length?highlights:doc.evidence||[])){const quote=node('div','quote',e.text);quote.append(node('small','',`原文位置 ${e.start}–${e.end} · ${doc.doc_id}`));box.append(quote);}
  box.append(node('p','small muted','以上是当前本地语料中的摘要原文。完整说明书、权利要求和法律状态需要另行查阅。'));if(!$('document-dialog').open)$('document-dialog').showModal();
}
async function loadHistory(){if(!S.study)return;S.history=await api(`/studies/${S.study.id}/jobs`);$('history').replaceChildren();if(!S.history.length){empty($('history'),'尚无分析记录','完成的分析会保留问题、证据、模型版本与结果。');return;}for(const job of S.history){const e=node('article','history-item');e.append(node('h3','',job.result.question||'检索追问'),node('p','',`${labels[job.status]||job.status} · ${new Date(job.created).toLocaleString('zh-CN')}`));if(job.status==='completed')e.append(button('打开分析记录 →','link-button',()=>{handleResult(job);}));else if(job.result.message)e.append(node('p','',job.result.message));$('history').append(e);}}
async function download(format){if(!S.study)return;const r=await fetch(`/api/studies/${S.study.id}/export?format=${format}`,{headers:headers()});if(!r.ok)throw Error('导出失败');const url=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=url;a.download=`${S.study.title}.${format}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
$('access-form').addEventListener('submit',async event=>{event.preventDefault();S.token=$('access-token').value.trim();localStorage.setItem('pw_token',S.token);$('access-error').textContent='';try{await connect();}catch(e){$('access-error').textContent=e.message;}});
$('new-study').addEventListener('click',async()=>{try{if(S.job||S.searching||S.deleting)throw Error('请先完成当前操作或停止分析。');const study=await api('/studies','POST',{title:'新的技术调研'});await loadStudies(study.id);$('study-title').focus();$('study-title').select();}catch(e){notice(e);}});
$('study-title').addEventListener('change',async()=>{try{if(!S.study)return;const id=S.study.id,title=$('study-title').value;await api(`/studies/${id}`,'PATCH',{title});if(S.study?.id===id){S.study.title=title;$('saved-status').textContent='标题已保存';}const current=document.querySelector(`.study-link[data-id="${id}"]`);if(current){current.textContent=title;current.title=title;}const remove=document.querySelector(`.study-row[data-id="${id}"] .study-delete`);if(remove)remove.setAttribute('aria-label',`删除调研：${title}`);}catch(e){notice(e);}});
$('search').addEventListener('click',()=>search().catch(notice));$('query').addEventListener('keydown',e=>{if(e.key==='Enter'&&!S.busy)search().catch(notice);});
$('compare').addEventListener('click',()=>startJob('compare').catch(notice));$('overview').addEventListener('click',()=>startJob('overview').catch(notice));
$('cancel').addEventListener('click',async()=>{if(S.job){try{await api(`/jobs/${S.job}/cancel`,'POST',{});$('job-status').textContent='正在取消上游生成…';}catch(e){notice(e);}}});
$('export-md').addEventListener('click',()=>download('md').catch(notice));$('export-csv').addEventListener('click',()=>download('csv').catch(notice));
$('close-document').addEventListener('click',()=>$('document-dialog').close());document.querySelectorAll('[data-tab]').forEach(e=>e.addEventListener('click',()=>tab(e.dataset.tab)));
$('confirm-delete-study').addEventListener('click',()=>deleteStudy().catch(notice));
$('cancel-delete-study').addEventListener('click',()=>$('delete-study-dialog').close());
$('delete-study-dialog').addEventListener('cancel',event=>{if(S.deleting)event.preventDefault();});
$('delete-study-dialog').addEventListener('close',()=>{if(!S.deleting)S.deleteTarget=null;});
$('logout').addEventListener('click',()=>{if(S.job){notice('请先停止当前分析再锁定。');return;}S.token='';localStorage.removeItem('pw_token');$('access-token').value='';$('access-dialog').showModal();});
window.addEventListener('beforeunload',()=>{if(S.job)fetch(`/api/jobs/${S.job}/cancel`,{method:'POST',headers:headers(),body:'{}',keepalive:true}).catch(()=>{});});
const fragment=new URLSearchParams(location.hash.slice(1));if(fragment.has('token')){S.token=fragment.get('token');localStorage.setItem('pw_token',S.token);history.replaceState(null,'',location.pathname);}
connect().catch(e=>{$('access-error').textContent=e.message;if(!$('access-dialog').open)$('access-dialog').showModal();});
