/* ============ HELPERS ============ */
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const SEV={high:['High attention','▲'],medium:['Medium attention','◆'],low:['Low attention','●'],none:['No flags','✓']};
const sevBadge=s=>`<span class="badge b-${s==='none'?'ok':s}">${SEV[s][1]} ${SEV[s][0]}</span>`;
const confCol=c=>c>=70?'var(--ok)':c>=40?'var(--med)':'var(--high)';
const confBar=c=>`<div class="row2" style="gap:6px"><div class="bar" style="width:70px"><i style="width:${c}%;background:${confCol(c)}"></i></div><span style="font-size:12px">${c}%</span></div>`;
const STATUS={processing:['Processing','b-low'],ready:['Ready for Review','b-med'],inreview:['In Review','b-ai'],reviewed:['Reviewed','b-ok']};
const stBadge=s=>`<span class="badge ${STATUS[s][1].replace('b-med','b-medium')}">${STATUS[s][0]}</span>`;
const store={get(k,d){try{return JSON.parse(localStorage.getItem(k))??d}catch(e){return d}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
function toast(m){const t=document.createElement('div');t.className='toast';t.textContent=m;document.body.appendChild(t);setTimeout(()=>t.remove(),2600)}
function dl(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click()}

/* ============ MOCK DATA ============ */
const F=(label,val,conf,st,x,y,w,h)=>({label,val,conf,st,x,y,w,h});
let uid=0;
function mkDocs(defs){return defs.map((d,i)=>({id:'D'+(i+1),name:d[0],type:d[1],pages:1,title:d[2],fields:d[3].map(f=>({id:'f'+(++uid),orig:null,...f}))}))}
const RULES=[
 ['R-01','Amount consistency','Claim amount must match invoice total within 2%','|claim-invoice|/invoice ≤ 2%','high','Cross-document',12],
 ['R-02','Filing window','Claim must be filed within 30 days of incident','filing gap ≤ 30 days','high','Policy',9],
 ['R-03','Mandatory signature','Claim form must carry claimant signature','signature present','medium','Completeness',6],
 ['R-04','Unique invoice number','Invoice number must not repeat across claims','no duplicates','high','Duplicate',4],
 ['R-05','Hospital on network list','Hospital must be an empanelled provider','hospital ∈ network','low','Policy',3],
 ['R-06','GSTIN format','GSTIN must be 15 chars, valid pattern','regex match','low','Validation',5],
 ['R-07','Policy active on incident date','Policy must be in force on incident date','start ≤ incident ≤ end','high','Policy',2],
 ['R-08','Document metadata','File edit date must not post-date submission','edit ≤ submit','medium','Anomaly',7]];
const chk=(id,st,thr,act,ex)=>({id,st,thr,act,ex});
const P_OK=[chk('R-01','pass','≤ 2%','0%','Amounts match.'),chk('R-02','pass','≤ 30 days','6 days','Filed within the window.'),chk('R-03','pass','Present','Present','Signature found.'),chk('R-04','pass','Unique','Unique','No duplicate invoice.'),chk('R-05','pass','In network','In network','Hospital empanelled.'),chk('R-06','pass','Valid','Valid','GSTIN format valid.'),chk('R-07','pass','In force','In force','Policy active.'),chk('R-08','pass','edit ≤ submit','OK','Metadata consistent.')];
const withP=(o)=>P_OK.map(c=>o[c.id]||c);
const fnd=(id,cat,sev,title,why,conf,ev,doc,box)=>({id,cat,sev,title,why,conf,ev,doc,box,status:'open',comments:[]});
const CLAIMS=[
{id:'CLM-2041',name:'Rahul Sharma',pol:'POL-77821',status:'ready',att:'high',date:'2026-10-14',assigned:'Priya N.',
 docs:mkDocs([['claim_form.pdf','Claim Form','Health Insurance Claim Form',[F('Claimant Name','Rahul Sharma',97,'ok',8,16,40,7),F('Policy No.','POL-77821',96,'ok',52,16,38,7),F('Date of Incident','12 Aug 2026',91,'ok',8,28,40,7),F('Hospital','Apollo Hospitals, Delhi',88,'ok',52,28,40,7),F('Claim Amount','₹85,000',95,'mis',8,40,40,7),F('Diagnosis','Acute appendicitis',82,'ok',52,40,40,7),F('Signature','Present',90,'ok',8,86,30,7)]],
  ['invoice.pdf','Invoice','Apollo Hospitals — Tax Invoice',[F('Invoice No.','INV-2291',93,'ok',8,16,40,7),F('Invoice Date','20 Aug 2026',92,'ok',52,16,38,7),F('Hospital','Apollo Hospitals, Delhi',89,'ok',8,28,50,7),F('Invoice Total','₹58,000',94,'mis',8,40,40,7),F('GSTIN','07AABCA1234C1Z5',78,'rev2',52,40,40,7)]]]),
 findings:[fnd('F1','mismatch','high','Claim amount differs from invoice total','The amount on the claim form is higher than the invoice total. A transcription error or an additional bill is possible.',94,['Claim Form p1: ₹85,000','Invoice p1: ₹58,000','Difference: ₹27,000 (46.6%)'],'D1',[8,40,40,7]),
  fnd('F2','anomaly','low','GSTIN read with moderate confidence','OCR confidence is below 80%. Please verify against the original document.',78,['OCR confidence 78%'],'D2',[52,40,40,7])],
 mism:[{label:'Claim amount',a:['Claim Form','₹85,000','D1'],b:['Invoice #INV-2291','₹58,000','D2'],diff:'+₹27,000 (46.6%)',fid:'F1'}],
 policy:withP({'R-01':chk('R-01','fail','≤ 2%','46.6%','Claim amount exceeds invoice total by ₹27,000.')}),
 ai:null,summary:['Claim amount on the form is ₹27,000 higher than the invoice total.','One field was read with moderate OCR confidence (GSTIN).','All other policy checks passed.']},
{id:'CLM-2042',name:'Anita Verma',pol:'POL-55410',status:'inreview',att:'medium',date:'2026-10-12',assigned:'Priya N.',
 docs:mkDocs([['claim_form.pdf','Claim Form','Health Insurance Claim Form',[F('Claimant Name','Anita Verma',96,'ok',8,16,40,7),F('Policy No.','POL-55410',95,'ok',52,16,38,7),F('Date of Incident','02 Jul 2026',90,'rev2',8,28,40,7),F('Filing Date','16 Aug 2026',93,'rev2',52,28,40,7),F('Claim Amount','₹32,400',95,'ok',8,40,40,7),F('Signature','Present',88,'ok',8,86,30,7)]],
  ['bill.jpg','Bill','City Care Hospital — Bill',[F('Bill No.','CCH-8812',90,'ok',8,16,40,7),F('Bill Total','₹32,400',92,'ok',8,28,40,7)]]]),
 findings:[fnd('F1','policy','medium','Claim filed outside the 30-day window','Filing happened 45 days after the incident, while the policy window is 30 days. A late-intimation clause may apply.',91,['Incident: 02 Jul 2026','Filed: 16 Aug 2026','Gap: 45 days · Rule R-02'],'D1',[52,28,40,7])],
 mism:[],policy:withP({'R-02':chk('R-02','fail','≤ 30 days','45 days','Claim filed 45 days after the incident. Policy window: 30 days.')}),ai:null,
 summary:['Claim was filed 45 days after the incident (window: 30 days).','Amounts across documents match.']},
{id:'CLM-2043',name:'Mohit Agarwal',pol:'POL-90233',status:'ready',att:'high',date:'2026-10-15',assigned:'—',
 docs:mkDocs([['medical_report.docx','Medical Report','Medical Narrative Summary',[F('Patient','Mohit Agarwal',95,'ok',8,16,40,7),F('Diagnosis','Lumbar disc prolapse',84,'ok',52,16,40,7),F('Doctor','Dr. S. Kulkarni',80,'rev2',8,28,40,7)]],
  ['claim_form.pdf','Claim Form','Health Insurance Claim Form',[F('Claimant Name','Mohit Agarwal',97,'ok',8,16,40,7),F('Claim Amount','₹1,20,000',94,'ok',8,28,40,7),F('Signature','Present',89,'ok',8,86,30,7)]]]),
 findings:[fnd('F1','ai_content','high','Medical narrative shows high estimated AI-generated content','About 68% of the narrative shares statistical patterns commonly seen in machine-generated text. This is a probabilistic estimate and may be a false positive.',72,['Uniform sentence length','Low lexical variety','Generic clinical phrasing'],'D1',[8,45,84,32])],
 mism:[],policy:P_OK,
 ai:{overall:68,per:[['medical_report.docx',68],['claim_form.pdf',4]],passages:[{id:'P1',doc:'D1',text:'The patient presented with persistent symptoms and underwent a comprehensive evaluation. It is important to note that the findings are consistent with the diagnosis, and a structured plan of care was subsequently recommended.',conf:72,signals:['Uniform sentence structure','Low lexical variety','Generic phrasing']}]},
 summary:['The medical narrative has a high estimated AI-content share (68%) — probabilistic, needs human judgement.','Other checks passed.']},
{id:'CLM-2044',name:'Sunita Rao',pol:'POL-12987',status:'ready',att:'high',date:'2026-10-16',assigned:'—',
 docs:mkDocs([['claim_form.pdf','Claim Form','Health Insurance Claim Form',[F('Claimant Name','Sunita Rao',96,'ok',8,16,40,7),F('Claim Amount','₹44,900',93,'ok',8,28,40,7),F('Signature','Missing',60,'mss',8,86,30,7)]],
  ['invoice.pdf','Invoice','Sai Clinic — Invoice',[F('Invoice No.','INV-1007',92,'mis',8,16,40,7),F('Invoice Total','₹44,900',93,'ok',8,28,40,7)]]]),
 findings:[fnd('F1','anomaly','high','Invoice number also seen in another claim','INV-1007 appears on CLM-1873 submitted earlier. This could be a resubmission or a data-entry coincidence.',88,['INV-1007 · CLM-1873 (04 Sep 2026)'],'D2',[8,16,40,7]),
  fnd('F2','missing','medium','Signature not detected on claim form','No signature was found in the expected region.',76,['Rule R-03'],'D1',[8,86,30,7])],
 mism:[],policy:withP({'R-03':chk('R-03','fail','Present','Missing','No signature detected.'),'R-04':chk('R-04','fail','Unique','Duplicate','Invoice INV-1007 appears in CLM-1873.')}),ai:null,
 summary:['Invoice number matches one in an earlier claim.','Signature missing on the claim form.']},
{id:'CLM-2045',name:'Karan Mehta',pol:'POL-33001',status:'reviewed',att:'none',date:'2026-10-09',assigned:'Arjun S.',
 docs:mkDocs([['claim_form.pdf','Claim Form','Health Insurance Claim Form',[F('Claimant Name','Karan Mehta',98,'ok',8,16,40,7),F('Claim Amount','₹12,500',96,'ok',8,28,40,7),F('Signature','Present',92,'ok',8,86,30,7)]]]),
 findings:[],mism:[],policy:P_OK,ai:null,summary:['No inconsistencies detected. All policy checks passed.']},
{id:'CLM-2046',name:'Fatima Khan',pol:'POL-66120',status:'ready',att:'medium',date:'2026-10-17',assigned:'—',
 docs:mkDocs([['scan_invoice.jpg','Invoice','Low-quality scan',[F('Invoice No.','INV-33?9',41,'rev2',8,16,40,7),F('Invoice Total','₹ 2?,800',38,'rev2',8,28,40,7)]],
  ['claim_form.pdf','Claim Form','Health Insurance Claim Form',[F('Claimant Name','Fatima Khan',95,'ok',8,16,40,7),F('Claim Amount','₹21,800',94,'ok',8,28,40,7)]]]),
 findings:[fnd('F1','anomaly','medium','Low OCR confidence on scanned invoice','The scan quality is poor, so key values could not be read reliably. Please verify manually or request a clearer copy.',41,['OCR confidence 38–41%'],'D1',[8,16,40,14])],
 mism:[],policy:withP({'R-01':chk('R-01','na','≤ 2%','—','Invoice total unreadable.')}),ai:null,
 summary:['Invoice scan is hard to read; some values are uncertain.']},
{id:'CLM-2047',name:'Vikram Joshi',pol:'POL-70811',status:'processing',att:'none',date:'2026-10-19',assigned:'—',docs:[],findings:[],mism:[],policy:[],ai:null,summary:[]}
];
const getClaim=id=>CLAIMS.find(c=>c.id===id);
const att=c=>{const o=c.findings.filter(f=>f.status!=='dismissed');return o.some(f=>f.sev==='high')?'high':o.some(f=>f.sev==='medium')?'medium':o.length?'low':'none'};

/* ============ STATE ============ */
const S={theme:store.get('theme_v2','light'),user:store.get('user','Priya Nair'),menu:null,lastR:null,q:'',fAtt:'',fStat:'',R:{claim:null,doc:0,tab:'summary',layers:{fields:true,flags:true,ai:true},sel:null,zoom:1,sevF:'',stF:''},
 U:{files:[],scope:{ocr:true,xdoc:true,rules:true,ai:false},meta:{}},P:null,audit:{},notes:{}};
document.documentElement.dataset.theme=S.theme;

/* ============ CHARTS (SVG) ============ */
function line(arr){const w=420,h=140,m=Math.max(...arr),pts=arr.map((v,i)=>[i*(w/(arr.length-1)),h-8-(v/m)*(h-24)]);const d=pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(0)+','+p[1].toFixed(0)).join(' ');
 return `<svg viewBox="0 0 ${w} ${h}" width="100%"><path d="${d} L${w},${h} L0,${h}Z" fill="rgba(99,102,241,.2)"/><path d="${d}" stroke="var(--pri)" fill="none" stroke-width="2.5"/></svg>`}
function donut(items){const t=items.reduce((a,b)=>a+b[1],0)||1;let a0=-Math.PI/2;const r=52,cx=70,cy=70;
 const arcs=items.map(([n,v,c])=>{const a1=a0+v/t*2*Math.PI,x0=cx+r*Math.cos(a0),y0=cy+r*Math.sin(a0),x1=cx+r*Math.cos(a1),y1=cy+r*Math.sin(a1);const p=`<path d="M${x0},${y0} A${r},${r} 0 ${v/t>.5?1:0} 1 ${x1},${y1}" stroke="${c}" stroke-width="18" fill="none"/>`;a0=a1;return p}).join('');
 return `<div class="row2"><svg viewBox="0 0 140 140" width="140">${arcs}<text x="70" y="76" text-anchor="middle" fill="var(--tx)" font-size="20" font-weight="700">${t}</text></svg><div>${items.map(([n,v,c])=>`<div style="font-size:12px"><span style="color:${c}">■</span> ${n} · ${v}</div>`).join('')}</div></div>`}
function hbars(items,col){const m=Math.max(...items.map(i=>i[1]))||1;return items.map(([n,v])=>`<div style="margin:7px 0"><div class="row2" style="font-size:12px"><span>${n}</span><span class="sp"></span><b>${v}</b></div><div class="bar"><i style="width:${v/m*100}%;background:${col||'var(--pri)'}"></i></div></div>`).join('')}
function ring(p){const r=44,c=2*Math.PI*r;return `<svg viewBox="0 0 110 110" width="110"><circle cx="55" cy="55" r="${r}" stroke="var(--bd)" stroke-width="12" fill="none"/><circle cx="55" cy="55" r="${r}" stroke="var(--ai)" stroke-width="12" fill="none" stroke-dasharray="${c*p/100} ${c}" transform="rotate(-90 55 55)" stroke-linecap="round"/><text x="55" y="61" text-anchor="middle" fill="var(--tx)" font-size="22" font-weight="700">${p}%</text></svg>`}

/* ============ VIEWS ============ */
function dashboard(){
 const all=CLAIMS,flags=all.flatMap(c=>c.findings),needs=all.filter(c=>c.status==='ready'||c.status==='inreview');
 const IC={'Claims Processed':'📥','Awaiting Review':'⏳','Flagged Items':'🚩','Avg Processing Time':'⚡','Avg AI-Content Est.':'🤖','Mismatch Rate':'🔀'};
 const kpi=(l,v,t,up)=>`<div class="card kpi"><span class="ic">${IC[l]||''}</span><div class="l">${l}</div><div class="v">${v}</div><div style="font-size:12px;color:${up?'var(--ok)':'var(--high)'}">${t}</div></div>`;
 const cat=k=>flags.filter(f=>f.cat===k).length;
 const rows=all.filter(c=>(!S.fAtt||att(c)===S.fAtt)&&(!S.fStat||c.status===S.fStat)&&(!S.q||(c.id+c.name).toLowerCase().includes(S.q.toLowerCase())));
 const hr=new Date().getHours(),gr=hr<12?'Good morning':hr<17?'Good afternoon':'Good evening',hi=all.filter(c=>att(c)==='high'&&c.status!=='reviewed').length;
 return `<div class="card hero"><div style="flex:1;min-width:240px"><h1>${gr}, ${esc(S.user.split(' ')[0])} 👋</h1><p><b>${needs.length}</b> claims are waiting for your review${hi?` · <b>${hi}</b> need high attention`:''}.</p></div><button class="btn" data-act="go" data-to="claims">Open review queue</button><button class="btn g" data-act="go" data-to="upload">+ Upload claim</button></div>
 <div class="grid g6">${kpi('Claims Processed',all.length*41,'▲ 8% vs last week',1)}${kpi('Awaiting Review',needs.length,'▼ 2 vs yesterday',1)}${kpi('Flagged Items',flags.length*7,'▲ 5%',0)}${kpi('Avg Processing Time','42 s','▼ 6 s',1)}${kpi('Avg AI-Content Est.','11%','▲ 1%',0)}${kpi('Mismatch Rate','9.4%','▼ 0.8%',1)}</div>
 <div class="grid g2" style="margin-top:14px">
  <div class="card"><h2>Claims processed over time</h2>${line([22,30,28,41,38,52,47,60,58,71])}</div>
  <div class="card"><h2>Flags by category</h2>${donut([['AI-content',cat('ai_content')+3,'var(--ai)'],['Mismatch',cat('mismatch')+5,'var(--high)'],['Policy',cat('policy')+4,'var(--med)'],['Missing field',cat('missing')+2,'#94a3b8'],['Anomaly',cat('anomaly')+3,'var(--low)']])}</div>
  <div class="card"><h2>Attention-level distribution</h2>${hbars([['High',all.filter(c=>att(c)==='high').length],['Medium',all.filter(c=>att(c)==='medium').length],['Low',all.filter(c=>att(c)==='low').length],['None',all.filter(c=>att(c)==='none').length]],'var(--high)')}</div>
  <div class="card"><h2>Top triggered policy rules</h2>${hbars(RULES.slice().sort((a,b)=>b[6]-a[6]).slice(0,5).map(r=>[r[0]+' · '+r[1],r[6]]))}</div>
 </div>
 <div class="card" style="margin-top:14px"><div class="row2"><h2 style="margin:0">Recent claims</h2><span class="sp"></span>
  <select style="width:auto" data-chg="fAtt"><option value="">All attention</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option><option value="none">None</option></select>
  <select style="width:auto" data-chg="fStat"><option value="">All status</option>${Object.entries(STATUS).map(([k,v])=>`<option value="${k}">${v[0]}</option>`).join('')}</select></div>
  ${table(rows)}</div>`}
function table(rows){return `<table style="margin-top:10px"><tr><th>Claim</th><th>Claimant</th><th>Docs</th><th>Attention</th><th>Flags</th><th>Status</th><th>Date</th></tr>
 ${rows.map(c=>`<tr class="row a-${att(c)}" data-act="open" data-id="${c.id}"><td><b>${c.id}</b></td><td>${esc(c.name)}</td><td>${c.docs.length}</td><td>${sevBadge(att(c))}</td><td>${c.findings.length}</td><td>${stBadge(c.status)}</td><td>${c.date}</td></tr>`).join('')||'<tr><td colspan=7>No claims match. Try clearing filters.</td></tr>'}</table>`}
function queue(){const order={high:0,medium:1,low:2,none:3};const rows=CLAIMS.filter(c=>!S.q||(c.id+c.name).toLowerCase().includes(S.q.toLowerCase())).sort((a,b)=>order[att(a)]-order[att(b)]);
 return `<h1>Claims Queue</h1><p class="sub">Sorted by attention level. Attention is a prioritisation aid, not a verdict.</p><div class="card">${table(rows)}</div>`}

function upload(){const U=S.U;
 const sc=(k,l,tip)=>`<div class="row2" style="margin:8px 0"><button class="tgl ${U.scope[k]?'on':''}" data-act="scope" data-k="${k}" aria-label="${l}"></button><div><b>${l}</b><div style="font-size:12px;color:var(--mut)">${tip}</div></div></div>`;
 return `<h1>Upload Claim</h1><p class="sub">Add claim form, invoices, bills and supporting documents.</p>
 <div class="grid g2"><div>
  <div class="drop" id="drop" data-act="pick"><div style="font-size:34px">📄</div><b>Drag & drop files here, or click to browse</b><div class="sub" style="margin:4px 0 0">PDF, DOCX, JPG, PNG · max 25 MB each</div></div>
  <input type="file" id="fi" multiple accept=".pdf,.docx,.jpg,.jpeg,.png" hidden>
  ${U.err?`<div class="err">${U.err}</div>`:''}
  <div class="card" style="margin-top:12px"><h2>Files (${U.files.length})</h2>${U.files.map((f,i)=>`<div class="row2" style="padding:6px 0;border-bottom:1px solid var(--bd)"><span style="font-size:22px">${/pdf/.test(f.name)?'📕':/docx/.test(f.name)?'📘':'🖼️'}</span><div><b>${esc(f.name)}</b><div style="font-size:12px;color:var(--mut)">${(f.size/1048576).toFixed(2)} MB</div></div><span class="sp"></span>
   <select style="width:150px" data-chg="ftype" data-i="${i}">${['Auto-detect','Claim Form','Invoice','Bill','Medical Report','ID Proof','Other'].map(t=>`<option ${f.type===t?'selected':''}>${t}</option>`).join('')}</select><button class="btn sec sm" data-act="rm" data-i="${i}" aria-label="Remove">✕</button></div>`).join('')||'<div class="sub">No files yet.</div>'}</div></div>
 <div><div class="card"><h2>Claim details</h2>
  <label class="f">Claim ID (optional)<input type="text" id="m_id" value="${esc(U.meta.id||'')}"></label>
  <label class="f">Policy Number *<input type="text" id="m_pol" value="${esc(U.meta.pol||'')}"></label>
  <label class="f">Claimant Name *<input type="text" id="m_name" value="${esc(U.meta.name||'')}"></label>
  <label class="f">Date of Incident<input type="date" id="m_dt" value="${esc(U.meta.dt||'')}"></label></div>
 <div class="card" style="margin-top:12px"><h2>Analysis scope</h2>
  ${sc('ocr','Field extraction (OCR)','Reads names, dates, amounts and line items.')}${sc('xdoc','Cross-document consistency','Compares values across documents.')}${sc('rules','Policy rule validation','Checks extracted data against policy rules.')}${sc('ai','AI-generated content detection','Optional. Probabilistic estimate; false positives are possible.')}
  <button class="btn" style="margin-top:10px;width:100%" data-act="start" ${U.files.length?'':'disabled'}>Start analysis</button></div></div></div>`}

const STEPS=['Uploading','OCR & field extraction','Cross-document comparison','Policy validation','AI-content analysis','Generating report'];
function processing(){const p=S.P;if(!p)return `<h1>No analysis running</h1><a href="#/upload">Go to upload</a>`;
 const steps=STEPS.map((n,i)=>({n,skip:(i===4&&!p.scope.ai)||(i===2&&!p.scope.xdoc)||(i===3&&!p.scope.rules)}));
 const done=p.done;const pct=Math.round(done/steps.length*100);
 return `<h1>Analysing ${esc(p.id)}</h1><p class="sub">${p.files} document(s) · elapsed ${Math.round((Date.now()-p.t0)/1000)}s</p>
 <div class="card" style="max-width:720px"><div class="bar" style="height:10px"><i style="width:${pct}%;background:var(--pri)"></i></div>
 ${steps.map((s,i)=>{const st=s.skip?'skip':i<done?'done':i===done&&!p.finished?'run':'pend';return `<div class="step"><div class="dot ${st}">${st==='done'?'✓':st==='skip'?'–':i+1}</div><b>${s.n}</b><span class="sp"></span><span style="color:var(--mut);font-size:12px">${st==='done'?'Done':st==='run'?'Running…':st==='skip'?'Skipped (off)':'Pending'}</span></div>`}).join('')}
 ${p.finished?`<div class="note" style="margin-top:12px">Analysis complete. Opening review in a moment…</div><button class="btn" data-act="open" data-id="${p.id}">Open review</button>`:`<button class="btn sec" style="margin-top:12px" data-act="cancel">Cancel</button>`}</div>`}

function review(){const R=S.R,c=R.claim;if(!c)return '';
 if(c.status==='processing')return `<h1>${c.id}</h1><div class="card">This claim is still being processed. Check back shortly.</div>`;
 const doc=c.docs[R.doc]||c.docs[0];const a=att(c);const open=c.findings.filter(f=>f.status==='open').length,rev=c.findings.length-open;
 const flagsDoc=c.findings.filter(f=>f.doc===doc.id&&f.box&&f.cat!=='ai_content');
 const aiDoc=c.findings.filter(f=>f.doc===doc.id&&f.cat==='ai_content');
 const paper=`<div class="paper" style="transform:scale(${R.zoom});margin-bottom:${(R.zoom-1)*680}px;width:520px"><div class="hd">${esc(doc.title)}<small>${esc(doc.type)} · ${esc(doc.name)}</small></div>
  ${doc.fields.map(f=>`<div class="fld ${R.layers.fields?'show':''} ${f.st==='mis'?'mis':f.st==='rev2'?'rev2':f.st==='mss'?'mss':''} ${R.sel===f.id?'sel':''}" data-act="field" data-id="${f.id}" title="${esc(f.label)}: ${esc(f.val)} (${f.conf}%)" style="left:${f.x}%;top:${f.y}%;width:${f.w}%;height:${f.h}%"><small>${esc(f.label)}</small>${esc(f.corr||f.val)}</div>`).join('')}
  ${aiDoc.map(f=>`<div class="fld" style="left:8%;top:45%;width:84%;height:32%;font-size:10px;line-height:1.5;color:#334;cursor:default">${esc((c.ai.passages[0]||{}).text||'')}</div>`).join('')}
  ${R.layers.flags?flagsDoc.map(f=>`<div class="flag ${R.sel===f.id?'sel':''}" data-act="focusF" data-id="${f.id}" style="left:${f.box[0]-1}%;top:${f.box[1]-1}%;width:${f.box[2]+2}%;height:${f.box[3]+2}%"></div>`).join(''):''}
  ${R.layers.ai?aiDoc.map(f=>`<div class="flag ai ${R.sel===f.id?'sel':''}" data-act="focusF" data-id="${f.id}" style="left:${f.box[0]}%;top:${f.box[1]}%;width:${f.box[2]}%;height:${f.box[3]}%"></div>`).join(''):''}</div>`;
 const lay=(k,l)=>`<label style="font-size:12px;margin-right:10px"><input type="checkbox" data-act="layer" data-k="${k}" ${R.layers[k]?'checked':''}> ${l}</label>`;
 const left=`<div class="card"><div class="tabs">${c.docs.map((d,i)=>`<button class="tab ${i===R.doc?'on':''}" data-act="doc" data-i="${i}">${esc(d.name)} ${c.findings.filter(f=>f.doc===d.id).length?'· '+c.findings.filter(f=>f.doc===d.id).length+' ⚑':''}</button>`).join('')}</div>
  <div class="row2" style="margin-bottom:8px">${lay('fields','Extracted fields')}${lay('flags','Flagged regions')}${lay('ai','AI passages')}<span class="sp"></span><button class="btn sec sm" data-act="zoom" data-d="-0.1">−</button><span style="font-size:12px">${Math.round(R.zoom*100)}%</span><button class="btn sec sm" data-act="zoom" data-d="0.1">+</button></div>
  <div class="paperwrap">${paper}</div></div>`;
 const center=`<div class="card pane"><h2>Extracted information</h2>${c.docs.map(d=>`<h3 style="margin-top:12px;color:var(--mut)">${esc(d.name)}</h3>${d.fields.map(f=>`<div class="row2" style="padding:7px 0;border-bottom:1px solid var(--bd)" >
  <div style="width:42%;cursor:pointer" data-act="field" data-id="${f.id}" data-doc="${d.id}"><div style="font-size:11px;color:var(--mut)">${esc(f.label)}</div>${f.corr?`<s style="color:var(--mut)">${esc(f.val)}</s> <b>${esc(f.corr)}</b>`:`<b>${esc(f.val)}</b>`}${f.conf<70?' <span title="Low OCR confidence">⚠️</span>':''}</div>
  ${confBar(f.conf)}<span class="sp"></span><span class="badge ${f.corr?'b-ai':f.st==='ok'?'b-ok':f.st==='mis'?'b-high':f.st==='mss'?'b-mut':'b-medium'}">${f.corr?'Corrected':f.st==='ok'?'Verified':f.st==='mis'?'Mismatch':f.st==='mss'?'Missing':'Needs Review'}</span>
  <button class="btn sec sm" data-act="edit" data-id="${f.id}" aria-label="Edit">✎</button></div>`).join('')}`).join('')}</div>`;
 const tabs=[['summary','Summary'],['mismatch','Mismatches'],['policy','Policy'],['ai','AI content'],['anom','Anomalies'],['audit','Audit']];
 const right=`<div class="card pane"><div class="tabs">${tabs.map(t=>`<button class="tab ${R.tab===t[0]?'on':''}" data-act="rtab" data-t="${t[0]}">${t[1]}</button>`).join('')}</div>${rightBody(c,R)}</div>`;
 return `<a class="back" href="#/claims">← Claims queue</a><div class="row2" style="margin-top:6px"><div><h1>${c.id} · ${esc(c.name)}</h1><div class="sub" style="margin:0">Policy ${c.pol} · ${stBadge(c.status)} · Filed ${c.date}</div></div><span class="sp"></span>${sevBadge(a)}
  <button class="btn sec" data-act="dlmenu">⬇ Report</button><button class="btn" data-act="markrev">Mark as reviewed</button></div>
 <div class="note" style="margin-top:12px">This is a prioritisation aid, not a fraud verdict. Press <kbd>?</kbd> for keyboard shortcuts.<div class="row2" style="margin-top:6px"><b>${rev} of ${c.findings.length} findings reviewed</b><div class="bar prog sp" style="margin:0"><i style="width:${c.findings.length?rev/c.findings.length*100:100}%;background:var(--ok)"></i></div></div></div>
 <div class="rev">${left}${center}${right}</div>`}

function fcard(f,R){const sel=R.sel===f.id;return `<div class="fc ${sel?'sel':''} ${f.status==='dismissed'?'dis':''}" id="fc-${f.id}">
 <div class="row2"><b style="flex:1;cursor:pointer" data-act="focusF" data-id="${f.id}">${esc(f.title)}</b>${sevBadge(f.sev)}</div>
 <div class="row2" style="margin:6px 0"><span style="font-size:12px;color:var(--mut)">Confidence</span>${confBar(f.conf)}<span class="badge ${f.status==='confirmed'?'b-ok':f.status==='dismissed'?'b-mut':'b-low'}">${f.status}</span></div>
 <details ${f.status==='open'?'open':''}><summary style="cursor:pointer;font-size:12px">Why was this flagged?</summary><p style="font-size:13px;margin:6px 0">${esc(f.why)}</p>${f.ev.map(e=>`<div class="ev">${esc(e)}</div>`).join('')}</details>
 ${f.comments.map(m=>`<div class="ev">💬 ${esc(m)}</div>`).join('')}
 <div class="row2" style="margin-top:8px"><button class="btn ok sm" data-act="conf" data-id="${f.id}">Confirm issue</button><button class="btn sec sm" data-act="dis" data-id="${f.id}">Dismiss as false positive</button><button class="btn sec sm" data-act="cmt" data-id="${f.id}">Comment</button>${f.status!=='open'?`<button class="btn sec sm" data-act="reopen" data-id="${f.id}">Reopen</button>`:''}</div></div>`}
function rightBody(c,R){const fl=c.findings;
 if(R.tab==='summary'){const by=s=>fl.filter(f=>f.sev===s).length;return `<h3>Why this claim needs review</h3>${c.summary.map(s=>`<div class="ev" style="font-size:13px;color:var(--tx)">${esc(s)}</div>`).join('')}
  <div class="row2" style="margin:12px 0"><span class="badge b-high">High ${by('high')}</span><span class="badge b-medium">Medium ${by('medium')}</span><span class="badge b-low">Low ${by('low')}</span><span class="badge b-mut">Dismissed ${fl.filter(f=>f.status==='dismissed').length}</span></div>
  <h3>All findings</h3>${fl.map(f=>fcard(f,R)).join('')||'<div class="sub">No findings. 🎉</div>'}<div class="warn" style="font-size:11px">Confidence scales: 0–40 low, 40–70 moderate, 70–100 high. OCR confidence (reading accuracy) is different from finding confidence (strength of the signal).</div>`}
 if(R.tab==='mismatch')return c.mism.map(m=>{const f=fl.find(x=>x.id===m.fid);return `<div class="fc"><b>${m.label}</b><div class="cmp"><div><small>${m.a[0]}</small><b>${m.a[1]}</b></div><div><small>${m.b[0]}</small><b>${m.b[1]}</b></div></div><div class="badge b-high">Difference ${m.diff}</div> <a href="#" data-act="focusF" data-id="${m.fid}">Go to location</a>${f?fcard(f,R):''}</div>`}).join('')||'<div class="sub">No cross-document mismatches found.</div>';
 if(R.tab==='policy')return c.policy.map(p=>`<div class="fc"><div class="row2"><b>${p.id} · ${RULES.find(r=>r[0]===p.id)[1]}</b><span class="sp"></span><span class="badge ${p.st==='pass'?'b-ok':p.st==='fail'?'b-high':'b-mut'}">${p.st==='pass'?'✓ Pass':p.st==='fail'?'✕ Fail':'– N/A'}</span></div><div style="font-size:12px;color:var(--mut)">Threshold ${p.thr} · Actual <b style="color:var(--tx)">${p.act}</b></div><div style="font-size:13px;margin-top:4px">${esc(p.ex)}</div></div>`).join('');
 if(R.tab==='ai'){if(!c.ai)return `<div class="note">AI-generated content detection was not run or found nothing notable for this claim. It can be enabled on the Upload page.</div>`;
  return `<div class="row2">${ring(c.ai.overall)}<div><b>Estimated AI-generated content</b><div style="font-size:12px;color:var(--mut)">Overall estimate</div></div></div>
  <div class="warn">Probabilistic estimate. Detection tools can produce false positives, especially for templated text or non-native English. Use as one signal among many.</div>
  ${hbars(c.ai.per)}<h3 style="margin-top:12px">Suspicious passages</h3>${c.ai.passages.map(p=>`<div class="fc"><p style="font-size:13px"><span class="hl">${esc(p.text)}</span></p>${confBar(p.conf)}<div class="row2" style="margin-top:6px">${p.signals.map(s=>`<span class="badge b-ai">${s}</span>`).join('')}</div></div>`).join('')}${fl.filter(f=>f.cat==='ai_content').map(f=>fcard(f,R)).join('')}`}
 if(R.tab==='anom')return fl.filter(f=>f.cat==='anomaly'||f.cat==='missing').map(f=>fcard(f,R)).join('')||'<div class="sub">No other anomalies.</div>';
 const log=S.audit[c.id]||[];return `<h3>Add note</h3><textarea id="note" rows="2" placeholder="Reviewer note…"></textarea><button class="btn sm" style="margin:6px 0" data-act="addnote">Save note</button>${log.slice().reverse().map(l=>`<div class="ev">${l.t} · ${esc(l.a)}</div>`).join('')||'<div class="sub">No actions yet.</div>'}`}

function reports(){return `<h1>Reports</h1><p class="sub">Download or preview analysis reports.</p><div class="card"><table><tr><th>Claim</th><th>Claimant</th><th>Generated</th><th>Actions</th></tr>
 ${CLAIMS.filter(c=>c.status!=='processing').map(c=>`<tr><td><b>${c.id}</b></td><td>${esc(c.name)}</td><td>${c.date}</td><td class="row2"><button class="btn sec sm" data-act="prev" data-id="${c.id}">Preview</button><button class="btn sec sm" data-act="dl" data-f="pdf" data-id="${c.id}">PDF</button><button class="btn sec sm" data-act="dl" data-f="json" data-id="${c.id}">JSON</button><button class="btn sec sm" data-act="dl" data-f="csv" data-id="${c.id}">CSV</button></td></tr>`).join('')}</table></div>`}
function reportHTML(c){return `<h1>ClaimLens Report — ${c.id}</h1><p class="sub">${esc(c.name)} · Policy ${c.pol} · Generated ${new Date().toLocaleString()}</p>
 <p><b>Attention level:</b> ${SEV[att(c)][0]} (prioritisation aid, not a verdict)</p><h2>Summary</h2><ul>${c.summary.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>
 <h2>Extracted data</h2><table><tr><th>Document</th><th>Field</th><th>Value</th><th>OCR conf.</th></tr>${c.docs.flatMap(d=>d.fields.map(f=>`<tr><td>${esc(d.name)}</td><td>${esc(f.label)}</td><td>${esc(f.corr||f.val)}</td><td>${f.conf}%</td></tr>`)).join('')}</table>
 <h2>Policy checks</h2><table>${c.policy.map(p=>`<tr><td>${p.id}</td><td>${p.st.toUpperCase()}</td><td>${esc(p.ex)}</td></tr>`).join('')}</table>
 <h2>Flagged items</h2>${c.findings.map(f=>`<p><b>${esc(f.title)}</b> — ${SEV[f.sev][0]}, confidence ${f.conf}%, status ${f.status}<br>${esc(f.why)}<br><i>${f.ev.map(esc).join('; ')}</i></p>`).join('')||'<p>No findings.</p>'}
 <h2>Methodology & limitations</h2><p>Automated OCR, rule checks and statistical text analysis. AI-content estimates are probabilistic and may produce false positives. Flags are for human review only.</p><p><b>AI-assisted analysis. The final decision rests with a human reviewer.</b></p>`}
function rules(){return `<h1>Policy Rules</h1><p class="sub">Predefined rules used for validation.</p><div class="grid g2"><div class="card"><table><tr><th>ID</th><th>Rule</th><th>Condition</th><th>Severity</th><th>On</th></tr>${RULES.map(r=>`<tr><td><b>${r[0]}</b></td><td>${r[1]}<div style="font-size:12px;color:var(--mut)">${r[2]}</div></td><td style="font-size:12px">${r[3]}</td><td>${sevBadge(r[4])}</td><td><button class="tgl on" onclick="this.classList.toggle('on')" aria-label="Toggle"></button></td></tr>`).join('')}</table></div>
 <div class="card"><h2>Test rule R-01 (amount consistency)</h2><label class="f">Claim amount (₹)<input type="number" id="t1" value="85000"></label><label class="f">Invoice total (₹)<input type="number" id="t2" value="58000"></label><button class="btn" data-act="testrule">Run test</button><div id="tres" style="margin-top:10px"></div></div></div>`}
function evaluation(){const m=[['Precision','91%'],['Recall','87%'],['F1','89%'],['OCR field accuracy','94%'],['Mismatch detection','96%'],['Avg processing','42 s']];
 return `<h1>Evaluation</h1><p class="sub">Results on sample documents (demo data).</p><div class="grid g6">${m.map(x=>`<div class="card kpi"><div class="l">${x[0]}</div><div class="v">${x[1]}</div></div>`).join('')}</div>
 <div class="grid g2" style="margin-top:14px"><div class="card"><h2>Confusion matrix</h2><div class="hm"><div style="background:rgba(34,197,94,.35)">TP 42</div><div style="background:rgba(249,115,22,.3)">FP 4</div><div style="background:rgba(249,115,22,.45)">FN 6</div><div style="background:rgba(34,197,94,.5)">TN 48</div></div></div>
 <div class="card"><h2>Accuracy by category</h2>${hbars([['Mismatch',96],['Policy',93],['Missing field',90],['Anomaly',82],['AI-content',74]],'var(--ok)')}</div></div>
 <div class="card" style="margin-top:14px"><h2>Sample test cases</h2><table><tr><th>Set</th><th>Expected</th><th>Detected</th><th>Result</th></tr>
 ${[['SET-01','Amount mismatch','Amount mismatch','Match'],['SET-02','Late filing','Late filing','Match'],['SET-03','Duplicate invoice','Duplicate invoice','Match'],['SET-04','AI narrative','AI narrative','Match'],['SET-05','None','Low OCR warning','False positive'],['SET-06','Missing signature','—','Miss']].map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td><span class="badge ${r[3]==='Match'?'b-ok':'b-medium'}">${r[3]}</span></td></tr>`).join('')}</table></div>`}
function settings(){return `<h1>Settings</h1><div class="card" style="max-width:560px"><div class="row2"><b>Dark theme</b><span class="sp"></span><button class="tgl ${S.theme==='dark'?'on':''}" data-act="theme" aria-label="Theme"></button></div><hr style="border:0;border-top:1px solid var(--bd)"><label class="f">Reviewer name<input type="text" value="${esc(S.user)}" onchange="S.user=this.value.trim()||S.user;store.set('user',S.user);render()"></label>
 <label class="f">High-attention confidence threshold: <b id="thv">${store.get('th',70)}%</b><input type="range" min="40" max="95" value="${store.get('th',70)}" oninput="store.set('th',+this.value);$('#thv').textContent=this.value+'%'"></label>
 <div class="sub">Data retention: uploaded documents are kept only for the demo session in this prototype.</div></div>`}

/* ============ SHELL & ROUTER ============ */
const ini=()=>S.user.split(/\s+/).map(w=>w[0]).slice(0,2).join('').toUpperCase();
const needsList=()=>CLAIMS.filter(c=>c.status==='ready'&&['high','medium'].includes(att(c)));
function topRight(){const n=needsList();return `<span class="sp"></span>
<div class="pop"><button class="icon" data-act="menu" data-m="bell" aria-label="Notifications">🔔${n.length?`<i class="cnt">${n.length}</i>`:''}</button>${S.menu==='bell'?`<div class="menu" style="width:310px"><h3 style="padding:6px 10px 2px">Needs your attention</h3>${n.slice(0,5).map(c=>`<div class="mi" data-act="open" data-id="${c.id}"><b>${c.id}</b> · ${esc(c.name)}<div style="margin-top:3px">${sevBadge(att(c))}</div></div>`).join('')||'<div class="mi">All caught up 🎉</div>'}</div>`:''}</div>
<button class="icon" data-act="theme" aria-label="Toggle theme" title="Toggle theme">${S.theme==='dark'?'☀️':'🌙'}</button>
<div class="pop"><button class="usr" data-act="menu" data-m="user" aria-label="Account menu"><span class="av">${esc(ini())}</span><span class="un"><b>${esc(S.user)}</b><small>Claims Reviewer</small></span><span style="color:var(--mut)">▾</span></button>${S.menu==='user'?`<div class="menu"><div class="mi" data-act="rename">✎ Change reviewer name</div><div class="mi" data-act="go" data-to="settings">⚙️ Settings</div><div class="mi" data-act="help">⌨️ Keyboard shortcuts</div></div>`:''}</div>`}
const NAV=[['dashboard','🏠','Dashboard'],['upload','⬆️','Upload Claim'],['claims','📋','Claims Queue'],['reports','📑','Reports'],['rules','⚖️','Policy Rules'],['eval','📊','Evaluation'],['settings','⚙️','Settings']];
function render(){const h=(location.hash||'#/dashboard').slice(2).split('/');const r=h[0]||'dashboard';
 if(r==='claim'){S.R.claim=getClaim(h[1]);}
 const body={dashboard,upload,processing,claims:queue,claim:review,reports,rules,eval:evaluation,settings}[r]||dashboard;
 const keep=document.activeElement&&document.activeElement.id;const scroll=window.scrollY;const ps=[...document.querySelectorAll('.pane,.paperwrap')].map(e=>e.scrollTop);
 $('#app').innerHTML=`<div class="shell"><aside class="side"><div class="logo"><i></i>ClaimLens</div>${NAV.map(n=>`<button class="nav ${r===n[0]||(r==='claim'&&n[0]==='claims')?'on':''}" data-act="go" data-to="${n[0]}">${n[1]} ${n[2]}</button>`).join('')}</aside>
 <div class="main"><div class="top"><input id="gs" placeholder="Search claim ID or claimant…  (Ctrl K)" value="${esc(S.q)}">${topRight()}</div>
 <div class="page">${body()}</div><div class="foot">AI-assisted analysis. Flags are prioritisation aids, not verdicts. The final decision rests with a human reviewer.</div></div></div>`;
 if(keep==='gs'){const g=$('#gs');g.focus();g.setSelectionRange(g.value.length,g.value.length)}
 if(S.lastR===r){window.scrollTo(0,scroll);document.querySelectorAll('.pane,.paperwrap').forEach((e,i)=>e.scrollTop=ps[i]||0)}else window.scrollTo(0,0);S.lastR=r;
 bindUpload()}
window.addEventListener('hashchange',()=>{S.R.sel=null;render()});

function bindUpload(){const d=$('#drop'),fi=$('#fi');if(!d)return;
 const add=list=>{const ok=['pdf','docx','jpg','jpeg','png'];S.U.err='';[...list].forEach(f=>{const e=f.name.split('.').pop().toLowerCase();if(!ok.includes(e))S.U.err=`"${esc(f.name)}" is not supported. Use PDF, DOCX, JPG or PNG.`;else if(f.size>26214400)S.U.err=`"${esc(f.name)}" exceeds 25 MB.`;else S.U.files.push({name:f.name,size:f.size,type:'Auto-detect'})});saveMeta();render()};
 d.ondragover=e=>{e.preventDefault();d.classList.add('over')};d.ondragleave=()=>d.classList.remove('over');d.ondrop=e=>{e.preventDefault();add(e.dataTransfer.files)};fi.onchange=()=>add(fi.files)}
function saveMeta(){const g=i=>($('#'+i)||{}).value;if($('#m_pol'))S.U.meta={id:g('m_id'),pol:g('m_pol'),name:g('m_name'),dt:g('m_dt')}}

/* ============ ACTIONS ============ */
const log=(c,a)=>{(S.audit[c.id]=S.audit[c.id]||[]).push({t:new Date().toLocaleTimeString(),a})};
const curF=()=>{const c=S.R.claim;return c.findings.find(f=>f.id===S.R.sel)};
function focusFinding(id){const c=S.R.claim,f=c.findings.find(x=>x.id===id);if(!f)return;S.R.sel=id;const di=c.docs.findIndex(d=>d.id===f.doc);if(di>=0)S.R.doc=di;if(f.cat==='ai_content')S.R.tab='ai';else if(S.R.tab==='audit')S.R.tab='summary';render();const el=$('#fc-'+id);if(el)el.scrollIntoView({block:'center'})}
const ACT={
 menu:d=>{S.menu=S._prev===d.m?null:d.m;render()},
 rename:()=>{const v=prompt('Reviewer name',S.user);if(v&&v.trim()){S.user=v.trim();store.set('user',S.user);toast('Name updated')}render()},
 help:()=>{const m=document.createElement('div');m.className='modal';m.innerHTML=`<div style="max-width:460px"><h2>Keyboard shortcuts</h2><table>${[['Ctrl / ⌘ + K','Focus search'],['J / K','Next / previous finding'],['C','Confirm selected finding'],['D','Dismiss as false positive'],['?','Show this help'],['Esc','Close dialogs & menus']].map(r=>`<tr><td><kbd>${r[0]}</kbd></td><td>${r[1]}</td></tr>`).join('')}</table><button class="btn" style="margin-top:12px" onclick="this.closest('.modal').remove()">Got it</button></div>`;document.body.appendChild(m)},
 go:d=>{location.hash='#/'+d.to},theme:()=>{S.theme=S.theme==='dark'?'light':'dark';store.set('theme_v2',S.theme);document.documentElement.dataset.theme=S.theme;render()},
 open:d=>{location.hash='#/claim/'+d.id;S.R.doc=0;S.R.tab='summary'},
 pick:()=>$('#fi').click(),scope:d=>{S.U.scope[d.k]=!S.U.scope[d.k];saveMeta();render()},rm:d=>{S.U.files.splice(+d.i,1);saveMeta();render()},
 start:()=>{saveMeta();const m=S.U.meta;if(!m.pol||!m.name){toast('Please fill Policy Number and Claimant Name');return}
  const id=m.id||'CLM-'+(2048+Math.floor(Math.random()*50));S.P={id,files:S.U.files.length,scope:{...S.U.scope},done:0,t0:Date.now(),finished:false};location.hash='#/processing';
  S.P.timer=setInterval(()=>{const p=S.P;if(!p)return;p.done++;if(p.done>=STEPS.length){p.finished=true;clearInterval(p.timer);
   const base=JSON.parse(JSON.stringify(CLAIMS[0]));base.id=id;base.name=m.name;base.pol=m.pol;base.date=new Date().toISOString().slice(0,10);base.status='ready';CLAIMS.unshift(base);toast('Analysis complete');
   p.redir=setTimeout(()=>{if(S.P&&S.P.finished&&location.hash==='#/processing')ACT.open({id})},3000)}
   if(location.hash==='#/processing')render()},1100)},
 cancel:()=>{clearInterval(S.P.timer);S.P=null;toast('Analysis cancelled');location.hash='#/upload'},
 doc:d=>{S.R.doc=+d.i;render()},layer:(d,t)=>{S.R.layers[d.k]=t.checked;render()},zoom:d=>{S.R.zoom=Math.min(2,Math.max(.5,S.R.zoom+ +d.d));render()},
 rtab:d=>{S.R.tab=d.t;render()},focusF:(d,t,e)=>{e.preventDefault();focusFinding(d.id)},
 field:d=>{const c=S.R.claim;S.R.sel=d.id;const di=c.docs.findIndex(x=>x.fields.some(f=>f.id===d.id));if(di>=0)S.R.doc=di;render()},
 edit:d=>{const f=S.R.claim.docs.flatMap(x=>x.fields).find(x=>x.id===d.id);const v=prompt('Correct value for '+f.label,f.corr||f.val);if(v&&v!==f.val){f.corr=v;log(S.R.claim,`Corrected "${f.label}": ${f.val} → ${v}`);toast('Correction saved');render()}},
 conf:d=>{const f=S.R.claim.findings.find(x=>x.id===d.id);f.status='confirmed';log(S.R.claim,'Confirmed: '+f.title);toast('Issue confirmed');render()},
 dis:d=>{const f=S.R.claim.findings.find(x=>x.id===d.id);f.status='dismissed';log(S.R.claim,'Dismissed as false positive: '+f.title);toast('Dismissed as false positive');render()},
 reopen:d=>{S.R.claim.findings.find(x=>x.id===d.id).status='open';render()},
 cmt:d=>{const t=prompt('Add comment');if(t){const f=S.R.claim.findings.find(x=>x.id===d.id);f.comments.push(t);log(S.R.claim,'Comment: '+t);render()}},
 addnote:()=>{const v=$('#note').value.trim();if(v){log(S.R.claim,'Note: '+v);toast('Note saved');render()}},
 markrev:()=>{const c=S.R.claim,o=c.findings.filter(f=>f.status==='open'),h=o.filter(f=>f.sev==='high').length;if(confirm(`${o.length} open finding(s)${h?`, including ${h} high-attention`:''}. Mark as reviewed?`)){c.status='reviewed';log(c,'Marked as reviewed');toast('Marked as reviewed');render()}},
 dlmenu:()=>{const f=prompt('Download format: pdf, json or csv','pdf');if(f)ACT.dl({f:f.toLowerCase(),id:S.R.claim.id})},
 dl:d=>{const c=getClaim(d.id);if(d.f==='json'){dl(c.id+'-report.json',JSON.stringify(c,null,2),'application/json');toast('JSON downloaded')}
  else if(d.f==='csv'){const rows=[['Finding','Severity','Confidence','Status','Explanation'],...c.findings.map(f=>[f.title,f.sev,f.conf,f.status,f.why])];dl(c.id+'-report.csv',rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(',')).join('\n'),'text/csv');toast('CSV downloaded')}
  else ACT.prev(d)},
 prev:d=>{const c=getClaim(d.id),m=document.createElement('div');m.className='modal';m.innerHTML=`<div>${reportHTML(c)}<div class="row2 noprint"><button class="btn" onclick="window.print()">Print / Save as PDF</button><button class="btn sec" onclick="this.closest('.modal').remove()">Close</button></div></div>`;document.body.appendChild(m)},
 testrule:()=>{const a=+$('#t1').value,b=+$('#t2').value,p=Math.abs(a-b)/b*100;$('#tres').innerHTML=p<=2?`<span class="badge b-ok">✓ Pass</span> Difference ${p.toFixed(1)}%`:`<span class="badge b-high">✕ Fail</span> Difference ${p.toFixed(1)}% exceeds 2% — flagged for review`}
};
document.addEventListener('click',e=>{const t=e.target.closest('[data-act]');S._prev=S.menu;
 if(S.menu&&!(t&&t.dataset.act==='menu'))S.menu=null;
 if(t&&ACT[t.dataset.act])ACT[t.dataset.act](t.dataset,t,e);
 if(S._prev&&!(t&&t.dataset.act==='menu'))render()});
document.addEventListener('change',e=>{const t=e.target;if(t.dataset.chg==='fAtt'||t.dataset.chg==='fStat'){S[t.dataset.chg]=t.value;render();const s=$(`[data-chg=${t.dataset.chg}]`);if(s)s.value=t.value}
 if(t.dataset.chg==='ftype')S.U.files[+t.dataset.i].type=t.value});
document.addEventListener('input',e=>{if(e.target.id==='gs'){S.q=e.target.value;if(!/claims|dashboard/.test(location.hash))location.hash='#/claims';else render()}});
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();$('#gs').focus();return}
 if(e.key==='Escape'){document.querySelectorAll('.modal').forEach(m=>m.remove());if(S.menu){S.menu=null;render()}return}
 if(/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;if(e.key==='?'){ACT.help();return}
 if(!location.hash.startsWith('#/claim/'))return;const c=S.R.claim;if(!c)return;const k=e.key.toLowerCase();
 const ids=c.findings.map(f=>f.id),i=ids.indexOf(S.R.sel);
 if(k==='j')focusFinding(ids[Math.min(ids.length-1,i+1)]);else if(k==='k')focusFinding(ids[Math.max(0,i-1)]);
 else if(k==='c'&&S.R.sel)ACT.conf({id:S.R.sel});else if(k==='d'&&S.R.sel)ACT.dis({id:S.R.sel})});
render();
