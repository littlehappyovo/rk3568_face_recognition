import { Records } from './records.js';
import { Stations } from './stations.js';
import { People } from './people.js';
export const $ = selector => document.querySelector(selector);
let csrf='', expires=0, refreshAt=0, checkRunning=false;
export function notice(message='',error=false) { $('#notice').textContent=message; $('#notice').classList.toggle('error',error); }
export function node(tag,text='',className='') { const element=document.createElement(tag);element.textContent=text;if(className)element.className=className;return element; }
export function localTime(value) { return value ? new Date(value).toLocaleString('zh-CN',{hour12:false}) : '—'; }
export async function api(path,options={}) {
    const headers=new Headers(options.headers||{});
    if(csrf) headers.set('X-CSRF-Token',csrf);
    let body=options.body;
    if(body && !(body instanceof FormData)) { body=JSON.stringify(body);headers.set('Content-Type','application/json'); }
    let response;
    try { response=await fetch(path,{...options,body,headers,credentials:'same-origin',cache:'no-store'}); }
    catch { throw new Error('连接中断，请检查板卡和网络。'); }
    let data; try { data=await response.json(); } catch { throw new Error('管理服务响应无效。'); }
    if(!response.ok) {
        if(response.status===401) lock();
        throw new Error(data.error?.message||`请求失败（${response.status}）`);
    } return data;
}
export async function run(action) { try { await action(); } catch(error) { notice(error.message,true); } }
function lock() { csrf='';expires=0;$('#workspace').hidden=true;$('#login-panel').hidden=false;$('#logout').hidden=true;document.querySelectorAll('dialog[open]').forEach(dialog=>dialog.close());document.dispatchEvent(new Event('session-ended')); }
const records=new Records(), stations=new Stations(), people=new People();
async function choosePanel(kind) {
    notice();document.querySelectorAll('nav button').forEach(button=>button.classList.toggle('selected',button.dataset.panel===kind));
    $('#records-panel').hidden=['stations','people'].includes(kind);$('#stations-panel').hidden=kind!=='stations';$('#people-panel').hidden=kind!=='people';
    if(kind==='stations') await stations.load(); else if(kind==='people') await people.load(); else await records.select(kind);
}
document.querySelectorAll('nav button').forEach(button=>button.addEventListener('click',()=>run(()=>choosePanel(button.dataset.panel))));
$('#login-form').addEventListener('submit',event=>{event.preventDefault();run(async()=>{
    const button=event.submitter;button.disabled=true;
    try { const data=await api('/api/auth/login',{method:'POST',body:{password:$('#password').value}});csrf=data.csrf_token;expires=data.expires_at*1000;refreshAt=Date.now();
        $('#password').value='';$('#login-panel').hidden=true;$('#workspace').hidden=false;$('#logout').hidden=false;await choosePanel('events');
    } finally {button.disabled=false;}
});});
$('#logout').addEventListener('click',()=>run(async()=>{try {await api('/api/auth/logout',{method:'POST',body:{}});}finally{lock();notice('已退出管理。');}}));
for(const kind of ['pointerdown','keydown','touchstart','wheel']) document.addEventListener(kind,()=>{
    if(!csrf||Date.now()-refreshAt<15000)return;refreshAt=Date.now();
    run(async()=>{const data=await api('/api/auth/session',{method:'POST',body:{}});if(csrf)expires=data.expires_at*1000;});
},{passive:true});
setInterval(()=>{
    if(!csrf)return;
    if(Date.now()>=expires){lock();notice('会话已过期，请重新登录。');return;}
},1000);
setInterval(()=>{if(!csrf||checkRunning)return;checkRunning=true;run(async()=>{try{const data=await api('/api/auth/check');if(csrf)expires=data.expires_at*1000;}finally{checkRunning=false;}});},15000);
run(async()=>{const status=await api('/api/auth/status');$('#setup-notice').textContent=status.initialized ? '使用板卡上设置的独立管理密码登录。' : '首次使用，请先在板卡上设置管理密码。';$('#login-form button').disabled=!status.initialized;});
