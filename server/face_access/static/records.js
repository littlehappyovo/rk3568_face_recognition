import { $, api, run, notice, node, localTime } from './api.js';
const names={events:'识别记录',visits:'进出配对',anomalies:'异常处理'};
const kinds={duplicate_entry:'重复进入',exit_without_entry:'无进入却离开',missing_exit:'超时未离开'};
const states={matched:'已配对',pending:'待出场',overdue:'已超时'};
export class Records {
    constructor(){ this.kind='events';this.page=1;this.total=0;this.generation=0;
        $('#timezone').textContent=`时间：${Intl.DateTimeFormat().resolvedOptions().timeZone}`;
        $('#filters').addEventListener('submit',event=>{event.preventDefault();this.page=1;run(()=>this.load());});
        $('#previous').onclick=()=>{if(this.page>1){--this.page;run(()=>this.load());}};
        $('#next').onclick=()=>{if(this.page*50<this.total){++this.page;run(()=>this.load());}};
        $('#export').onclick=()=>{window.location.href='/api/events.csv?'+this.query(false).toString();};
        document.addEventListener('session-ended',()=>{++this.generation;$('#records-table tbody').replaceChildren();});
    }
    query(paging=true){const query=new URLSearchParams();const allowed={events:['person_id','from','to','direction','station_id','decision'],visits:['person_id','from','to','status'],anomalies:['person_id','kind','resolved','include_history']}[this.kind];
        for(const [key,value] of new FormData($('#filters')))if(value&&allowed.includes(key))query.set(key,['from','to'].includes(key)?new Date(value).toISOString():value);
        if(paging){query.set('page',this.page);query.set('limit',50);}return query;
    }
    async select(kind){this.kind=kind;this.page=1;$('#records-title').textContent=names[kind];$('#export').hidden=kind!=='events';
        document.querySelectorAll('[data-filter]').forEach(element=>element.hidden=!(element.dataset.filter===kind || element.dataset.filter==='time'&&kind!=='anomalies'));
        await this.load();
    }
    async load(){const generation=++this.generation,kind=this.kind;$('#previous').disabled=true;$('#next').disabled=true;
        const data=await api('/api/'+kind+'?'+this.query());if(generation!==this.generation)return;this.total=data.total;
        const head=$('#records-table thead'),body=$('#records-table tbody');head.replaceChildren();body.replaceChildren();
        const columns=kind==='events'?['时间','姓名快照','人员ID','方向','终端','结果']:kind==='visits'?['姓名快照','人员ID','进入时间','离开时间','状态']:['人员ID','异常类型','状态','备注','处理时间','操作'];
        const row=node('tr');columns.forEach(text=>row.append(node('th',text)));head.append(row);
        for(const item of data.items){const tr=node('tr');let cells;
            if(kind==='events')cells=[localTime(item.observed_at),item.display_name||'陌生人 / 无权限',item.person_id||'—',item.direction==='IN'?'进':'出',item.station_id==='board-in'?'板卡':'历史电脑终端',item.decision==='allowed'?'已授权':'无权限'];
            else if(kind==='visits')cells=[item.display_name,item.person_id,localTime(item.entered_at),localTime(item.exited_at),states[item.status]||item.status];
            else cells=[item.person_id,kinds[item.kind]||item.kind,`${item.active?'当前':'历史'} · ${item.resolved_at?'已处理':'待处理'}`,item.note||'—',localTime(item.resolved_at)];
            cells.forEach(text=>{const td=node('td',text);td.title=String(text);tr.append(td);});
            if(kind==='anomalies'){const td=node('td');if(!item.resolved_at){const button=node('button','处理');button.onclick=()=>this.resolve(item);td.append(button);}tr.append(td);}body.append(tr);
        }
        if(!data.items.length){const tr=node('tr'),td=node('td','没有符合条件的记录。','empty');td.colSpan=columns.length;tr.append(td);body.append(tr);}
        const pages=Math.max(1,Math.ceil(data.total/50));$('#page-info').textContent=`共${data.total}条 · 第${this.page}/${pages}页`;
        $('#previous').disabled=this.page<=1;$('#next').disabled=this.page>=pages;
    }
    resolve(item){const dialog=node('dialog'),form=node('form');form.className='dialog-form';form.append(node('h2','处理异常'));
        form.append(node('p',`${item.person_id} · ${kinds[item.kind]||item.kind}`));const label=node('label','处理备注');const note=node('textarea');note.required=true;note.maxLength=2000;label.append(note);form.append(label);
        const actions=node('div','','actions'),cancel=node('button','取消'),save=node('button','确认处理');cancel.type='button';cancel.onclick=()=>dialog.close();save.type='submit';actions.append(cancel,save);form.append(actions);dialog.append(form);document.body.append(dialog);
        dialog.addEventListener('close',()=>dialog.remove());form.onsubmit=event=>{event.preventDefault();if(!note.value.trim()){note.setCustomValidity('请输入处理备注');note.reportValidity();return;}save.disabled=true;run(async()=>{try{await api('/api/anomalies/'+encodeURIComponent(item.id)+'/resolve',{method:'POST',body:{note:note.value.trim()}});dialog.close();await this.load();notice('异常处理备注已保存。');}finally{save.disabled=false;}});};
        note.oninput=()=>note.setCustomValidity('');dialog.showModal();note.focus();
    }
}
