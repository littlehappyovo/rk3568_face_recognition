import {$,api,run,notice,node} from './api.js';
function confirmRemoval(title,message){return new Promise(resolve=>{
    const dialog=node('dialog'),form=node('form');form.className='dialog-form';form.append(node('h2',title),node('p',message));
    const actions=node('div','','actions'),cancel=node('button','取消'),remove=node('button','确认删除');cancel.type='button';remove.type='submit';remove.className='danger';actions.append(cancel,remove);form.append(actions);dialog.append(form);document.body.append(dialog);
    let approved=false;form.onsubmit=event=>{event.preventDefault();approved=true;dialog.close();};cancel.onclick=()=>dialog.close();
    dialog.onclose=()=>{dialog.remove();resolve(approved);};dialog.showModal();cancel.focus();
});}
export class People {
    constructor(){this.page=1;this.samplePage=1;this.selected=null;this.busy=false;this.generation=0;
        $('#people-search').onsubmit=event=>{event.preventDefault();this.page=1;run(()=>this.load());};
        $('#people-previous').onclick=()=>{if(this.page>1){--this.page;run(()=>this.load());}};
        $('#people-next').onclick=()=>{++this.page;run(()=>this.load());};
        $('#samples-previous').onclick=()=>{if(this.samplePage>1){--this.samplePage;run(()=>this.loadSamples());}};
        $('#samples-next').onclick=()=>{++this.samplePage;run(()=>this.loadSamples());};
        $('#new-person').onclick=()=>this.createDialog();
        $('#person-form').onsubmit=event=>{event.preventDefault();if(!this.selected)return;const id=this.selected.person_id;this.write(async()=>{
            await this.job('/api/people/'+encodeURIComponent(id),'PATCH',{display_name:$('#person-name').value.trim(),group:$('#person-group').value,enabled:$('#person-enabled').checked});
            await this.load(id);notice('人员资料已保存；身份图库可能需要重建。');
        });};
        $('#delete-person').onclick=()=>run(async()=>{if(!this.selected||this.busy)return;const person={...this.selected};
            if(!await confirmRemoval('删除人员',`确认删除${person.display_name}（${person.person_id}）及其样本索引？照片会归档保留，历史识别记录不变。`))return;
            this.write(async()=>{await this.job('/api/people/'+encodeURIComponent(person.person_id),'DELETE',{});this.selected=null;$('#person-detail').hidden=true;await this.load();notice('人员及样本索引已归档，请重建图库。');});
        });
        $('#sample-upload').onsubmit=event=>{event.preventDefault();if(!this.selected)return;const id=this.selected.person_id,files=Array.from($('#sample-files').files);
            if(!files.length)return;
            this.write(async()=>{let completed=0;for(const file of files){if(file.size>8*1024*1024)throw new Error(`${file.name}超过8MiB。`);const form=new FormData();form.append('image',file);await this.job('/api/people/'+encodeURIComponent(id)+'/samples','POST',form);++completed;notice(`已导入${completed}/${files.length}张照片。`);}$('#sample-files').value='';await this.load(id);notice(`已导入${completed}张照片，请重建图库。`);});
        };
        $('#rebuild-gallery').onclick=()=>this.write(async()=>{const result=await this.job('/api/gallery/rebuild','POST',{});
            notice(result.recognition_enabled ? '图库重建完成，识别已开启。' : '图库已生成；当前识别后端尚未就绪，请在板卡管理界面检查模型。',!result.recognition_enabled);});
        document.addEventListener('session-ended',()=>{++this.generation;this.selected=null;$('#person-detail').hidden=true;$('#people-table tbody').replaceChildren();$('#sample-list').replaceChildren();$('#sample-files').value='';});
    }
    async load(selectedId=this.selected?.person_id){const generation=++this.generation;
        const query=new URLSearchParams({q:$('#people-query').value.trim(),page:this.page,limit:50});const data=await api('/api/people?'+query);
        if(generation!==this.generation)return;const body=$('#people-table tbody');body.replaceChildren();
        for(const person of data.items){const tr=node('tr');for(const text of [person.person_id,person.display_name,person.enabled?'启用':'禁用',person.sample_count])tr.append(node('td',text));
            const td=node('td'),button=node('button','管理');button.onclick=()=>run(()=>this.select(person));td.append(button);tr.append(td);body.append(tr);}
        if(!data.items.length){const tr=node('tr'),td=node('td','没有符合条件的人员。','empty');td.colSpan=5;tr.append(td);body.append(tr);}
        $('#people-page').textContent=`共${data.total}人 · 第${this.page}/${Math.max(1,Math.ceil(data.total/50))}页`;
        $('#people-previous').disabled=this.page<=1;$('#people-next').disabled=this.page*50>=data.total;
        const selected=data.items.find(person=>person.person_id===selectedId);if(selected)await this.select(selected);else{this.selected=null;$('#person-detail').hidden=true;}
    }
    async select(person){this.selected={...person};this.samplePage=1;$('#person-detail').hidden=false;$('#person-id').textContent=person.person_id;$('#person-name').value=person.display_name;$('#person-group').value=person.group;$('#person-enabled').checked=!!person.enabled;await this.loadSamples();}
    async loadSamples(){if(!this.selected)return;const id=this.selected.person_id,page=this.samplePage;const data=await api('/api/people/'+encodeURIComponent(id)+'/samples?page='+page+'&limit=50');
        if(id!==this.selected?.person_id||page!==this.samplePage)return;const list=$('#sample-list');list.replaceChildren();
        for(const sample of data.items){const figure=node('figure'),img=node('img');img.src='/api/people/'+encodeURIComponent(id)+'/samples/'+encodeURIComponent(sample.sample_id);img.alt='人员样本';img.loading='lazy';
            const caption=node('figcaption',sample.sample_id),remove=node('button','删除照片');remove.type='button';remove.dataset.libraryWrite='';remove.disabled=this.busy;
            remove.onclick=()=>run(async()=>{if(this.busy)return;const sampleId=sample.sample_id;if(!await confirmRemoval('删除照片',`确认删除人员${id}的这张样本索引？原照片会归档保留。`))return;
                this.write(async()=>{await this.job('/api/people/'+encodeURIComponent(id)+'/samples/'+encodeURIComponent(sampleId),'DELETE',{});await this.load(id);notice('样本索引已归档，请重建图库。');});});
            figure.append(img,caption,remove);list.append(figure);}
        if(!data.items.length)list.append(node('p','此人员尚无照片，请上传清晰的单人脸照片。','muted'));
        $('#samples-page').textContent=`共${data.total}张 · 第${page}/${Math.max(1,Math.ceil(data.total/50))}页`;
        $('#samples-previous').disabled=page<=1;$('#samples-next').disabled=page*50>=data.total;
    }
    write(action){if(this.busy){notice('已有人员库任务正在执行，请稍候。',true);return;}this.busy=true;this.controls(true);
        run(async()=>{try{await action();}finally{this.busy=false;this.controls(false);}});
    }
    controls(disabled){document.querySelectorAll('[data-library-write]').forEach(control=>control.disabled=disabled);}
    async job(path,method,body){const started=await api(path,{method,body});if(!started.job_id)throw new Error('管理任务响应无效。');
        const deadline=Date.now()+185000;
        for(;;){notice('人脸库任务处理中，请保持此页面打开。');const state=await api('/api/jobs/'+encodeURIComponent(started.job_id));
            if(state.status==='completed')return state.result||{};
            if(state.status==='failed')throw new Error(state.error?.message||'任务失败，请核对板端状态。');
            if(Date.now()>=deadline)throw new Error('等待任务超时，请刷新核对实际结果，避免重复提交。');
            await new Promise(resolve=>setTimeout(resolve,500));
        }
    }
    createDialog(){if(this.busy)return;const dialog=node('dialog'),form=node('form');form.className='dialog-form';form.append(node('h2','新增人员'));
        const field=(text,id)=>{const label=node('label',text),input=node('input');input.id=id;input.required=true;label.append(input);form.append(label);return input;};
        const id=field('人员ID','new-person-id');id.pattern='[A-Za-z0-9_-]{1,32}';id.maxLength=32;id.placeholder='如P001';
        const name=field('姓名','new-person-name');name.maxLength=128;
        const actions=node('div','','actions'),cancel=node('button','取消'),save=node('button','创建');cancel.type='button';save.type='submit';save.dataset.libraryWrite='';cancel.onclick=()=>dialog.close();actions.append(cancel,save);form.append(actions);dialog.append(form);document.body.append(dialog);dialog.onclose=()=>dialog.remove();
        form.onsubmit=event=>{event.preventDefault();const personId=id.value.trim(),displayName=name.value.trim();this.write(async()=>{await this.job('/api/people','POST',{person_id:personId,display_name:displayName,group:'in_group',enabled:true});dialog.close();this.page=1;$('#people-query').value=personId;await this.load(personId);notice('人员已创建，请上传照片。');});};dialog.showModal();id.focus();
    }
}
