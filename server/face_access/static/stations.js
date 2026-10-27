import {$,api,run,notice,node} from './api.js';
export class Stations {
    constructor(){
        $('#timeout-form').onsubmit=event=>{event.preventDefault();run(async()=>{const minutes=Number($('#timeout-minutes').value);if(!Number.isInteger(minutes)||minutes<0||minutes>525600)throw new Error('请输入有效的整数分钟数。');await api('/api/settings',{method:'PATCH',body:{missing_exit_seconds:minutes*60}});notice('超时规则已保存。');});};
    }
    async load(){const data=await api('/api/stations');$('#timeout-minutes').value=data.settings.missing_exit_seconds/60;const list=$('#station-list');list.replaceChildren();
        for(const station of data.items){const form=node('form');form.append(node('h3','板卡摄像头'));form.append(node('p',`配置版本${station.version}`,'muted'));
            const nameLabel=node('label','终端名称'),name=node('input');name.value=station.name;name.required=true;name.maxLength=64;nameLabel.append(name);
            const directionLabel=node('label','进出方向'),direction=node('select');for(const [value,text] of [['IN','进'],['OUT','出']]){const option=node('option',text);option.value=value;direction.append(option);}direction.value=station.direction;directionLabel.append(direction);
            const enabledLabel=node('label'),enabled=node('input');enabled.type='checkbox';enabled.checked=!!station.enabled;enabledLabel.append(enabled,document.createTextNode('启用此终端'));
            const save=node('button','保存终端');save.type='submit';form.append(nameLabel,directionLabel,enabledLabel,save);form.onsubmit=event=>{event.preventDefault();save.disabled=true;run(async()=>{try{await api('/api/stations/'+station.id,{method:'PATCH',body:{name:name.value.trim(),direction:direction.value,enabled:enabled.checked}});await this.load();notice('终端配置已保存。');}finally{save.disabled=false;}});};list.append(form);
        }
    }
}
