from __future__ import annotations
import base64,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write_json,clean,db

def wrapped(y):
    z=np.rad2deg(np.angle(np.exp(1j*y)))
    z[np.r_[False,abs(np.diff(z))>180]]=np.nan
    return z

def build(analyses,c,out,acceptance=None):
    out=Path(out);plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'figure.facecolor':'white'})
    count=len(analyses);fig,axes=plt.subplots(4,count,figsize=(7*count,13),squeeze=False,layout='constrained')
    records=[];series={}
    for col,a in enumerate(analyses):
        f=a['f'];actual=a.get('actual');phi=a.get('normalized_phase',a['phase'])
        before=-np.gradient(phi,f)/(2*np.pi)*1000
        after=before.copy();phase_after=phi.copy()
        if actual:
            after=a['predicted_gd_ms'];phase_after=a['predicted_phase']
        valid=a['valid'];mask=(f>=a['min_usable_hz'])&(f<=c['lf_stop_hz'])
        ax=axes[0,col];ax.plot(f[mask],np.where(valid[mask],before[mask],np.nan),color='#777777',label='校准前')
        ax.plot(f[mask],np.where(valid[mask],after[mask],np.nan),color='#0072B2',label='应用后预测')
        ax.axhline(0,color='#555',lw=.7);ax.set(xlabel='频率 / Hz',ylabel='剩余群延时 / ms',title=a['name']+'：低频峰谷（纵轴含全部峰值）')
        if a.get('calibration_ranges'):
            ax.axvline(a['calibration_ranges']['local_limit_hz'],color='#884EA0',ls='--',lw=.8,label='局部细节上限')
        ax.legend();ax.grid(alpha=.2)
        ax=axes[1,col];ax.semilogx(f,np.where(a['valid'],np.rad2deg(phi),np.nan),color='#777777',label='校准前')
        ax.semilogx(f,np.where(a['valid'],np.rad2deg(phase_after),np.nan),color='#0072B2',label='应用后预测')
        ax.set(xlim=(max(20,a['min_usable_hz']),min(c['pair_end_hz'],f[-1])),ylabel='连续剩余相位 / °',xlabel='频率 / Hz',title='复数矢量平滑后展开；灰区不连线')
        ax.grid(alpha=.2);ax.legend()
        ax=axes[2,col]
        if actual:
            ax.semilogx(actual['f'][1:],actual['gain_db'][1:],color='#009E73')
        ax.axhline(c['max_gain_error_db'],color='#D55E00',ls='--');ax.axhline(-c['max_gain_error_db'],color='#D55E00',ls='--')
        ax.set(xlim=(10,c['sample_rate']/2),ylabel='滤波增益 / dB',xlabel='频率 / Hz',title='实际 float32 FIR：越靠近 0 dB 越接近全通')
        ax.grid(alpha=.2)
        ax=axes[3,col]
        if 'h' in a:
            t=(np.arange(len(a['h']))-c['delay_samples'])/c['sample_rate']*1000
            ax.plot(t,db(a['h']),color='#0072B2',lw=.7)
            ax.set_ylim(-150,3);ax.axvline(0,color='#555',ls='--')
            ax.set(xlim=(-c['delay_samples']/c['sample_rate']*1000,25),xlabel='相对公共参考延迟 / ms',ylabel='脉冲绝对幅度 / dB',title='负时间能量：越低越少，不等于必然不可闻')
        for lo,hi in [(f[0],a['min_usable_hz'])]:
            axes[1,col].axvspan(lo,hi,color='#bbb',alpha=.2)
        for axis in axes[:2,col]:
            for start,stop in __import__('epfir.common',fromlist=['runs']).runs(~valid):
                axis.axvspan(f[start],f[min(stop,len(f)-1)],color='#aaa',alpha=.15)
        axes[0,col].set_xlim(max(40,a['min_usable_hz']),c['lf_stop_hz'])
        matrix=np.c_[f,a['magnitude_db'],a['confidence'],a['smoothing_width'],phi,phase_after,before,after]
        np.savetxt(out/(a['name']+'_curves.csv'),matrix,delimiter=',',header='frequency_hz,magnitude_db,confidence,smoothing_octaves,before_excess_rad,predicted_excess_rad,before_gd_ms,predicted_gd_ms',comments='')
        records.append(dict(channel=a['name'],metadata=a['metadata'],notes=a['notes'],model=a.get('model'),
            design=a.get('design_metrics'),ringing=a.get('ringing'),design_candidates=a.get('design_candidates'),
            calibration_ranges=a.get('calibration_ranges'),global_regularization=a.get('global_regularization')))
        series[a['name']]=dict(f=f.tolist(),before=before.tolist(),after=after.tolist(),confidence=a['confidence'].tolist())
    limits=[ax.get_ylim() for ax in axes[0,:]]
    for ax in axes[0,:]:
        ax.set_ylim(min(x[0] for x in limits),max(x[1] for x in limits))
    fig.suptitle('剩余相位校准 · 系数与预测验收（尚未做应用后声学复测）',fontsize=16)
    fig.savefig(out/'overview.png',dpi=155);plt.close(fig)
    from .transitions import build as transition_report
    transition_report(analyses,c,out)
    multiscale=all('selected_global_smooth_phase' in a for a in analyses)
    if multiscale:
        fig,axes=plt.subplots(2,count,figsize=(7*count,7),squeeze=False,layout='constrained')
        for col,a in enumerate(analyses):
            f=a['f'];limit=a['calibration_ranges']['local_limit_hz'];low=a['min_usable_hz']
            ax=axes[0,col]
            visible=a['valid']&(f>=low)&(f<=10000)
            ax.semilogx(f,np.where(visible,np.rad2deg(a['baseline_predicted_phase']),np.nan),color='#777',label='整体正则化后（无局部项）')
            ax.semilogx(f,np.where(visible,np.rad2deg(a['predicted_phase']),np.nan),color='#0072B2',label='整体 + 局部，实际 FIR 预测')
            ax.set(xlim=(low,10000),xlabel='Hz',ylabel='剩余相位 / °',title=a['name']+'：应用后残余相位放大');ax.legend();ax.grid(alpha=.2)
            ax=axes[1,col]
            ax.semilogx(f,np.rad2deg(a['selected_global_smooth_phase']),color='#884EA0',label='整体平滑校准（含先验和高频连接）')
            ax.semilogx(f,np.rad2deg(a['selected_local_phase']),color='#009E73',label='局部细节校准')
            ax.axvline(limit,color='#555',ls='--',label=f'局部上限 {limit:.0f} Hz')
            ax.set(xlim=(low,10000),xlabel='Hz',ylabel='校准相位 / °',title='整体跨过局部上限；局部在上限前归零');ax.legend();ax.grid(alpha=.2)
        fig.savefig(out/'multiscale.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(3,count,figsize=(7*count,10),squeeze=False,layout='constrained')
    for col,a in enumerate(analyses):
        f=a['f'];ax=axes[0,col]
        for label,p in zip(['1/12 oct','1/6 oct 默认','1/3 oct'],a['smooth_candidates']):
            ax.semilogx(f,np.rad2deg(p),label=label,lw=.8)
        ax.set(xlim=(max(20,a['min_usable_hz']),min(20000,f[-1])),xlabel='Hz',ylabel='展开相位 / °',title=a['name']+'：平滑和相位圈数诊断');ax.legend()
        ax=axes[1,col];ax.semilogx(f,a['confidence'],label='可信度（不是概率）')
        ax.semilogx(f,np.minimum(np.rad2deg(a['uncertainty_rad'])/30,1),label='窗口/FFT敏感度 ÷30°')
        ax.set(xlabel='Hz',ylim=(0,1.05));ax.legend()
        ax=axes[2,col]
        if 'h' in a:
            # 高斯窄带脉冲也有固有时间展宽，图只用于定位异常能量。
            hh=a['h'];N=65536;ff=np.fft.rfftfreq(N,1/c['sample_rate']);H=np.fft.rfft(hh,N)
            centers=np.geomspace(40,10000,50);T=(np.arange(N)-N//2)/c['sample_rate']*1000
            rows=[]
            for fc in centers:
                w=np.exp(-.5*((ff-fc)/max(20,fc*.22))**2)
                time=np.fft.fftshift(np.fft.irfft(H*w*np.exp(2j*np.pi*ff*c['delay_samples']/c['sample_rate']),N))
                rows.append(db(time))
            window=(T>=-40)&(T<=20)
            im=ax.pcolormesh(T[window],centers,np.array(rows)[:,window],cmap='magma',vmin=-110,vmax=-20,shading='auto')
            ax.set(yscale='log',xlabel='相对参考时间 / ms',ylabel='中心频率 / Hz',title='窄带时间图（包含分析窗的固有展宽）');fig.colorbar(im,ax=ax,label='绝对幅度 / dB')
    fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
    lookup={a['name']:a for a in analyses}
    if c.get('pairs'):
        fig,axes=plt.subplots(len(c['pairs']),2,figsize=(13,4*len(c['pairs'])),squeeze=False,layout='constrained')
        for row,(l,r) in enumerate(c['pairs']):
            a,b=lookup[l],lookup[r];f=np.geomspace(8000,min(22000,a['f'][-1],b['f'][-1]),600)
            left=np.interp(f,a['f'],a['normalized_phase']);right=np.interp(f,b['f'],b['normalized_phase'])
            pair_valid=(np.interp(f,a['f'],a['valid'].astype(float))>.999)&(np.interp(f,b['f'],b['valid'].astype(float))>.999)
            ca=np.interp(f,a['f'],a['actual_residual_phase']) if a.get('actual') else np.zeros(len(f))
            cb=np.interp(f,b['f'],b['actual_residual_phase']) if b.get('actual') else np.zeros(len(f))
            predicted_left=np.interp(f,a['f'],a.get('predicted_phase',a['normalized_phase']))
            predicted_right=np.interp(f,b['f'],b.get('predicted_phase',b['normalized_phase']))
            axes[row,0].plot(f,np.where(pair_valid,wrapped(left-right),np.nan),color='#777',label='校准前相位差')
            axes[row,0].plot(f,np.where(pair_valid,wrapped(predicted_left-predicted_right),np.nan),color='#0072B2',label='应用后预测相位差')
            axes[row,0].set(title=l+'/'+r+'：10 kHz 附近优先缩小差异；无效区留空',xlabel='Hz',ylabel='左右差 / °',xlim=(f[0],f[-1]));axes[row,0].legend()
            hf=f>=10000
            axes[row,1].plot(f[hf],np.rad2deg((ca[hf]+cb[hf])/2),label='共同校准量（平滑接回，非全程 0°）')
            axes[row,1].set(title='10 kHz 以上共同趋势平滑衔接',xlabel='Hz',ylabel='校准相位均值 / °');axes[row,1].set_xlim(10000,f[-1]);axes[row,1].legend()
        fig.savefig(out/'hf_pairs.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,count,figsize=(7*count,3.5),squeeze=False,layout='constrained')
    for col,a in enumerate(analyses):
        ax=axes[0,col]
        if a.get('native_ir') is not None:
            h=a['native_ir'];p=a['metadata']['direct_peak_sample'];fs=a['native_fs']
            center=int(round(p));indices=np.arange(center-int(fs*.003),center+int(fs*.006))
            if a['metadata'].get('reconstructed'):
                y=h[indices%len(h)]
            else:
                valid=(indices>=0)&(indices<len(h));indices=indices[valid];y=h[indices]
            ax.plot((indices-p)/fs*1000,y,lw=.8);ax.axvline(0,color='#555',ls='--')
        ax.set(title=a['name']+'：分析峰对齐（不是播放延迟）',xlabel='相对识别峰 / ms',ylabel='输入 IR 幅度')
    fig.savefig(out/'input_peaks.png',dpi=150);plt.close(fig)
    write_json(out/'analysis_summary.json',records)
    metrics=html.escape(json.dumps(clean(acceptance or {}),ensure_ascii=False,indent=2))
    images=[('overview.png','全频段总览'),('band_detail.png','按校准范围显示的频段细节'),('diagnostics.png','平滑与时间图'),('input_peaks.png','输入脉冲峰')]
    if multiscale:
        images.insert(1,('multiscale.png','整体慢变校准与局部细节分工'))
    if c.get('pairs'):
        images.append(('hf_pairs.png','高频配对'))
    pictures=''.join('<details open><summary>'+label+'</summary><img style="width:100%" src="data:image/png;base64,'+base64.b64encode((out/file).read_bytes()).decode()+'"></details>' for file,label in images)
    code=json.dumps(clean(series),ensure_ascii=False).replace('</',r'<\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>剩余相位校准报告</title><style>body{font:16px system-ui;max-width:1450px;margin:auto;padding:24px;background:#f5f7fa;color:#17232b}canvas{width:100%;background:white}details,section{background:white;padding:16px;margin:16px 0}pre{overflow:auto;font-size:13px}input{width:110px}img{height:auto}</style><h1>剩余相位校准</h1><p>蓝线是应用滤波后的预测，灰线是原测量；它不是应用后的实测。灰色区域可信度较低。先看低频峰谷是否变小，再看增益与负时间能量是否满足预算。</p><p>'''+html.escape(c.get('measurement_notes',''))+'''</p><section><b>低频群延时放大查看</b>　声道 <select id="ch"></select>　起点 <input id="lo" type="number" value="40"> Hz　终点 <input id="hi" type="number" value="400"> Hz <button onclick="draw()">更新</button><canvas id="plot" width="1400" height="430"></canvas><p id="hover">移动鼠标查看频率；调整起止频率可放大。</p></section>'''+pictures+'<details><summary>独立验收数字</summary><pre>'+metrics+'</pre></details><p>剩余相位校准后，必须在本次 FIR 启用时重新测量并重新校准频响，已有频响 EQ 也一样；最终在 FIR 与重新调整后的 EQ 同时启用时复测。系统公共参考延迟不代表真实音箱到达时间已配对。峰值余量建议按所有声道共同衰减处理。</p><script>const data='+code+''';const sel=document.getElementById('ch');for(const n in data){sel.add(new Option(n,n))}sel.onchange=draw;let bounds;function draw(){let d=data[sel.value],lo=+document.getElementById('lo').value,hi=+document.getElementById('hi').value;if(!(hi>lo))return;let rows=d.f.map((f,i)=>[f,d.before[i],d.after[i]]).filter(r=>r[0]>=lo&&r[0]<=hi);if(!rows.length)return;let yy=rows.flatMap(r=>r.slice(1));let mn=Math.min(0,...yy),mx=Math.max(0,...yy);if(mx-mn<.1){mn-=.1;mx+=.1}let c=document.getElementById('plot').getContext('2d'),W=1400,H=430;c.clearRect(0,0,W,H);c.font='17px system-ui';c.fillStyle='#263238';c.fillText('灰：校准前　蓝：应用后预测　纵轴 ms',75,24);for(let j=0;j<6;j++){let y=45+j*65;c.strokeStyle='#ddd';c.beginPath();c.moveTo(75,y);c.lineTo(1370,y);c.stroke();c.fillText((mx-(mx-mn)*j/5).toFixed(2),4,y+6)}for(let k=1;k<=2;k++){c.strokeStyle=k==1?'#777':'#0072b2';c.lineWidth=2;c.beginPath();rows.forEach((r,i)=>{let x=75+(r[0]-lo)/(hi-lo)*1295,y=45+(mx-r[k])/(mx-mn)*325;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()}for(let j=0;j<=6;j++)c.fillText((lo+(hi-lo)*j/6).toFixed(0)+' Hz',65+j*215,410);bounds=[lo,hi]}document.getElementById('plot').onmousemove=e=>{if(!bounds)return;let rect=e.target.getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width*1400,f=bounds[0]+(x-75)/1295*(bounds[1]-bounds[0]);document.getElementById('hover').textContent='约 '+f.toFixed(1)+' Hz'};draw()</script></html>'''
    if analyses[0].get('calibration_ranges'):
        descriptions=[]
        for a in analyses:
            ranges=a['calibration_ranges'];move=ranges['movement_estimate_hz']
            movement=f'{move:.0f} Hz' if np.isfinite(move) else '未构成有限上限'
            descriptions.append(html.escape(f"{a['name']}：整体 {ranges['global_low_hz']:.0f}–{ranges['global_high_hz']:.0f} Hz；局部细节 ≤{ranges['local_limit_hz']:.0f} Hz；头动经验估计 {movement}。"))
        note='<section><b>本次校准范围</b><p>'+'<br>'.join(descriptions)+'</p><p>整体校准使用跨频带的正则化慢变基底；头部移动只限制局部细节。局部上限不是整体退出点。单点输入未验证移动区域，8–10 kHz 与高频规则平顺衔接。</p></section>'
        page=page.replace('<h1>剩余相位校准</h1>','<h1>剩余相位校准</h1>'+note)
    (out/'report.html').write_text(page,encoding='utf-8')
