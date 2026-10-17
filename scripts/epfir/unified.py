"""联合选择宽带、局部和高频连接，旧 FIR 只合并一次，公共延迟只保留一次。"""
from __future__ import annotations
import numpy as np
import soundfile as sf
from scipy.interpolate import CubicSpline
from .common import ease,smooth,minimum,db,sha
from .analysis import frozen_phase
from .multiscale import prepare,local
from .design import render,transfer,ringing,fdw


def continuation(f,y,start,end):
    """匹配起点相位/斜率/曲率，终点三者归零。只延伸校准量。"""
    if end<=start:
        raise ValueError('高频退出区没有可用宽度')
    spline=CubicSpline(f,y);length=end-start
    a=np.zeros(6);a[:3]=[spline(start),spline(start,1)*length,spline(start,2)*length**2/2]
    a[3:]=np.linalg.solve([[1,1,1],[3,4,5],[6,12,20]],
        [-sum(a[:3]),-a[1]-2*a[2],-2*a[2]])
    u=np.clip((f-start)/length,0,1)
    tail=sum(v*u**k for k,v in enumerate(a))
    return np.where(f<=start,y,np.where(f<end,tail,0.))


def loaded(a,c):
    n=c['taps'];h=np.zeros(n);h[c['delay_samples']]=1
    for item in c.get('loaded_filters',[]):
        if a['name'] not in item['channels']:
            continue
        if sha(item['path'])!=item['sha256']:
            raise ValueError('旧 FIR 在分析期间变化，停止生成')
        samples,rate=sf.read(item['path'],always_2d=True,dtype='float64')
        if rate!=c['sample_rate'] or samples.shape!=(n,len(item['channels'])) or not np.isfinite(samples).all():
            raise ValueError('旧 FIR 必须匹配采样率、长度、声道映射且系数有限')
        h=samples[:,item['channels'].index(a['name'])]
        if abs(db(np.fft.rfft(h,262144))).max()>c['max_gain_error_db']:
            raise ValueError('旧 FIR 不是允许误差内的近全通；不能丢弃其幅度后称作完整替换')
        if abs(int(np.argmax(abs(h)))-c['delay_samples'])>2:
            raise ValueError('旧 FIR 脉冲峰不符合所声明公共延迟，需确认时间规约')
        break
    return h.astype(np.float32)


def prediction(a,phase,c):
    return frozen_phase(a['f'],a['phase_raw']-a['nuisance_phase']+phase,
        a['smoothing_input_valid'],a['smoothing_selection'],c)[0]


def half_difference(f,left,right):
    delta=np.unwrap(np.angle(np.exp(1j*(left-right))))
    anchor=(f>=8000)&(f<=10000)
    if not anchor.any():anchor=f>=np.percentile(f,80)
    # 低频完整绕圈不能被除成高频的半圈（180°）差分校准。
    delta-=2*np.pi*np.rint(np.median(delta[anchor])/(2*np.pi))
    return smooth(f,delta,1/6)/2


def mask_metrics(a,p,target,h,base,old,actual,c,end):
    f=a['f'];conf=a['confidence'];before=a['normalized_phase'];gd=-np.gradient(p,f)*1000/(2*np.pi)
    gd0=-np.gradient(before,f)*1000/(2*np.pi)
    limit=a['calibration_ranges']['local_limit_hz'];core=(f>=a['min_usable_hz'])&(f<=a['max_usable_hz'])&(conf>.12)
    lf=core&(f<=limit);edge=(f>=limit)&(f<=limit*1.34)&core
    hf=(f>=min(8000,a['max_usable_hz']*.8))&(f<=min(10000,a['max_usable_hz']))&core
    baseline_phase=prediction(a,np.interp(f,actual['f'],actual['baseline_add']),c)
    extra=gd+np.gradient(baseline_phase,f)*1000/(2*np.pi)
    ring=ringing(h,c['sample_rate'],c['delay_samples']);br=ringing(base,c['sample_rate'],c['delay_samples'])
    previous=ringing(old,c['sample_rate'],c['delay_samples']);stop=actual['f']>=limit
    ratio=actual['H']/actual['baseline_H'];beyond=actual['f']>=end
    change=actual['H']/actual['old_H'];notches=a.get('notch_mask',np.zeros(len(f),bool))&(f>=a['min_usable_hz'])&(f<limit)
    rms=lambda x:float(np.sqrt(np.mean(x*x))) if len(x) else 0.
    peak=lambda x:float(abs(x).max()) if len(x) else 0.
    metrics=dict(global_smooth_phase_rms_deg=rms(np.rad2deg(p[core])),
        lf_predicted_gd_rms_ms=rms(gd[lf]),lf_predicted_gd_peak_ms=peak(gd[lf]),
        gain_error_db=peak(actual['gain_db']),local_stop_gain_db=peak(db(ratio[stop])),
        local_stop_phase_deg=peak(np.rad2deg(np.angle(ratio[stop]))),
        transition_added_gd_peak_ms=peak(extra[edge]),transition_added_gd_rms_ms=rms(extra[edge]),
        hf_gd_peak_us=peak(gd[hf])*1000,hf_before_gd_peak_us=peak(gd0[hf])*1000,
        hf_exit_hz=end,notch_added_phase_deg=peak(np.rad2deg(target[notches])),
        exit_phase_error_deg=peak(np.rad2deg(np.angle(change[beyond]))),exit_gain_error_db=peak(db(change[beyond])))
    relative=ring['pre_1ms_energy_fraction']<=br['pre_1ms_energy_fraction']*1.03+c['max_added_pre_energy']
    # 复测替换还须对比实际旧滤波，避免以更差的新基线隐藏提前能量增加。
    if c.get('loaded_filters'):
        relative &= ring['pre_1ms_energy_fraction']<=max(previous['pre_1ms_energy_fraction']*2,
            previous['pre_1ms_energy_fraction']+c['max_added_pre_energy'])
        relative &= ring['pre_5ms_energy_fraction']<=previous['pre_5ms_energy_fraction']+c['max_added_pre_energy']/3
    shape=(not lf.any()) or (peak(gd[lf])<=max(peak(gd0[lf])*1.05,peak(gd0[lf])+.15)
        and rms(gd[lf])<=rms(gd0[lf])*1.05+.02)
    passed=(metrics['gain_error_db']<=c['max_gain_error_db']*.98 and relative and shape
        and ring['pre_1ms_energy_fraction']<=c['max_far_pre_energy']
        and metrics['local_stop_gain_db']<=c['stop_gain_error_db']
        and metrics['local_stop_phase_deg']<=c['stop_phase_error_deg']
        and metrics['transition_added_gd_peak_ms']<=c['transition_max_added_gd_ms']
        and metrics['transition_added_gd_rms_ms']<=c['transition_rms_added_gd_ms']
        and metrics['hf_gd_peak_us']<=max(c['hf_transition_max_gd_us'],metrics['hf_before_gd_peak_us']*1.05)
        and metrics['exit_phase_error_deg']<=c['stop_phase_error_deg']
        and metrics['exit_gain_error_db']<=c['stop_gain_error_db']
        and metrics['notch_added_phase_deg']<=c['notch_max_added_phase_deg'])
    return metrics,ring,bool(passed)


def design_group(group,c):
    for a in group:
        if 'global_variants' not in a:prepare(a,c)
        mag=a['magnitude_db'];f=a['f']
        a['notch_mask']=np.isfinite(mag)&((mag-smooth(f,np.nan_to_num(mag),1/3))<c['notch_threshold_db'])
    f=group[0]['f'];paired=len(group)==2
    high=min(a['calibration_ranges']['global_high_hz'] for a in group)
    # 带宽受限声道在自身上限平顺退出，不生成不存在的高频对称约束。
    usable=min(float(a['f'][np.flatnonzero(a['valid'])[-1]]) for a in group)
    paired=paired and high>=10000 and usable>11000
    start=min(8000.,high*.8)
    maxend=min(c['pair_end_hz'],usable*.97)
    if high<10000:maxend=high
    if maxend<=high and high>=10000:
        raise ValueError('10 kHz 以上缺少可信衔接频带，请改善测量或明确降低校准上限')
    ends=sorted(set([max(high+500,maxend-2000),maxend])) if c['optimize'] and maxend>high+1000 else [maxend]
    old=[loaded(a,c) for a in group]
    n=262144;ff=np.fft.rfftfreq(n,1/c['sample_rate'])
    oldH=[np.fft.rfft(h,n) for h in old]
    # 保留旧完整滤波的总相位；合并目标不带第二份公共延迟。
    oldp=[np.unwrap(np.angle(H*np.exp(2j*np.pi*ff*c['delay_samples']/c['sample_rate']))) for H in oldH]
    oldp=[p-2*np.pi*np.rint(p[-1]/(2*np.pi)) for p in oldp]
    best=None;records=[]
    for end in ends:
      phases=[np.interp(f,a['f'],a['normalized_phase']) for a in group]
      if paired:
        common=smooth(f,(phases[0]+phases[1])/2,1/6)
        diff=half_difference(f,phases[0],phases[1])
        hc=continuation(f,-common,high,end)
        fade=max(high,end-4000);weight=ease((f-start)/(high-start))*(1-ease((f-fade)/(end-fade)))
        hd=-diff*(1-ease((f-fade)/(end-fade)))
        for _ in range(3):
            ps=[prediction(a,np.interp(a['f'],f,hc+(hd if j==0 else -hd)),c) for j,a in enumerate(group)]
            error=.5*np.angle(np.exp(1j*(ps[0]-np.interp(f,group[1]['f'],ps[1]))))
            hd-=smooth(f,error,1/6)*weight
        high_targets=[hc+hd,hc-hd]
      else:
        high_targets=[continuation(a['f'],-smooth(a['f'],a['normalized_phase'],1/6),high if end>high else start,end) for a in group]
      choices=[(v,s) for v in range(max(len(a['global_variants']) for a in group))
               for s in ((1.,.7,.3) if c['optimize'] else (1.,))]
      for variant_index,strength in choices:
       for depth in ((c['local_depth'],c['local_depth']*.3,0.) if c['optimize'] else (c['local_depth'],)):
        products=[];passed=True;score=0.
        for j,a in enumerate(group):
          af=a['f'];variant=a['global_variants'][min(variant_index,len(a['global_variants'])-1)]
          # 低端参考不确定时仅逐渐减弱低频残差；不能为保护低端让整个中高频一起失去校准。
          low=a['min_usable_hz'];recover=max(low+1,min(300.,a['max_usable_hz']*.8))
          strength_curve=strength+(1-strength)*ease((af-low)/(recover-low))
          forward=variant['reference']+strength_curve*variant['smooth_forward']
          blend=ease((af-max(start*.75,a['min_usable_hz']))/max(1,start-max(start*.75,a['min_usable_hz'])))
          hi=np.interp(af,f,high_targets[j]) if paired else high_targets[j]
          core=-forward*(1-blend)+hi*blend
          limit=a['calibration_ranges']['local_limit_hz']
          detail=fdw(af,local(a,forward,c),{**c,'min_usable_hz':a['min_usable_hz'],
              'lf_stop_hz':limit,'lf_local_high_hz':a['calibration_ranges']['local_fade_start_hz']}) if limit>a['min_usable_hz']+15 else np.zeros(len(af))
          # 自动陷波保护：不精确补偿低可信快速旋转，也不将它外推到宽带。
          detail*=.05+.95*ease((a['confidence']-.08)/.4)
          target=core+depth*detail
          baseline=render(ff,oldp[j]+np.interp(ff,af,core,left=0,right=0),c)
          h=render(ff,oldp[j]+np.interp(ff,af,target,left=0,right=0),c)
          tr=transfer(h,c,n);B=np.fft.rfft(baseline,n);ratio=tr['H']/oldH[j]
          add=np.unwrap(np.angle(ratio/minimum(abs(ratio),n)));add-=2*np.pi*np.rint(add[-1]/(2*np.pi))
          bp=np.unwrap(np.angle((B/oldH[j])/minimum(abs(B/oldH[j]),n)));bp-=2*np.pi*np.rint(bp[-1]/(2*np.pi))
          tr.update(old_H=oldH[j],baseline_H=B,baseline_add=bp)
          pred=prediction(a,np.interp(af,ff,add),c)
          metrics,ring,ok=mask_metrics(a,pred,target,h,baseline,old[j],tr,c,end)
          passed &= ok;score+=metrics['global_smooth_phase_rms_deg']/8+metrics['lf_predicted_gd_rms_ms']
          products.append((h,baseline,pred,target,core,depth*detail,tr,metrics,ring,add))
        symmetry=None
        if paired:
            mask=(f>=high)&(f<=min(end-1000,18000))&group[0]['valid']&(np.interp(f,group[1]['f'],group[1]['valid'].astype(float))>.99)
            if mask.sum()<12:passed=False
            else:
                error=np.rad2deg(np.angle(np.exp(1j*(products[0][2]-np.interp(f,group[1]['f'],products[1][2])))))
                symmetry=float(np.percentile(abs(error[mask]),95));passed &= symmetry<=c['hf_pair_tolerance_deg'];score+=symmetry/5
        row=dict(hf_exit_hz=end,baseline_variant_index=variant_index,global_smooth_strength=strength,local_depth=depth,pair_p95_deg=symmetry,
            channels=[p[7] for p in products],passed=bool(passed),score=float(score))
        records.append(row)
        if passed and (best is None or score<best[0]):best=(score,products,row)
    if best is None:
        from .common import clean
        import json
        raise ValueError('没有候选同时通过增益、衔接、陷波和提前能量限制：'+json.dumps(clean(records[-2:]),ensure_ascii=False))
    for j,a in enumerate(group):
        h,base,p,target,core,detail,tr,metrics,ring,add=best[1][j]
        actual=transfer(h,c)
        a.update(h=h,baseline_h=base,loaded_h=old[j],actual=actual,predicted_phase=p,
            predicted_gd_ms=-np.gradient(p,a['f'])*1000/(2*np.pi),design_metrics={**metrics,**best[2]},
            ringing=ring,design_candidates=records,selected_global_smooth_phase=core,selected_local_phase=detail,
            residual_target=target,actual_residual_phase=np.interp(a['f'],ff,add),
            global_regularization=a['global_variants'][0]['regularization'])
        a['baseline_predicted_phase']=prediction(a,np.interp(a['f'],ff,tr['baseline_add']),c)
        a['baseline_predicted_gd_ms']=-np.gradient(a['baseline_predicted_phase'],a['f'])*1000/(2*np.pi)
    return group
