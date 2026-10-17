from __future__ import annotations
import numpy as np
from scipy.signal.windows import tukey
from .common import db,minimum,power2,ease,smooth
from .analysis import frozen_phase
from .multiscale import prepare,local as local_detail

def ringing(h,fs,delay):
    h=np.asarray(h,float);energy=float(h@h);t=(np.arange(len(h))-delay)/fs
    out={}
    for ms in (0,1,2,5,10,20):
        part=h[t < -ms/1000]
        out[f'pre_{ms}ms_energy_fraction']=float(part@part/max(energy,1e-30))
        out[f'pre_{ms}ms_peak_db']=float(db(abs(part).max() if len(part) else 0))
    out['l1_peak_bound_db']=float(db(abs(h).sum()))
    out['peak_sample']=int(np.argmax(abs(h)))
    return out

def fdw(f,phase,c):
    """局部校准专用频变时间窗；不对整体宽带分频模型作错误截短。"""
    n=power2(max(c['taps']*8,65536));fs=c['sample_rate']
    ff=np.fft.rfftfreq(n,1/fs)
    target=np.interp(ff,f,phase,left=0,right=0)
    # 小相位目标的时域核；线性化只用于限制支持区，最终仍取 exp(j*phase)。
    kernel=np.fft.fftshift(np.fft.irfft(1j*target,n))
    t=(np.arange(n)-n//2)/fs;delay=c['delay_samples']/fs
    if delay<=0:
        return np.zeros_like(phase)
    result=np.zeros(len(ff))
    centers=np.geomspace(max(15,c['min_usable_hz']*.7),c['lf_stop_hz'],12)
    weights=np.exp(-.5*(np.log(np.maximum(ff[:,None],1e-3)/centers)/.32)**2)
    weights/=np.maximum(weights.sum(axis=1,keepdims=True),1e-20)
    for k,fc in enumerate(centers):
        duration=min(delay*.90,c['fdw_cycles']/fc)
        time_window=1-ease((abs(t)-duration*.65)/max(duration*.35,1/fs))
        part=np.fft.rfft(np.fft.ifftshift(kernel*time_window),n)
        result+=part.imag*weights[:,k]
    # 窗后重新施加频带边界，渲染后另查实际带外泄漏。
    result*=ease((ff-c['min_usable_hz'])/max(8,c['min_usable_hz']*.35))
    result*=1-ease((ff-c['lf_local_high_hz'])/(c['lf_stop_hz']-c['lf_local_high_hz']))
    return np.interp(f,ff,result)

def render(f,phase,c,iterations=65):
    n=power2(max(c['taps']*16,65536));fs=c['sample_rate'];d=c['delay_samples']
    ff=np.fft.rfftfreq(n,1/fs)
    p=np.interp(ff,f,phase,left=phase[0],right=0)
    # 实数 IR 的 DC 必须为实数。只在不可用低频内延伸到最近的整数 pi。
    dc=np.pi*np.rint(p[0]/np.pi)
    blend=ease(ff/max(1,min(f[0]*2,c['min_usable_hz']*.5)))
    p=dc*(1-blend)+p*blend
    wanted=np.exp(1j*p-2j*np.pi*ff*d/fs)
    wanted[0]=np.cos(dc);wanted[-1]=(-1.)**d
    win=tukey(c['taps'],.06);gain=np.ones(len(ff));best=None;besterr=float('inf')
    for _ in range(iterations):
        h=np.fft.irfft(wanted*gain,n)[:c['taps']]*win
        response=abs(np.fft.rfft(h,n));error=float(abs(db(response)).max())
        if error<besterr:
            best=h.copy();besterr=error
        if error<c['max_gain_error_db']*.35:
            break
        gain=np.clip(gain/np.maximum(response,1e-8),.35,3.)
    return best.astype(np.float32)

def transfer(h,c,nfft=None):
    nfft=nfft or power2(max(1048576,c['taps']*32))
    f=np.fft.rfftfreq(nfft,1/c['sample_rate'])
    H=np.fft.rfft(np.asarray(h,float),nfft)
    M=minimum(abs(H),nfft)
    phase=np.unwrap(np.angle(H*np.exp(2j*np.pi*f*c['delay_samples']/c['sample_rate'])/M))
    phase-=2*np.pi*np.rint(phase[-1]/(2*np.pi))
    weighted=np.fft.rfft(np.asarray(h,float)*(np.arange(len(h))-c['delay_samples'])/c['sample_rate'],nfft)
    gd=1000*np.real(weighted/H)
    minweighted=np.fft.rfft(np.fft.irfft(M,nfft)*np.arange(nfft)/c['sample_rate'],nfft)
    gd-=1000*np.real(minweighted/M)
    return dict(f=f,H=H,gain_db=db(H),phase=phase,gd_ms=gd)

def design(a,c):
    if 'global_variants' not in a:
        prepare(a,c)
    f=a['f'];band=a['calibration_ranges'];limit=band['local_limit_hz']
    candidates=[];chosen=None
    for variant in a['global_variants']:
      strengths=(1.,.65,0.) if c['optimize'] and abs(variant['smooth_target']).max()>np.deg2rad(1) else (1.,)
      for strength in strengths:
        base=variant['parametric_target']+strength*variant['smooth_target']+a['pair_target']
        forward=variant['reference']+strength*variant['smooth_forward']
        raw_local=local_detail(a,forward,c)
        local_c={**c,'min_usable_hz':a['min_usable_hz'],'lf_stop_hz':limit,'lf_local_high_hz':band['local_fade_start_hz']}
        local=fdw(f,raw_local,local_c) if limit>a['min_usable_hz']+15 else np.zeros(len(f))
        local[f>=limit]=0
        baseline=render(f,base,c);bt=transfer(baseline,c,262144)
        if abs(bt['gain_db']).max()>c['max_gain_error_db'] or ringing(baseline,c['sample_rate'],c['delay_samples'])['pre_1ms_energy_fraction']>c['max_far_pre_energy']:
            continue
        baseline_phase=frozen_phase(f,a['phase_raw']-a['nuisance_phase']+np.interp(f,bt['f'],bt['phase']),a['smoothing_input_valid'],a['smoothing_selection'],c)[0]
        baseline_gd=-np.gradient(baseline_phase,f)/(2*np.pi)*1000
        for depth in ((c['local_depth'],.5*c['local_depth'],0.) if c['optimize'] else (c['local_depth'],0.)):
            h=baseline if depth==0 else render(f,base+depth*local,c)
            tr=transfer(h,c,262144);ring=ringing(h,c['sample_rate'],c['delay_samples'])
            high=tr['f']>=limit
            difference=tr['H']/bt['H']
            leakage_gain=float(abs(db(difference[high])).max())
            leakage_phase=float(abs(np.rad2deg(np.angle(difference[high]))).max())
            gain=float(abs(tr['gain_db']).max())
            valid=(f>=max(40,a['min_usable_hz']))&(f<=c['lf_local_high_hz'])&(a['confidence']>.12)
            before=-np.gradient(a['normalized_phase'],f)/(2*np.pi)*1000
            predicted=frozen_phase(f,a['phase_raw']-a['nuisance_phase']+np.interp(f,tr['f'],tr['phase']),
                                   a['smoothing_input_valid'],a['smoothing_selection'],c)[0]
            after=-np.gradient(predicted,f)/(2*np.pi)*1000
            edge=(f>=limit)&(f<=min(1.25*limit,band['global_high_hz']))&(a['confidence']>.12)
            extra=after[edge]-baseline_gd[edge]
            edge_peak=float(abs(extra).max()) if edge.any() else 0.
            edge_rms=float(np.sqrt(np.mean(extra**2))) if edge.any() else 0.
            edge_ok=edge_peak<=c['transition_max_added_gd_ms'] and edge_rms<=c['transition_rms_added_gd_ms']
            rms=float(np.sqrt(np.mean(after[valid]**2))) if valid.any() else float('inf')
            peak=float(abs(after[valid]).max());before_peak=float(abs(before[valid]).max())
            before_rms=float(np.sqrt(np.mean(before[valid]**2)))
            shape_ok=peak<=max(before_peak*1.05,before_peak+.15) and rms<=before_rms*1.05+.02
            br=ringing(baseline,c['sample_rate'],c['delay_samples'])
            absolute_ok=ring['pre_1ms_energy_fraction']<=c['max_far_pre_energy']
            relative_ok=ring['pre_1ms_energy_fraction']<=br['pre_1ms_energy_fraction']*1.03+c['max_added_pre_energy']
            ok=gain<=c['max_gain_error_db']*.95 and leakage_gain<=c['stop_gain_error_db'] and leakage_phase<=c['stop_phase_error_deg'] and absolute_ok and relative_ok and shape_ok and edge_ok
            wide=(f>=a['min_usable_hz'])&(f<=band['global_transition_start_hz'])&(a['confidence']>.12)
            slow=smooth(f,predicted,.7)
            global_error=float(np.rad2deg(np.sqrt(np.average(slow[wide]**2,weights=a['confidence'][wide]))))
            item=dict(global_depth=1.,global_smooth_strength=strength,baseline_variant=variant['name'],local_depth=depth,gain_error_db=gain,
                      local_limit_hz=limit,global_smooth_phase_rms_deg=global_error,
                      local_stop_gain_db=leakage_gain,local_stop_phase_deg=leakage_phase,
                      lf_predicted_gd_rms_ms=rms,lf_predicted_gd_peak_ms=peak,shape_ok=shape_ok,
                      transition_added_gd_peak_ms=edge_peak,transition_added_gd_rms_ms=edge_rms,transition_ok=edge_ok,
                      pre_ok=absolute_ok and relative_ok,passed=ok)
            candidates.append(item)
            objective=global_error/8+.8*rms+.05*peak+20*ring['pre_1ms_energy_fraction']
            if ok and (chosen is None or objective<chosen[0]):
                chosen=(objective,h,baseline,item,ring,variant['regularization'])
    if chosen is None:
        raise ValueError(f"{a['name']} 在指定长度/延迟/增益/前振铃限制下无法形成有效补偿；需要修改预算或改善测量，不能用纯延迟冒充成功。")
    _,h,baseline,metrics,ring,regularization=chosen
    tr=transfer(h,c)
    if abs(tr['gain_db']).max()>c['max_gain_error_db']:
        raise ValueError('加密网格检验未通过全通误差；没有发布滤波文件')
    a.update(h=h,baseline_h=baseline,actual=tr,design_metrics=metrics,ringing=ring,design_candidates=candidates)
    a['global_regularization']={**regularization,'strength':metrics['global_smooth_strength']}
    selected=next(v for v in a['global_variants'] if v['name']==metrics['baseline_variant'])
    a['selected_global_smooth_phase']=-metrics['global_smooth_strength']*selected['smooth_forward']
    a['selected_local_phase']=metrics['local_depth']*fdw(f,local_detail(a,selected['reference']+metrics['global_smooth_strength']*selected['smooth_forward'],c),
        {**c,'min_usable_hz':a['min_usable_hz'],'lf_stop_hz':limit,'lf_local_high_hz':band['local_fade_start_hz']}) if limit>a['min_usable_hz']+15 else np.zeros(len(f))
    a['selected_local_phase'][f>=limit]=0
    a['predicted_phase']=frozen_phase(f,a['phase_raw']-a['nuisance_phase']+np.interp(f,tr['f'],tr['phase']),
                                    a['smoothing_input_valid'],a['smoothing_selection'],c)[0]
    a['predicted_gd_ms']=-np.gradient(a['predicted_phase'],f)/(2*np.pi)*1000
    bt=transfer(baseline,c)
    a['baseline_predicted_phase']=frozen_phase(f,a['phase_raw']-a['nuisance_phase']+np.interp(f,bt['f'],bt['phase']),
                                    a['smoothing_input_valid'],a['smoothing_selection'],c)[0]
    a['baseline_predicted_gd_ms']=-np.gradient(a['baseline_predicted_phase'],f)/(2*np.pi)*1000
    return a
