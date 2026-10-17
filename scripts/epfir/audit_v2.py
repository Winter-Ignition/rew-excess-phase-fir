"""独立回读系数与冻结预测；不信任设计器的 passed 标记。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import sha,db,minimum,write_json
from .analysis import frozen_phase


def audit(directory,write=True):
    root=Path(directory);m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    c=m['settings'];fs=c['sample_rate'];N=c['taps'];D=c['delay_samples'];results={};fail=[];predictions={}
    if D/fs*1000>c['max_latency_ms']+1e-8:fail.append('实际公共延迟超过预算')
    nfft=max(2097152,1<<(N*64-1).bit_length());f=np.fft.rfftfreq(nfft,1/fs);idx=np.arange(N)
    peak=lambda v:float(abs(v).max()) if len(v) else 0.
    rms=lambda v:float(np.sqrt(np.mean(v*v))) if len(v) else 0.
    ring=lambda h,ms:float(h[idx<D-fs*ms/1000]@h[idx<D-fs*ms/1000]/max(h@h,1e-30))
    data=np.load(root/'analysis.npz',allow_pickle=False)
    for item in m['files']:
        path=root/item['file'];info=sf.info(path);h,rate=sf.read(path,dtype='float64',always_2d=True)
        if rate!=fs or h.shape!=(N,len(item['channels'])) or info.subtype!='FLOAT' or info.format!='WAV':
            fail.append(item['file']+' 文件格式/声道/点数不符');continue
        if sha(path)!=item['sha256']:fail.append(item['file']+' 内容与清单不符')
        for j,name in enumerate(item['channels']):
            co=h[:,j]
            if not np.isfinite(co).all():fail.append(name+' 非有限系数');continue
            H=np.fft.rfft(co,nfft);gain=db(H);bins=np.linspace(0,len(f)-1,67,dtype=int)
            exact=np.exp(-2j*np.pi*f[bins,None]*idx/fs)@co
            metrics=dict(gain_max_abs_db=peak(gain),pre_1ms_energy_fraction=ring(co,1),
                pre_5ms_energy_fraction=ring(co,5),dtft_crosscheck_error=peak(exact-H[bins]),
                peak_sample=int(np.argmax(abs(co))),l1_peak_bound_db=float(db(abs(co).sum())))
            if metrics['gain_max_abs_db']>c['max_gain_error_db']+1e-7 or metrics['dtft_crosscheck_error']>1e-7:
                fail.append(name+' 增益/DTFT 核验失败')
            if ring(co,1)>c['max_far_pre_energy']+1e-9:fail.append(name+' 提前能量超限')
            if abs(metrics['peak_sample']-D)>2:fail.append(name+' 公共延迟/峰位不符')
            if name in m['identity_channels']:
                want=np.zeros(N);want[D]=1
                if not np.array_equal(co,want):fail.append(name+' 未处理声道纯延迟不符')
                results[name]=metrics;continue
            get=lambda k:data[name+'__'+k]
            old=get('loaded_h');base=get('baseline_h');O=np.fft.rfft(old,nfft);B=np.fft.rfft(base,nfft)
            freq=get('f');valid=get('valid');conf=get('confidence');raw=get('phase_raw')-get('nuisance_phase')
            def predict(response):
                ratio=response/O;phase=np.unwrap(np.angle(ratio/minimum(abs(ratio),nfft)))
                phase-=2*np.pi*np.rint(phase[-1]/(2*np.pi));addition=np.interp(freq,f,phase)
                return frozen_phase(freq,raw+addition,get('smoothing_input_valid'),get('smoothing_selection'),c)[0],addition
            p,addition=predict(H);bp,_=predict(B)
            before=get('normalized_phase');gd0=-np.gradient(before,freq)*1000/(2*np.pi)
            gd=-np.gradient(p,freq)*1000/(2*np.pi);bgd=-np.gradient(bp,freq)*1000/(2*np.pi)
            predictions[name]=(freq,p,valid)
            settings=m['channels'][name];limit=settings['calibration_ranges']['local_limit_hz']
            low=settings['min_usable_hz'];high=settings['calibration_ranges']['global_high_hz'];core=(freq>=low)&(freq<=high)&(conf>.12)
            lf=core&(freq<=limit);edge=(freq>=limit)&(freq<=limit*1.34)&core
            hf=(freq>=min(8000,high*.8))&(freq<=min(10000,high))&core
            stop=f>=limit;ratio=H/B;comp=np.rad2deg(np.angle(np.exp(1j*(addition-get('residual_target')))))
            metrics.update(local_stop_gain_db=peak(db(ratio[stop])),local_stop_phase_deg=peak(np.rad2deg(np.angle(ratio[stop]))),
                lf_gd_before_rms_ms=rms(gd0[lf]),lf_gd_before_peak_ms=peak(gd0[lf]),
                lf_gd_after_rms_ms=rms(gd[lf]),lf_gd_after_peak_ms=peak(gd[lf]),valid_lf_points=int(lf.sum()),
                transition_added_gd_peak_ms=peak((gd-bgd)[edge]),transition_added_gd_rms_ms=rms((gd-bgd)[edge]),
                hf_gd_peak_us=peak(gd[hf])*1000,hf_before_gd_peak_us=peak(gd0[hf])*1000,
                core_phase_rms_deg=rms(np.rad2deg(p[core])),composition_phase_error_deg=peak(comp[core]),
                composition_above300_error_deg=peak(comp[core&(freq>=300)]),
                prediction_replay_error_deg=peak(np.rad2deg(np.angle(np.exp(1j*(p-get('predicted_phase')))))[core]),
                relative_pre_energy_increase=float(ring(co,1)-ring(base,1)),old_pre_1ms_energy_fraction=ring(old,1))
            for key,lo,hi in [('local_boundary_band',limit,min(high,limit*1.34)),('upper_core_band',min(8000,high*.8),high)]:
                mask=(freq>=lo)&(freq<=hi)&valid
                metrics[key]=dict(frequency_hz=[lo,hi],valid_points=int(mask.sum()),phase_peak_deg=peak(np.rad2deg(p[mask])),gd_peak_ms=peak(gd[mask]))
            if metrics['local_stop_gain_db']>c['stop_gain_error_db'] or metrics['local_stop_phase_deg']>c['stop_phase_error_deg']:
                fail.append(name+' 局部项带外泄漏超限')
            if metrics['transition_added_gd_peak_ms']>c['transition_max_added_gd_ms'] or metrics['transition_added_gd_rms_ms']>c['transition_rms_added_gd_ms']:
                fail.append(name+' 局部退出区制造起伏')
            if peak(gd[lf])>max(peak(gd0[lf])*1.05,peak(gd0[lf])+.15) or rms(gd[lf])>rms(gd0[lf])*1.05+.02:
                fail.append(name+' 低频群延时明显恶化')
            if metrics['hf_gd_peak_us']>max(c['hf_transition_max_gd_us'],metrics['hf_before_gd_peak_us']*1.05):
                fail.append(name+' 高频衔接群延时恶化')
            if ring(co,1)>ring(base,1)*1.03+c['max_added_pre_energy']+1e-9:fail.append(name+' 局部提前能量增加超限')
            if c.get('loaded_filters'):
                if ring(co,1)>max(ring(old,1)*2,ring(old,1)+c['max_added_pre_energy'])+1e-9 or ring(co,5)>ring(old,5)+c['max_added_pre_energy']/3+1e-9:
                    fail.append(name+' 相对旧 FIR 的提前能量增加超限')
            if metrics['composition_phase_error_deg']>c['composition_max_phase_deg'] or metrics['composition_above300_error_deg']>.3:
                fail.append(name+' 完整替换组合误差超限')
            if metrics['prediction_replay_error_deg']>.2:fail.append(name+' 实际系数预测回放不一致')
            notch=get('notch_mask')&(freq>=low)&(freq<limit)
            metrics['notch_added_phase_deg']=peak(np.rad2deg(addition[notch]))
            if metrics['notch_added_phase_deg']>c['notch_max_added_phase_deg']+.1:fail.append(name+' 陷波处修正过强')
            end=settings['design']['hf_exit_hz'];beyond=f>=end;change=H/O
            metrics.update(exit_phase_error_deg=peak(np.rad2deg(np.angle(change[beyond]))),exit_gain_error_db=peak(db(change[beyond])))
            if metrics['exit_phase_error_deg']>c['stop_phase_error_deg'] or metrics['exit_gain_error_db']>c['stop_gain_error_db']:
                fail.append(name+' 高频退出后仍有明显修正')
            results[name]=metrics
    for l,r in m.get('pairs',[]):
        if l not in predictions or r not in predictions:continue
        fa,pa,va=predictions[l];fb,pb,vb=predictions[r]
        end=min(m['channels'][l]['design']['hf_exit_hz'],m['channels'][r]['design']['hf_exit_hz'])
        mask=(fa>=10000)&(fa<=min(18000,end-1000))&va&(np.interp(fa,fb,vb.astype(float))>.99)
        if mask.sum()<12:
            results[l+'_'+r+'_pair']=dict(status='not_assessed_no_trusted_hf_band');continue
        difference=np.rad2deg(np.angle(np.exp(1j*(pa-np.interp(fa,fb,pb)))))
        error=float(np.percentile(abs(difference[mask]),95))
        results[l+'_'+r+'_pair']=dict(hf_residual_difference_p95_deg=error,valid_points=int(mask.sum()))
        if error>c['hf_pair_tolerance_deg']:fail.append(l+'/'+r+' 高频剩余相位差超限')
    output=dict(passed=not fail,failures=fail,channels=results,fft_samples=nfft,
        validation='actual_coefficients_and_frozen_prediction_not_acoustic_remeasurement',version=2)
    data.close()
    if write:write_json(root/'acceptance.json',output)
    return output
