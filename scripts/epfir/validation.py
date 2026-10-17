"""从磁盘读取交付物重新核验，不使用设计器的缓存响应或 passed 标记。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import sha,db,write_json,minimum
from .analysis import frozen_phase

def audit(directory,write=True):
    root=Path(directory);m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    c=m['settings'];fs=c['sample_rate'];N=c['taps'];D=c['delay_samples'];results={};fail=[]
    if D/fs*1000>c['max_latency_ms']+1e-8:
        fail.append('实际公共参考延迟超过用户预算')
    nfft=max(2097152,1 << (N*64-1).bit_length())
    f=np.fft.rfftfreq(nfft,1/fs)
    for item in m['files']:
        path=root/item['file'];info=sf.info(path)
        h,rate=sf.read(path,dtype='float64',always_2d=True)
        if rate!=fs or len(h)!=N or info.subtype!='FLOAT' or h.shape[1]!=len(item['channels']) or info.format!='WAV':
            fail.append(f"{item['file']} 文件格式/点数/声道映射不符")
            continue
        if sha(path)!=item['sha256']:
            fail.append(f"{item['file']} 内容与清单不符")
        for j,name in enumerate(item['channels']):
            coeff=h[:,j]
            if not np.isfinite(coeff).all():
                fail.append(f'{name} 系数包含非有限数');continue
            H=np.fft.rfft(coeff,nfft);gain=db(H);energy=coeff@coeff
            idx=np.arange(len(coeff));pre=idx<D-fs*.001
            pre_energy=float(coeff[pre]@coeff[pre]/max(energy,1e-30))
            # 独立直接 DTFT 抽查，避免只依赖同一种 FFT 路径。
            bins=np.linspace(0,len(f)-1,67,dtype=int)
            exact=np.exp(-2j*np.pi*f[bins,None]*idx/fs)@coeff
            dtft_error=float(abs(exact-H[bins]).max())
            metrics=dict(gain_max_abs_db=float(abs(gain).max()),
                         pre_1ms_energy_fraction=pre_energy,dtft_crosscheck_error=dtft_error,
                         peak_sample=int(np.argmax(abs(coeff))),l1_peak_bound_db=float(db(abs(coeff).sum())))
            if metrics['gain_max_abs_db']>c['max_gain_error_db']+1e-7 or dtft_error>1e-7:
                fail.append(f'{name} 增益或 DTFT 核验失败')
            if pre_energy>c['max_far_pre_energy']+1e-9:
                fail.append(f'{name} 提前 1ms 的能量超过绝对限制')
            if name in m['identity_channels']:
                wanted=np.zeros(N);wanted[D]=1.
                if not np.array_equal(coeff,wanted):
                    fail.append(f'{name} 公共纯延迟不符')
            else:
                data=np.load(root/'analysis.npz',allow_pickle=False)
                local_limit=m['channels'][name].get('calibration_ranges',{}).get('local_limit_hz',c['lf_stop_hz'])
                base=data[name+'__baseline_h'];B=np.fft.rfft(base,nfft);ratio=H/B;stop=f>=local_limit
                metrics['local_stop_gain_db']=float(abs(db(ratio[stop])).max())
                metrics['local_stop_phase_deg']=float(abs(np.rad2deg(np.angle(ratio[stop]))).max())
                if metrics['local_stop_gain_db']>c['stop_gain_error_db'] or metrics['local_stop_phase_deg']>c['stop_phase_error_deg']:
                    fail.append(f'{name} 低频局部项带外泄漏超标')
                before=data[name+'__normalized_phase'];freq=data[name+'__f'];conf=data[name+'__confidence']
                # 独立倒谱分解最后导出的 FIR，去掉其极小幅度误差对应的最小相位。
                cep=np.fft.irfft(np.log(np.maximum(abs(H),1e-15)),nfft)
                causal=np.zeros(nfft);causal[0]=cep[0];causal[1:nfft//2]=2*cep[1:nfft//2];causal[nfft//2]=cep[nfft//2]
                M=np.exp(np.fft.rfft(causal))
                gd_h=np.real(np.fft.rfft(coeff*(idx-D)/fs,nfft)/H)*1000
                hm=np.fft.irfft(M,nfft)
                gd_m=np.real(np.fft.rfft(hm*np.arange(nfft)/fs,nfft)/M)*1000
                gd0=-np.gradient(before,freq)/(2*np.pi)*1000
                phase_h=np.unwrap(np.angle(H*np.exp(2j*np.pi*f*D/fs)/M))
                predicted=frozen_phase(freq,data[name+'__phase_raw']-data[name+'__nuisance_phase']+np.interp(freq,f,phase_h),
                    data[name+'__smoothing_input_valid'],data[name+'__smoothing_selection'],c)[0]
                gd1=-np.gradient(predicted,freq)/(2*np.pi)*1000
                base_phase=np.unwrap(np.angle(B*np.exp(2j*np.pi*f*D/fs)/minimum(abs(B),nfft)))
                base_prediction=frozen_phase(freq,data[name+'__phase_raw']-data[name+'__nuisance_phase']+np.interp(freq,f,base_phase),
                    data[name+'__smoothing_input_valid'],data[name+'__smoothing_selection'],c)[0]
                baseline_gd=-np.gradient(base_prediction,freq)/(2*np.pi)*1000
                if c.get('phase_framework')=='multiscale_regularized_v2':
                    edge=(freq>=local_limit)&(freq<=local_limit*1.25)&(conf>.12)
                else:
                    edge=(freq>=c['lf_local_high_hz'])&(freq<=c['lf_stop_hz'])&(conf>.12)
                extra=gd1[edge]-baseline_gd[edge]
                metrics['transition_added_gd_peak_ms']=float(abs(extra).max()) if edge.any() else 0.
                metrics['transition_added_gd_rms_ms']=float(np.sqrt(np.mean(extra**2))) if edge.any() else 0.
                if metrics['transition_added_gd_peak_ms']>c.get('transition_max_added_gd_ms',.25) or metrics['transition_added_gd_rms_ms']>c.get('transition_rms_added_gd_ms',.12):
                    fail.append(f'{name} 局部校准在退出区制造额外群延时起伏')
                band=(freq>=max(40,m['channels'][name]['min_usable_hz']))&(freq<=c['lf_local_high_hz'])&(conf>.12)
                metrics.update(lf_gd_before_rms_ms=float(np.sqrt(np.mean(gd0[band]**2))),
                               lf_gd_before_peak_ms=float(abs(gd0[band]).max()),
                               lf_gd_after_rms_ms=float(np.sqrt(np.mean(gd1[band]**2))),
                               lf_gd_after_peak_ms=float(abs(gd1[band]).max()),valid_lf_points=int(band.sum()))
                if metrics['lf_gd_after_peak_ms']>max(metrics['lf_gd_before_peak_ms']*1.05,metrics['lf_gd_before_peak_ms']+.15) or metrics['lf_gd_after_rms_ms']>metrics['lf_gd_before_rms_ms']*1.05+.02:
                    fail.append(f'{name} 低频峰值或 RMS 明显恶化')
                br=base@base;bp=base[pre]@base[pre]/max(br,1e-30)
                if pre_energy>bp*1.03+c['max_added_pre_energy']+1e-9:
                    fail.append(f'{name} 局部补偿增加的提前能量超限')
                metrics['relative_pre_energy_increase']=float(pre_energy-bp)
            results[name]=metrics
    # 对称项实际均值应近零；不把两侧独立宽带相位强行统一。
    for pair in m.get('pairs',[]):
        l,r=pair;data=np.load(root/'analysis.npz',allow_pickle=False)
        fcheck=np.geomspace(10000,min(c['pair_end_hz'],fs*.49),160)
        phases=[]
        for name in pair:
            item=next(z for z in m['files'] if name in z['channels'])
            hh,_=sf.read(root/item['file'],dtype='float64',always_2d=True);co=hh[:,item['channels'].index(name)]
            Hx=np.exp(-2j*np.pi*fcheck[:,None]*np.arange(N)/fs)@co
            phases.append(np.angle(Hx*np.exp(2j*np.pi*fcheck*D/fs)))
        common=abs(np.rad2deg(np.angle(np.exp(1j*(phases[0]+phases[1])))/2)).max()
        results[l+'_'+r+'_pair']=dict(hf_common_correction_max_deg=float(common))
        if common>.25:
            fail.append(f'{l}/{r} 10kHz 以上校准未保持近似差分对称')
    output=dict(passed=not fail,failures=fail,channels=results,
                validation='coefficients_and_prediction_only_not_acoustic_remeasurement',fft_samples=nfft)
    if write:
        write_json(root/'acceptance.json',output)
    return output
