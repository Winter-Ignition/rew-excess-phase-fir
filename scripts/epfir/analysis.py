from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .common import db,power2,minimum,smooth,runs,fractional_peak,ease
from .inputs import load
from .low_reference import reference as low_reference,model_spread

def direct_peak(h,fs,declared=None):
    if declared is not None:
        p=float(declared)*fs
        if not 0<=p<len(h):
            raise ValueError('声明的直达峰不在 IR 样本范围内')
        return p,False
    global_peak=int(np.argmax(abs(h)))
    envelope=gaussian_filter1d(abs(h),max(.3,fs*.00002))
    peaks,_=find_peaks(envelope,height=envelope.max()*.18,prominence=envelope.max()*.06)
    choices=peaks[(peaks>=max(0,global_peak-int(fs*.05)))&(peaks<=global_peak+int(fs*.0001))]
    if global_peak<2:
        choices=np.r_[global_peak,choices]
    k=int(choices[0]) if len(choices) else global_peak
    lo=max(0,k-int(.00008*fs));hi=min(len(h),k+int(.00008*fs)+1)
    k=lo+int(np.argmax(abs(h[lo:hi])))
    ambiguous=bool(abs(h[k])<.45*abs(h[global_peak]) and global_peak-k>fs*.001)
    return fractional_peak(h,k),ambiguous

def segment_unwrap(z,valid):
    out=np.full(len(z),np.nan)
    for lo,hi in runs(valid):
        part=np.unwrap(np.angle(z[lo:hi]))
        if lo and np.isfinite(out[:lo]).any():
            prev=np.flatnonzero(np.isfinite(out[:lo]))[-1]
            part+=2*np.pi*np.rint((out[prev]-part[0])/(2*np.pi))
        out[lo:hi]=part
    ok=np.isfinite(out)
    if ok.sum()<8:
        raise ValueError('可用相位频点太少，不能生成可靠的补偿')
    return np.interp(np.arange(len(out)),np.flatnonzero(ok),out[ok])

def anchor_phase(f,phase,valid):
    anchor=valid&(f>=min(6000,f[-1]*.5))&(f<=min(8000,f[-1]*.8))
    if not anchor.any():
        anchor=valid&(f>=np.percentile(f[valid],75))
    return phase-2*np.pi*np.rint(np.median(phase[anchor])/(2*np.pi))

def vector_windows(f,phase,valid,c):
    z=np.exp(1j*phase);den=[];vectors=[]
    for width in [1/12,c['smoothing_octaves'],1/3]:
        weight=smooth(f,valid.astype(float),width)
        v=(smooth(f,z.real*valid,width)+1j*smooth(f,z.imag*valid,width))/np.maximum(weight,1e-12)
        den.append(weight);vectors.append(v)
    return np.asarray(vectors),np.asarray(den)

def frozen_phase(f,phase,valid,selection,c):
    vectors,den=vector_windows(f,phase,valid,c)
    weights=np.stack([gaussian_filter1d((selection==k).astype(float),3) for k in range(3)])
    weights/=np.maximum(weights.sum(axis=0),1e-30)
    value=np.sum(weights*vectors,axis=0)
    coverage=np.sum(weights*den,axis=0)
    good=(coverage>.35)&(abs(value)>.3)&valid
    result=anchor_phase(f,segment_unwrap(value,good),good)
    return result,abs(value),good

def phase_views(f,phase,valid,uncertainty,c):
    # 先在复平面平滑再展开；不把噪声/整圈跳变的整数圈平均成虚假斜坡。
    phase=anchor_phase(f,np.asarray(phase,float).copy(),valid)
    widths=[1/12,c['smoothing_octaves'],1/3]
    vectors,coverage=vector_windows(f,phase,valid,c)
    candidates=[]
    for vector,den in zip(vectors,coverage):
        good=valid&(abs(vector)>.3)&(den>.35)
        candidates.append(anchor_phase(f,segment_unwrap(vector,good),good))
    candidates=np.asarray(candidates)
    sensitivity=np.stack([smooth(f,uncertainty,w) for w in widths])
    selected=np.ones(len(f),int)
    if c['adaptive_smoothing']:
        unstable=sensitivity[1]>np.deg2rad(5)
        selected[unstable&(sensitivity[2]<.8*sensitivity[1])]=2
        # 仅当默认宽度把可信快速旋转平均到近零、窄窗又稳定时才收窄。
        # 不能因邻域窗口敏感度稍低，就把低频噪声区域默认改成 1/12。
        narrow=(abs(vectors[1])<.3)&(abs(vectors[0])>.7)&(sensitivity[0]<np.deg2rad(5))
        selected[narrow]=0
    # 连续混合局部宽度，防止自适应切换本身制造阶跃。
    weights=np.stack([gaussian_filter1d((selected==k).astype(float),3) for k in range(3)])
    weights/=np.maximum(weights.sum(axis=0),1e-30)
    chosen,coherence,good=frozen_phase(f,phase,valid,selected,c)
    used_width=np.sum(weights*np.array(widths)[:,None],axis=0)
    return phase,chosen,candidates,used_width,selected,coherence,good

def analyze(ch,c):
    source=load(ch,c['max_fft_samples']);meta=source['meta'];notes=[]
    if source['kind']=='frd' and source['fs']>48000:
        ff=source['f'];mag=source['mag'];mid=(ff>=200)&(ff<=3000);ultra=(ff>=30000)&(ff<=min(80000,source['fs']*.45))
        if ultra.sum()>20 and np.median(mag[ultra])<np.median(mag[mid])-35:
            # 超声已为噪声时，不让长段噪声平台决定可听频带最小相位参考。
            # 统一为 48k 分析规约；输出采样率仍服从请求，且不改播放延迟。
            native=source['fs'];spectrum=np.fft.rfft(source['samples']);n=int(round(len(source['samples'])*48000/native))
            if n%2==0 and len(spectrum)>=n//2+1:
                spectrum=spectrum[:n//2+1].copy();spectrum[-1]=spectrum[-1].real
                source={**source,'samples':np.fft.irfft(spectrum,n),'fs':48000.}
                meta.update(native_export_sample_rate=native,analysis_reference_sample_rate=48000,
                    ultrasonic_noise_policy='48k_audible_band_reference_not_physical_ultrasonic_reconstruction')
                notes.append('超声段由噪声主导：采用 48 kHz 可听带分析参考；低端/高端延伸仍是模型规约，非真实带外响应。')
    low=float(ch['min_usable_hz']);native_ir=None;native_fs=None
    if source['kind']=='excess':
        fn=source['f'];pn=source['phase'];mn=source['mag']
        valid_native=np.ones(len(fn),bool)
        # 已有剩余相位没有幅度/SNR 证据，只给予有限可信度。
        un_native=pn.copy() if source.get('phase_unwrapped',False) else segment_unwrap(np.exp(1j*pn),valid_native)
        raw_phase=un_native
        upper=min(24000.,fn[-1]);f=np.geomspace(max(5.,fn[0]),upper,1800)
        phi=np.interp(f,fn,raw_phase)
        mag=np.full(len(f),np.nan);confidence=np.full(len(f),.65)
        uncertainty=np.full(len(f),np.deg2rad(3));snr=np.full(len(f),np.nan)
        notes.append('已有剩余相位兼容通路：没有再次计算最小相位；不能检查原始噪声或真实脉冲。')
        meta.update(peak_alignment_available=False,spatial_validation=False)
    else:
        h=source['samples'].astype(float);fs=source['fs'];native_fs=fs
        if len(h)<64 or not np.isfinite(h).all() or abs(h).max()<=1e-20:
            raise ValueError('IR 样本过少、全零或存在非有限值')
        if source['kind']=='frd':
            # 保持 FRD 的完整线性网格，不将重建结果冒充未经处理的采集 IR。
            nfft=len(h);native_ir=h.copy()
            H=np.fft.rfft(h,nfft);fn=np.fft.rfftfreq(nfft,1/fs)
            peak,ambiguous=direct_peak(h,fs,ch.get('direct_peak_s'))
            rolled=np.roll(h,int(fs*.1)-int(round(peak)))
            shifted_peak=peak+int(fs*.1)-int(round(peak))
            window=np.ones(len(h));start=max(0,int(shifted_peak-c['window_pre_ms']*fs/1000))
            stop=min(len(h),int(shifted_peak+c['window_post_ms']*.8*fs/1000))
            window[:start]=0;window[stop:]=0
            tail=max(2,int(.05*fs));j=np.arange(max(start,stop-tail),stop)
            window[j]=1-ease((j-(stop-tail))/tail)
            alt_H=np.fft.rfft(rolled*window,nfft)
            alt_H*=np.exp(2j*np.pi*fn*(shifted_peak-peak)/fs)
            notes.append('频响重建通路：脉冲用于估计峰位，原始采集噪声、削波和绝对时序未认证。')
        else:
            peak,ambiguous=direct_peak(h,fs,ch.get('direct_peak_s'))
            start=max(0,int(peak-c['window_pre_ms']*fs/1000))
            stop=min(len(h),int(peak+c['window_post_ms']*fs/1000))
            original_peak=peak;h=h[start:stop].copy();peak-=start
            fade=max(2,int(.05*len(h)))
            h[-fade:]*=1-ease(np.linspace(0,1,fade))
            native_ir=h.copy();nfft=power2(max(4*len(h),16384))
            if nfft*2>c['max_fft_samples']:
                raise ValueError('IR 分解及 FFT 加倍检查超出 max_fft_samples；请明确增加资源上限或提供适当长度 IR。')
            H=np.fft.rfft(h,nfft);fn=np.fft.rfftfreq(nfft,1/fs)
            alt=h.copy();j0=int(peak+.8*(len(h)-peak));alt[j0:]=0
            alt_H=np.fft.rfft(alt,nfft)
            meta.update(analysis_start_sample=start,original_peak_sample=original_peak)
        phase_transfer=H.copy()
        calibration=ch.get('calibration')
        if calibration:
            from .inputs import read_numeric
            cal,_,_=read_numeric(calibration['path'])
            if ch.get('calibration_state','unknown')!='not_applied':
                raise ValueError('使用校准文件前必须明确其尚未应用，避免重复校准')
            if calibration.get('trust_phase',False):
                if cal.shape[1]<3:
                    raise ValueError('复数麦克风校准缺少相位列')
                cc=10**(np.interp(fn,cal[:,0],cal[:,1])/20)*np.exp(1j*np.deg2rad(np.interp(fn,cal[:,0],cal[:,2])))
                phase_transfer/=cc;alt_H/=cc
                meta['microphone_phase_policy']='declared_complex_calibration'
            else:
                notes.append('麦克风仅有幅度校准：相位分析保留自洽原始复响应，不猜测麦克风相位。')
                meta['microphone_phase_policy']='raw_chain_excess_microphone_excess_unknown'
        cutoff=ch.get('measurement_valid_low_hz',c.get('measurement_valid_low_hz'))
        extended,low_model=low_reference(fn,abs(phase_transfer),cutoff,ch.get('frozen_low_reference'))
        M=minimum(extended,nfft)
        spread=model_spread(fn,abs(phase_transfer),low_model,nfft)
        meta['low_reference']=low_model
        if cutoff is None:
            notes.append('有效测量低频下限未知：保留全频带参考，低端绝对剩余相位未认证；建议补充噪声截止点。')
        min_peak=fractional_peak(np.fft.irfft(M,nfft))
        # 同时处理实测和最小相位峰规约。纯最小相位+传播延迟应得到零剩余相位。
        removal=peak-min_peak
        if source['kind']=='frd':
            removal=(removal+nfft/2)%nfft-nfft/2
        Z=phase_transfer/M*np.exp(2j*np.pi*fn*removal/fs)
        alt_mag,_=low_reference(fn,abs(alt_H),cutoff,low_model if cutoff is not None else None)
        Ma=minimum(alt_mag,nfft)
        Za=alt_H/Ma*np.exp(2j*np.pi*fn*removal/fs)
        window_sensitivity=abs(np.angle(Za/Z))
        if source['kind']=='frd':
            ff2=np.fft.rfftfreq(nfft*2,1/fs)
            mag2=np.exp(np.interp(ff2,fn,np.log(np.maximum(abs(phase_transfer),1e-15))))
            M2=minimum(mag2,nfft*2)
        else:
            M2=minimum(abs(np.fft.rfft(h,nfft*2)),nfft*2)
            if calibration and calibration.get('trust_phase',False):
                # 已知完整校准时，同样对加倍网格重建自洽幅度。
                ff2=np.fft.rfftfreq(nfft*2,1/fs)
                mag2=abs(np.fft.rfft(h,nfft*2))/10**(np.interp(ff2,cal[:,0],cal[:,1])/20)
                M2=minimum(mag2,nfft*2)
        ff2=np.fft.rfftfreq(nfft*2,1/fs)
        ext2,_=low_reference(ff2,np.interp(ff2,fn,abs(phase_transfer)),cutoff,low_model if cutoff is not None else None)
        M2=minimum(ext2,nfft*2)
        fft_sensitivity=abs(np.angle(M2[::2]/M))
        # 尾噪是保守估计，不声明它可以分辨所有时钟、谐波或工频污染。
        if source['kind']=='ir' and peak>fs*.01:
            part=h[max(0,int(peak-fs*.018)):max(1,int(peak-fs*.004))]
        else:
            part=h[int(len(h)*.9):]
        noise=max(float(np.std(part))*np.sqrt(len(h)),float(abs(H).max())*1e-8)
        native_snr=db(abs(H)/noise)
        if source['kind']=='frd':
            # 周期重建 IR 的尾部可能是负时间直达能量，不能拿它估计采集底噪。
            native_snr=np.full(len(H),np.nan)
        valid_native=(fn>=max(2,source.get('f',np.array([2]))[0]))&(fn<=fs*.49)
        if cutoff is not None:
            valid_native&=fn>=cutoff
        if source['kind']=='ir':
            valid_native&=native_snr>8
        raw_unwrapped=segment_unwrap(Z,valid_native)
        upper=min(24000.,fs*.49,source.get('f',np.array([np.inf]))[-1])
        f=np.geomspace(5.,upper,1800)
        phi=np.interp(f,fn,raw_unwrapped);mag=np.interp(f,fn,db(H))
        snr=np.interp(f,fn,native_snr)
        uncertainty=np.interp(f,fn,window_sensitivity+fft_sensitivity)
        meta['low_reference_phase_spread_deg']=dict(frequency_hz=[low, max(low,80.),300.],
            max_deviation_deg=np.rad2deg(np.interp([low,max(low,80.),300.],fn,spread)).tolist())
        notch=mag-smooth(f,mag,1/3)
        evidence=np.full(len(f),.75) if source['kind']=='frd' else np.clip((snr-8)/24,0,1)
        confidence=evidence*np.exp(-(uncertainty/np.deg2rad(15))**2)
        confidence*=ease((notch+20)/12)
        # 深陷波不作精确反演，邻域仍保留可视化而非裁掉峰值。
        confidence*=.04+.96*ease((notch-c.get('notch_threshold_db',-6.))/4.)
        if cutoff is not None:
            confidence*=ease((f-cutoff)/max(2.,cutoff*.2))
        confidence*=1-ease((f-min(10000,fs*.20))/max(1,fs*.49-min(10000,fs*.20)))
        mid=(f>=200)&(f<=3000)
        hf_noise=ease((mag-float(np.median(mag[mid]))+40)/15)
        confidence*=np.where(f>=10000,hf_noise,1.)
        confidence[f>ch.get('measurement_valid_high_hz',22000.)]=0
        # 导出相位通常在低频噪声处不可靠；不让它影响展开锚点。
        if ambiguous:
            confidence*=.5;notes.append('早期峰与最大峰存在歧义；建议在脉冲图确认直达峰。')
        meta.update(peak_alignment_available=True,direct_peak_sample=peak,
            minimum_phase_peak_sample=min_peak,analysis_delay_removed_ms=removal/fs*1000,
            alignment_is_playback_delay=False,fft_samples=nfft,
            max_fft_doubling_sensitivity_deg=float(np.rad2deg(fft_sensitivity[(fn>=low)&(fn<=8000)]).max()),
            direct_peak_ambiguous=ambiguous,spatial_validation=False)
    upper_usable=ch.get('max_usable_hz',120. if ch.get('role')=='lfe' else c['independent_end_hz'])
    if ch.get('role')=='lfe' or upper_usable<c['independent_end_hz']:
        confidence*=1-ease((f-upper_usable*.9)/(upper_usable*.1))
    valid=(confidence>.08)&(f>=max(5,low*.7,ch.get('measurement_valid_low_hz') or 0))
    input_valid=valid.copy()
    phi,selected,candidates,width,selection,coherence,good=phase_views(f,phi,valid,uncertainty,c)
    confidence*=ease((coherence-.25)/.5);valid&=good
    # 无效区只供模型连续延伸，保留无效掩码，不伪装为观测值。
    confidence=np.clip(confidence,0,1)
    from scipy.interpolate import PchipInterpolator
    fit=PchipInterpolator(f,selected)
    gd=-fit.derivative()(f)/(2*np.pi)*1000
    return dict(name=ch['name'],f=f,phase_raw=phi,phase=selected,gd_ms=gd,magnitude_db=mag,
        confidence=confidence,valid=valid,snr_estimate_db=snr,uncertainty_rad=uncertainty,
        smooth_candidates=candidates,smoothing_width=width,smoothing_selection=selection,
        smoothing_input_valid=input_valid,vector_coherence=coherence,
        min_usable_hz=low,metadata=meta,notes=notes,native_ir=native_ir,native_fs=native_fs,
        crossover_hz=ch.get('crossover_hz',[]),max_usable_hz=upper_usable,kind=source['kind'])
