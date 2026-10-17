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
    from .unified import design_group
    return design_group([a],c)[0]
