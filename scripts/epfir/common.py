from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

def power2(n):
    return 1 << max(1, int(np.ceil(n))-1).bit_length()

def db(x):
    return 20*np.log10(np.maximum(np.abs(x),1e-15))

def ease(x):
    x=np.clip(x,0.,1.)
    return x**3*(10-15*x+6*x*x)

def minimum(magnitude,nfft):
    c=np.fft.irfft(np.log(np.maximum(magnitude,1e-15)),nfft)
    m=np.zeros(nfft);m[0]=c[0];m[1:nfft//2]=2*c[1:nfft//2];m[nfft//2]=c[nfft//2]
    return np.exp(np.fft.rfft(m))

def smooth(f,y,width):
    if width<=0:
        return np.asarray(y).copy()
    x=np.log2(f);grid=np.linspace(x[0],x[-1],len(f))
    z=np.interp(grid,x,y)
    z=gaussian_filter1d(z,max(.01,width/(grid[1]-grid[0])/2.354820045),mode='nearest')
    return np.interp(x,grid,z)

def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))

def fractional_peak(h,index=None):
    k=int(np.argmax(abs(h))) if index is None else int(index)
    if 0<k<len(h)-1:
        # 因果尖峰一侧为零时，抛物插值会凭空产生约半样本偏移。
        if min(abs(h[k-1]),abs(h[k+1]))<abs(h[k])*1e-4:
            return float(k)
        a,b,c=np.log(np.maximum(abs(h[k-1:k+2]),1e-30))
        d=.5*(a-c)/(a-2*b+c) if abs(a-2*b+c)>1e-15 else 0.
        return k+float(np.clip(d,-.5,.5))
    return float(k)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value,dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value,np.ndarray):
        return clean(value.tolist())
    if isinstance(value,np.generic):
        return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value

def write_json(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

DEFAULTS=dict(sample_rate=192000,taps=16384,max_latency_ms=42.67,smoothing_octaves=1/6,
    adaptive_smoothing=True,lf_stop_hz=400.,lf_local_high_hz=300.,independent_end_hz=10000.,
    independent_transition_hz=8000.,pair_end_hz=20000.,head_half_ranges_cm=None,
    min_usable_hz=40.,max_gain_error_db=.03,stop_gain_error_db=.005,stop_phase_error_deg=.1,
    local_depth=1.,optimize=True,max_fft_samples=4194304,existing_phase_correction='off',
    window_pre_ms=20.,window_post_ms=1000.,fdw_cycles=4.,max_far_pre_energy=.005,
    max_added_pre_energy=.0005,transition_max_added_gd_ms=.25,transition_rms_added_gd_ms=.12,
    local_detail_cap_hz=300.,global_basis_sigma_octaves=.35,global_regularization=.8,
    local_highpass_octaves=.8,phase_framework='unified_v2',measurement_valid_low_hz=None,
    hf_pair_tolerance_deg=5.,hf_transition_max_gd_us=40.,composition_max_phase_deg=2.,
    notch_threshold_db=-6.,notch_max_added_phase_deg=8.)

def request(path,remeasure=False):
    p=Path(path).resolve();r=json.loads(p.read_text(encoding='utf-8-sig'))
    c={**DEFAULTS,**r};channels=c.get('channels',[])
    if not channels:
        raise ValueError('请求里缺少 channels；每个逻辑声道需要一个输入文件。')
    if c.get('reference_directory'):
        q=Path(c['reference_directory']);q=(p.parent/q).resolve() if not q.is_absolute() else q.resolve()
        reference=json.loads((q/'manifest.json').read_text(encoding='utf-8'))
        if reference.get('version')!=2:
            raise ValueError('冻结参考目录需要 v2 清单；v1 WAV 可单独作为 loaded_filters 输入')
        for key in ['smoothing_octaves','adaptive_smoothing','window_pre_ms','window_post_ms']:
            c[key]=reference['settings'][key]
        for ch in channels:
            frozen=reference['channels'][ch['name']].get('low_reference')
            if frozen and frozen.get('models'):
                declared=ch.get('measurement_valid_low_hz',c['measurement_valid_low_hz'])
                if declared is not None and declared!=frozen['cutoff_hz']:
                    raise ValueError('冻结参考的噪声下限不一致；不能重新拟合后作同一规约比较')
                ch.update(frozen_low_reference=frozen,measurement_valid_low_hz=frozen['cutoff_hz'])
    for key in ['sample_rate','taps','max_fft_samples']:
        if not isinstance(c[key],int):
            raise ValueError(f'{key} 必须为整数')
    if not 8000<=c['sample_rate']<=384000 or not 256<=c['taps']<=131072:
        raise ValueError('采样率或点数超出支持范围')
    for key in ['max_latency_ms','min_usable_hz','lf_stop_hz','lf_local_high_hz','smoothing_octaves','local_depth']:
        if not np.isfinite(c[key]):
            raise ValueError(f'{key} 必须为有限数值')
    if not 0<=c['local_depth']<=1 or c['max_latency_ms']<0:
        raise ValueError('延迟不得为负，校准强度须在 0–1 之间')
    if not (0<c['min_usable_hz']<c['independent_end_hz']<c['sample_rate']/2 and
            0<c['lf_stop_hz']<=c['independent_transition_hz']<c['independent_end_hz']):
        raise ValueError('校准频带次序或输出采样率不正确')
    if not c['independent_end_hz']<c['pair_end_hz']<c['sample_rate']/2:
        raise ValueError('左右高频差分退出频率须高于独立校准上限')
    if c['existing_phase_correction'] not in ('off','none','this_filter'):
        raise ValueError('先关闭待替换的混合/剩余相位校正，再使用相同状态的测量。')
    if c['existing_phase_correction']=='this_filter' and not remeasure and not c.get('loaded_filters'):
        raise ValueError('带旧 FIR 的复测需要 loaded_filters，不能当作未校准测量生成完整替换文件')
    if c.get('loaded_filters') and c['existing_phase_correction']!='this_filter':
        raise ValueError('loaded_filters 仅适用于明确声明 this_filter 的复测')
    head=c['head_half_ranges_cm']
    if head is not None and (len(head)!=3 or not np.isfinite(head).all() or min(head)<0):
        raise ValueError('头部范围需为三个非负半范围，单位厘米')
    labels=set();hashes=set()
    for key in ['max_gain_error_db','stop_gain_error_db','stop_phase_error_deg','window_pre_ms','window_post_ms','fdw_cycles','max_far_pre_energy','max_added_pre_energy','smoothing_octaves','transition_max_added_gd_ms','transition_rms_added_gd_ms','local_detail_cap_hz','global_basis_sigma_octaves','global_regularization','local_highpass_octaves','hf_pair_tolerance_deg','hf_transition_max_gd_us','composition_max_phase_deg','notch_max_added_phase_deg']:
        if not np.isfinite(c[key]) or c[key]<=0:
            raise ValueError(f'{key} 必须为有限正数')
    if not 0<c['lf_local_high_hz']<c['lf_stop_hz'] or c['max_fft_samples']<2*c['taps']:
        raise ValueError('低频局部上限或 FFT 资源上限不正确')
    for ch in channels:
        name=ch.get('name','')
        if not name or len(name)>32 or not all(x.isalnum() or x in '_-' for x in name) or name in labels:
            raise ValueError('声道名必须唯一，只含文字、数字、下划线或短横线')
        labels.add(name)
        q=Path(ch['path']);ch['path']=str((p.parent/q).resolve() if not q.is_absolute() else q.resolve())
        ident=(sha(ch['path']),ch.get('input_channel',0))
        if ident in hashes:
            raise ValueError('不同声道指向了完全相同的数据；请核对左右导出，不能把复制文件当作独立测量。')
        hashes.add(ident)
        ch.setdefault('kind','auto');ch.setdefault('min_usable_hz',c['min_usable_hz'])
        ch.setdefault('measurement_valid_low_hz',c['measurement_valid_low_hz'])
        cutoff=ch['measurement_valid_low_hz']
        if cutoff is not None:
            if not np.isfinite(cutoff) or not 0<cutoff<ch.get('max_usable_hz',c['independent_end_hz']):
                raise ValueError('有效测量低频下限必须为有限正数且低于声道上限')
            ch['min_usable_hz']=max(ch['min_usable_hz'],cutoff)
        ch.setdefault('max_usable_hz',120. if ch.get('role')=='lfe' else c['independent_end_hz'])
        if not np.isfinite(ch['min_usable_hz']) or not 0<ch['min_usable_hz']<ch['max_usable_hz']:
            raise ValueError('声道最低可用频率必须低于声道上限')
        if not np.isfinite(ch['max_usable_hz']) or not ch['min_usable_hz']<ch['max_usable_hz']<c['sample_rate']/2:
            raise ValueError('声道最高可用频率不正确')
        if ch.get('calibration'):
            q=Path(ch['calibration']['path']);ch['calibration']['path']=str((p.parent/q).resolve())
    pairs=c.get('pairs',[]);used=set()
    for pair in pairs:
        if len(pair)!=2 or pair[0]==pair[1] or not set(pair)<=labels or set(pair)&used:
            raise ValueError('每个高频配对须包含两个不同且未重复配对的声道')
        used.update(pair)
    if set(c.get('unprocessed_channels',[]))&labels:
        raise ValueError('未处理声道不能与输入声道重名')
    unprocessed=c.get('unprocessed_channels',[])
    if len(set(unprocessed))!=len(unprocessed) or any(not n or not all(x.isalnum() or x in '_-' for x in n) for n in unprocessed):
        raise ValueError('未处理声道名必须有效且唯一')
    # 多支路系统必须明确挂载点，不能猜测共享低音炮路由。
    if c.get('bass_management') and c.get('mount') not in ('before_bass_management','common_all_branches'):
        raise ValueError('有低音管理时，先确认 FIR 位于低音重定向之前，或同一输入的所有支路共用该 FIR。')
    c['delay_samples']=min(c['taps']//2,int(np.floor(c['max_latency_ms']*c['sample_rate']/1000+1e-8)))
    loaded_names=set()
    for item in c.get('loaded_filters',[]):
        q=Path(item['path']);item['path']=str((p.parent/q).resolve() if not q.is_absolute() else q.resolve())
        if sha(item['path'])!=item.get('sha256'):
            raise ValueError('已加载 FIR 的 SHA256 不符；不能猜测实际使用版本')
        mapping=item.get('channels',[])
        if not mapping or set(mapping)&loaded_names or not set(mapping)<=labels:
            raise ValueError('已加载 FIR 声道映射重复、缺失或未知')
        loaded_names.update(mapping)
        if item.get('delay_samples')!=c['delay_samples']:
            raise ValueError('旧 FIR 必须声明与新 FIR 相同的公共延迟；不自动改变系统延迟')
    if c.get('loaded_filters') and loaded_names!=labels:
        raise ValueError('每个被校准声道都需要实际已加载 FIR 的映射')
    c['range_basis']=dict(min_usable_source='request' if 'min_usable_hz' in r else 'default',
        core_high_source='request' if 'lf_local_high_hz' in r else 'default',
        stop_source='request' if 'lf_stop_hz' in r else 'default',
        movement_role='local_detail_limit_and_soft_weight_not_global_limit',
        local_limit_policy='min(local_detail_cap_hz,movement_estimate_hz)',
        global_policy='declared_min_usable_to_independent_end_hz',spatial_validation=False)
    return c
