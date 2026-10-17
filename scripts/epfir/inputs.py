"""只读数值导入。输入中的注释和说明不作为指令执行。"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import power2,sha

def text_file(path):
    data=Path(path).read_bytes()
    for encoding in ('utf-8-sig','gb18030'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError('文本编码无法识别')

def read_numeric(path):
    lines=text_file(path).splitlines();rows=[];headers=[];meta={}
    for lineno,line in enumerate(lines,1):
        line=line.strip()
        if not line:
            continue
        if '//' in line:
            v,k=line.split('//',1)
            try:
                meta[k.strip().lower()]=float(v.strip())
            except ValueError:
                headers.append(line)
            continue
        if line.startswith(('*','#','%')) or not re.match(r'^[+\-.0-9]',line):
            headers.append(line);continue
        try:
            row=[float(x) for x in re.split(r'[,;\s]+',line)]
        except ValueError as exc:
            raise ValueError(f'第 {lineno} 行不是可识别数值；请导出点号小数的原始文件') from exc
        rows.append(row)
    if not rows or len(set(map(len,rows)))!=1:
        raise ValueError('数值列数不一致或没有数据')
    a=np.asarray(rows,dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('输入包含 NaN 或无穷值')
    return a,headers,meta

def inspect(path):
    suffix=Path(path).suffix.lower()
    if suffix in ('.wav','.flac','.aiff','.aif'):
        info=sf.info(path)
        return dict(type='ir',frames=info.frames,sample_rate=info.samplerate,channels=info.channels,
                    sha256=sha(path),note='WAV 本身不携带 REW 的时间参考和处理状态，须另行声明。')
    a,h,m=read_numeric(path)
    is_ir=any('impulse response' in x.lower() for x in h)
    result=dict(type='ir' if is_ir else ('frequency_response' if a.shape[1]>=3 else 'unknown'),
                rows=len(a),columns=a.shape[1],sha256=sha(path),has_rew_ir_metadata=is_ir)
    if a.shape[1]>=3 and not is_ir:
        f=a[:,0]
        if np.any(f<=0) or np.any(np.diff(f)<=0):
            raise ValueError('频率须为严格递增的正数')
        result.update(frequency_hz=[float(f[0]),float(f[-1])],magnitude_span_db=float(np.ptp(a[:,1])),
                      phase_may_be_excess=bool(np.ptp(a[:,1])<.2),
                      note='三列文件须明确是实测总相位还是已计算的剩余相位；不按文件名猜测。')
    return result

def load(ch,max_fft):
    path=ch['path'];kind=ch.get('kind','auto');meta=dict(sha256=sha(path))
    if Path(path).suffix.lower() in ('.wav','.flac','.aiff','.aif'):
        h,fs=sf.read(path,dtype='float64',always_2d=True)
        index=ch.get('input_channel',0)
        if not 0<=index<h.shape[1]:
            raise ValueError('输入 WAV 声道索引不存在')
        if kind not in ('auto','ir'):
            raise ValueError('WAV 输入应声明为 ir')
        h=h[:,index];meta.update(source_type='ir',sample_rate=fs,reconstructed=False,
            time_origin_s=ch.get('time_origin_s'),timing_reference=ch.get('timing_reference','unknown'),
            capture_clipping_status='unknown_from_exported_ir')
        return dict(kind='ir',samples=h,fs=float(fs),meta=meta)
    a,headers,m=read_numeric(path);head='\n'.join(headers).lower()
    if 'impulse response' in head:
        if 'ir is the min phase version' in head or 'minimum phase version: yes' in head:
            raise ValueError('这是最小相位 IR；需要原始实测 IR 才能提取剩余相位')
        if a.shape[1]!=1:
            raise ValueError('标准 REW IR 文本应为单列样本并保留头部元数据')
        interval=next((v for k,v in m.items() if 'sample interval' in k),None)
        if interval is None or interval<=0:
            raise ValueError('REW IR 文本缺少采样间隔')
        h=a[:,0];fs=1/interval
        length=next((v for k,v in m.items() if 'response length' in k),len(h))
        if int(length)!=len(h):
            raise ValueError('IR 头部长度与样本数不符，可能被截断')
        scale=next((v for k,v in m.items() if 'peak value before' in k),None)
        if 'ir is normalised' in head or 'ir is normalized' in head:
            if scale is not None:
                h=h*scale
        meta.update(source_type='ir',sample_rate=fs,reconstructed=False,
            time_origin_s=next((v for k,v in m.items() if 'start time' in k),None),
            rew_peak_index=next((v for k,v in m.items() if 'peak index' in k),None),
            timing_reference=ch.get('timing_reference','unknown'),
            export_window_applied='window has been applied' in head)
        return dict(kind='ir',samples=h,fs=fs,meta=meta)
    if a.shape[1]<3:
        raise ValueError('需要原始 IR，或包含频率、dB、相位角三列的频响')
    f,mag,phase=a[:,:3].T
    if np.any(f<=0) or np.any(np.diff(f)<=0):
        raise ValueError('频率须为严格递增的正数')
    if kind=='auto':
        raise ValueError('三列频响须显式设置 kind=frd（总相位）或 kind=excess（已有剩余相位），避免重复分解。')
    if kind=='excess':
        meta.update(source_type='excess',reconstructed=False,timing_reference='already_processed',
                    sample_rate=ch.get('measurement_sample_rate'),capture_clipping_status='unavailable')
        return dict(kind=kind,f=f,mag=mag,phase=np.deg2rad(phase),phase_unwrapped=ch.get('phase_unwrapped',bool(np.max(abs(phase))>180.01)),meta=meta)
    if kind!='frd':
        raise ValueError('频响文本 kind 应为 frd 或 excess')
    # 明确声明的原始全通测量可以平直，不能仅凭幅度平直推翻类型声明。
    meta['flat_magnitude_type_declared']=bool(np.ptp(mag)<.2)
    fs=ch.get('measurement_sample_rate')
    delta=float(np.median(np.diff(f)));linear=np.max(abs(f-np.arange(1,len(f)+1)*delta))<.03*delta
    if not linear and not ch.get('phase_unwrapped',False):
        raise ValueError('非线性/稀疏网格的包裹总相位可能已发生时间混叠；请导出原生 IR、密集线性频点，或真正未包裹相位并声明 phase_unwrapped=true。')
    if fs is None:
        standard=np.array([44100,48000,88200,96000,176400,192000],float)
        candidate=standard[np.argmin(abs(standard-2*f[-1]))]
        if not linear or abs(candidate/2-f[-1])>max(.1,2*delta):
            raise ValueError('该 FRD 无法可靠推断采样率，请声明 measurement_sample_rate')
        fs=float(candidate);meta['sample_rate_inferred']=True
    lowest=float(ch.get('min_usable_hz',40))
    if f[0]>lowest/16 or f[-1]<.499*fs:
        raise ValueError('FRD 未保留足够的低频/奈奎斯特边界，带外延伸可能伪造剩余相位；请改用原生 IR，或导出完整原始 FFT 频带（首频点不高于最低可用频率的 1/16，高端接近采样率的一半）。')
    if linear:
        nfft=int(round(fs/delta))
        if nfft%2 or nfft<64 or abs(nfft-power2(nfft))>.02*nfft:
            nfft=power2(nfft)
    else:
        nfft=262144
    if nfft>max_fft:
        raise ValueError('频响重建所需 FFT 超出资源上限')
    if nfft*2>max_fft:
        raise ValueError('频响重建的 FFT 加倍检查超出资源上限')
    ff=np.fft.rfftfreq(nfft,1/fs)
    # 只为数值稳定减去一个公共幅度常数；不据此更改播放电平。
    band=(f>=200)&(f<=3000)
    scale=float(np.median(mag[band] if band.any() else mag))
    mdb=np.interp(ff,f,mag,left=mag[0],right=mag[-1])-scale
    unwrapped=np.deg2rad(phase) if ch.get('phase_unwrapped',False) else np.unwrap(np.deg2rad(phase))
    pp=np.interp(ff,f,unwrapped,left=unwrapped[0],right=unwrapped[-1])
    H=10**(mdb/20)*np.exp(1j*pp)
    H[0]=abs(H[0]);H[-1]=abs(H[-1])*np.sign(np.cos(pp[-1]))
    samples=np.fft.irfft(H,nfft)
    meta.update(source_type='frd',reconstructed=True,sample_rate=fs,
        spectral_scale_db=scale,frequency_hz=[float(f[0]),float(f[-1])],
        source_frequency_step_hz=delta,grid_is_linear=bool(linear),
        time_origin_s=None,timing_reference=ch.get('timing_reference','unknown'),
        capture_clipping_status='unavailable',
        reconstruction_note='从频响重建的周期脉冲；不是原始采集 IR，不能认证采集削波或绝对到达时间。')
    return dict(kind='frd',samples=samples,fs=float(fs),f=f,mag=mag,phase=unwrapped,meta=meta)
