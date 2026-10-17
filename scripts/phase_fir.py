"""本地 REW 剩余相位工具：无联网、无自动播放或声卡修改。"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from epfir.common import request,write_json,sha,clean,smooth
from epfir.inputs import inspect
from epfir.analysis import analyze,frozen_phase
from epfir.model import fit,pair_models
from epfir.design import design,transfer
from epfir.validation import audit
from epfir.report import build
from epfir.multiscale import prepare

def save_arrays(analyses,out):
    arrays={}
    for a in analyses:
        for key in ['f','phase','phase_raw','normalized_phase','nuisance_phase','confidence','valid','smoothing_width','smoothing_selection','smoothing_input_valid','vector_coherence','smooth_candidates','magnitude_db','uncertainty_rad','h','baseline_h','predicted_phase','predicted_gd_ms','baseline_predicted_phase','baseline_predicted_gd_ms','selected_global_smooth_phase','selected_local_phase','loaded_h','residual_target','actual_residual_phase','notch_mask']:
            if key in a:
                arrays[a['name']+'__'+key]=a[key]
    np.savez_compressed(Path(out)/'analysis.npz',**arrays)

def run(mode,path,out):
    c=request(path);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('输出目录已有滤波清单，请为这次生成选择新目录')
    analyses=[]
    for ch in c['channels']:
        print('分析 '+ch['name'],flush=True)
        analyses.append(prepare(fit(analyze(ch,c),c),c))
    pair_models(analyses,c)
    if mode=='analyze':
        save_arrays(analyses,out);build(analyses,c,out);return dict(report=str(out/'report.html'))
    from epfir.unified import design_group
    lookup={a['name']:a for a in analyses}
    paired=set(sum(c.get('pairs',[]),[]))
    groups=c.get('pairs',[])+[[a['name']] for a in analyses if a['name'] not in paired]
    for group in groups:
        print('联合生成并筛选 '+','.join(group),flush=True)
        design_group([lookup[n] for n in group],c)
    lookup={a['name']:a for a in analyses}
    save_arrays(analyses,out)
    paired=set(sum(c.get('pairs',[]),[]));groups=c.get('pairs',[])+[[a['name']] for a in analyses if a['name'] not in paired]
    identity=c.get('unprocessed_channels',[])
    coeff={a['name']:a['h'] for a in analyses}
    for name in identity:
        h=np.zeros(c['taps'],np.float32);h[c['delay_samples']]=1;coeff[name]=h;groups.append([name])
    settings={k:v for k,v in c.items() if k not in ['channels','pairs','unprocessed_channels']}
    manifest=dict(version=2,role='complete_replacement' if c.get('loaded_filters') else 'complete',replaces=c.get('loaded_filters',[]),status='candidate',settings=settings,pairs=c.get('pairs',[]),identity_channels=identity,
                  channels={a['name']:dict(min_usable_hz=a['min_usable_hz'],model=a['model'],source=a['metadata'],design=a['design_metrics'],low_reference=a['metadata'].get('low_reference'),
                        calibration_ranges=a['calibration_ranges'],global_regularization=a['global_regularization']) for a in analyses},
                  files=[],mount=c.get('mount','full_channel_series'),
                  note='每个逻辑声道只应用一次完整滤波；延迟表不负责系统物理对齐。')
    manifest['conservative_common_attenuation_db']=float(np.ceil(max(a['ringing']['l1_peak_bound_db'] for a in analyses)*10)/10)
    manifest['headroom_note']='滤波未作电平衰减；该 L1 上界给出对任意满幅输入保证不削波的保守共同衰减量。实际音乐可用数字峰值测试决定较小的共同余量。'
    for group in groups:
        file='_'.join(group)+f"_complete_{c['sample_rate']}Hz_{c['taps']}_float32.wav"
        sf.write(out/file,np.column_stack([coeff[n] for n in group]),c['sample_rate'],subtype='FLOAT',format='WAV')
        manifest['files'].append(dict(file=file,channels=group,sha256=sha(out/file),role='complete',delay_samples=c['delay_samples'],
                                      delay_ms=c['delay_samples']/c['sample_rate']*1000))
    write_json(out/'manifest.json',manifest)
    result=audit(out)
    manifest['status']='accepted' if result['passed'] else 'rejected';write_json(out/'manifest.json',manifest)
    build(analyses,c,out,result)
    if not result['passed']:
        raise ValueError('最终 WAV 未通过独立验收：'+'；'.join(result['failures']))
    return dict(passed=True,output=str(out),channels=result['channels'])

def compare(path,reference,out):
    """应用后复测：固定原先平滑宽度，只报告观察，不再生成第二层补偿。"""
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import least_squares
    c=request(path,remeasure=True);ref=Path(reference);old=np.load(ref/'analysis.npz',allow_pickle=False)
    manifest=json.loads((ref/'manifest.json').read_text(encoding='utf-8'));original=manifest['settings']
    if manifest.get('version')!=2:
        old.close()
        raise ValueError('compare 需要 v2 配套参考清单；不能混用 v1 报告规约')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);result={}
    for ch in c['channels']:
        name=ch['name']
        model=manifest['channels'][name].get('low_reference')
        if model and model.get('models'):
            ch={**ch,'frozen_low_reference':model,'measurement_valid_low_hz':model['cutoff_hz']}
        for setting in ['smoothing_octaves','adaptive_smoothing','window_pre_ms','window_post_ms']:
            c[setting]=original[setting]
        a=analyze(ch,c);f=old[name+'__f'];width=old[name+'__smoothing_width']
        phi=np.interp(f,a['f'],a['phase_raw'])
        if name+'__smoothing_input_valid' not in old:
            raise ValueError('该参考包使用旧版标量平滑，需要先重新分析，不能混用验收规约。')
        selection=old[name+'__smoothing_selection'];input_valid=old[name+'__smoothing_input_valid']
        chosen=frozen_phase(f,phi,input_valid,selection,original)[0]
        conf=np.minimum(old[name+'__confidence'],np.interp(f,a['f'],a['confidence']))
        predicted=transfer(old[name+'__h'],original,262144)
        expected=old[name+'__predicted_phase']
        fitband=(f>=500)&(f<=8000)&(conf>.2)
        if fitband.sum()<20:
            fitband=(conf>.2)&(f>=ch['min_usable_hz'])&(f<=ch['max_usable_hz'])
        def timing_error(v):
            e=chosen[fitband]-expected[fitband]-2*np.pi*f[fitband]*v[0]*1e-6
            return np.r_[np.sin(e),1-np.cos(e)]
        grid=np.linspace(-1000.,1000.,2001)
        seed=min(grid,key=lambda t:float(np.mean(timing_error([t])**2)))
        timing=least_squares(timing_error,[seed],bounds=(-1000.,1000.),loss='soft_l1',f_scale=.1).x[0]
        chosen=frozen_phase(f,phi-2*np.pi*f*timing*1e-6,input_valid,selection,original)[0]
        gd=-np.gradient(chosen,f)/(2*np.pi)*1000;mask=(f>=max(40,ch['min_usable_hz']))&(f<=300)&(conf>.12)
        gd0=-np.gradient(old[name+'__normalized_phase'],f)/(2*np.pi)*1000
        result[name]=dict(before_rms_ms=float(np.sqrt(np.mean(gd0[mask]**2))),remeasured_rms_ms=float(np.sqrt(np.mean(gd[mask]**2))),
                          analysis_reference_difference_us=float(timing),
                          note='对齐预测和复测的线性时间规约；这不是实际声道延迟测量，不用于播放延迟设置。')
        bands={}
        for lo,hi in [(ch['min_usable_hz'],80),(80,300),(300,400),(400,1000),(1000,8000),(8000,10000),(10000,18000)]:
            band=(f>=lo)&(f<=hi)&(conf>.12)&input_valid
            if band.sum()<5:continue
            circular=np.rad2deg(np.angle(np.exp(1j*chosen[band])))
            error=np.rad2deg(np.angle(np.exp(1j*(chosen[band]-expected[band]))))
            bands[f'{lo:g}-{hi:g}']=dict(valid_points=int(band.sum()),phase_rms_deg=float(np.sqrt(np.mean(circular**2))),
                gd_rms_ms=float(np.sqrt(np.mean(gd[band]**2))),prediction_error_rms_deg=float(np.sqrt(np.mean(error**2))))
        result[name]['bands']=bands
        result[name]['low_reference']=a['metadata'].get('low_reference')
        result[name]['interpretation']='逐频带判断；不能用相位 RMS 改善代替群延时峰谷或空间复测验收'
        np.savetxt(out/(name+'_remeasure.csv'),np.c_[f,gd0,gd,conf,chosen,expected],delimiter=',',header='Hz,before_ms,remeasured_ms,confidence,remeasured_phase_rad,expected_phase_rad',comments='')
    old.close()
    write_json(out/'remeasurement.json',result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect');p.add_argument('files',nargs='+')
    for cmd in ['analyze','design']:
        p=sub.add_parser(cmd);p.add_argument('request');p.add_argument('--out',required=True)
    p=sub.add_parser('verify');p.add_argument('directory')
    p=sub.add_parser('compare');p.add_argument('request');p.add_argument('--reference',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args()
    try:
        if args.command=='inspect':
            result=[inspect(p) for p in args.files]
        elif args.command=='verify':
            result=audit(args.directory)
        elif args.command=='compare':
            result=compare(args.request,args.reference,args.out)
        else:
            result=run(args.command,args.request,args.out)
        print(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False))
        return 1 if isinstance(result,dict) and result.get('passed') is False else 0
    except (ValueError,OSError,KeyError) as exc:
        print('未完成：'+str(exc),file=sys.stderr);return 2

if __name__=='__main__':
    raise SystemExit(main())
