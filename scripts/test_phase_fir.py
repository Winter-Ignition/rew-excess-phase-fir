"""只使用可重建合成数据的行为测试；不依赖个人测量或绝对机器路径。"""
from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from epfir.common import DEFAULTS,request,write_json
from epfir.analysis import analyze,phase_views,frozen_phase
from epfir.inputs import load,read_numeric
from epfir.model import fit,ap,pair_models
from epfir.design import design,render,transfer
from epfir.multiscale import ranges,prepare,local as local_detail
from epfir.validation import audit
from phase_fir import run,compare

class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='excess_phase_synthetic_');self.root=Path(self.temp.name)
        self.c={**DEFAULTS,'delay_samples':8192,'head_half_ranges_cm':[10,10,10]}
    def tearDown(self):
        self.temp.cleanup()
    def measurement(self,name='X',crossovers=(),delay=233,minimum_only=False):
        fs=48000;n=131072;f=np.fft.rfftfreq(n,1/fs);z=np.exp(-2j*np.pi*f/fs)
        H=1/(1-.6*z)
        for fc in crossovers:
            H*=np.exp(1j*ap(f,fc,.707))
        H*=np.exp(-2j*np.pi*f*delay/fs);H[0]=H[0].real;H[-1]=H[-1].real
        path=self.root/(name+'.wav');sf.write(path,np.fft.irfft(H,n),fs,subtype='FLOAT')
        return dict(name=name,path=str(path),kind='ir',min_usable_hz=40,crossover_hz=list(crossovers)),f,H
    def test_minimum_phase_plus_delay_is_zero_excess(self):
        ch,_,_=self.measurement(delay=733)
        a=analyze(ch,self.c);band=(a['f']>=40)&(a['f']<=8000)
        self.assertLess(np.rad2deg(abs(a['phase'][band])).max(),.2)
        self.assertAlmostEqual(a['metadata']['analysis_delay_removed_ms'],733/48,places=2)
    def test_known_allpass_is_recovered(self):
        ch,_,_=self.measurement(crossovers=[2000]);a=fit(analyze(ch,self.c),self.c)
        band=(a['f']>=80)&(a['f']<=7000)
        error=np.angle(np.exp(1j*(a['normalized_phase']-ap(a['f'],2000,.707))))
        self.assertLess(np.rad2deg(np.sqrt(np.mean(error[band]**2))),1.)
        self.assertLess(a['model']['rms_deg'],2.)
    def test_unknown_threeway_model(self):
        ch,_,_=self.measurement(crossovers=[650,4200]);ch['crossover_hz']=[]
        a=fit(analyze(ch,self.c),self.c)
        self.assertLess(a['model']['rms_deg'],4.)
        self.assertGreaterEqual(a['model']['k'],2)
    def test_integer_turns_do_not_change_smoothing(self):
        f=np.geomspace(20,24000,1800);p=ap(f,2000,.7);valid=np.ones(len(f),bool);u=np.zeros(len(f))
        a=phase_views(f,p,valid,u,self.c)[1]
        b=phase_views(f,p+14*np.pi,valid,u,self.c)[1]
        self.assertLess(abs(a-b).max(),1e-10)
    def test_local_turn_steps_and_wrapping_do_not_create_gd_peaks(self):
        f=np.geomspace(20,24000,1800);p=ap(f,2000,.7);v=np.ones(len(f),bool);u=np.zeros(len(f))
        reference=phase_views(f,p,v,u,self.c)[1]
        integer_steps=2*np.pi*((f>150).astype(int)-3*(f>900).astype(int)+2*(f>4000).astype(int))
        stepped=phase_views(f,p+integer_steps,v,u,self.c)[1]
        wrapped=phase_views(f,np.angle(np.exp(1j*p)),v,u,self.c)[1]
        np.testing.assert_allclose(stepped,reference,atol=1e-11)
        np.testing.assert_allclose(wrapped,reference,atol=1e-11)
        self.assertLess(abs(np.gradient(stepped-reference,f)).max(),1e-10)
    def test_frd_and_excess_not_confused(self):
        f=np.geomspace(.5,24000,200);p=self.root/'flat.txt';np.savetxt(p,np.c_[f,np.zeros(len(f)),np.zeros(len(f))])
        with self.assertRaises(ValueError):load(dict(path=str(p),kind='auto'),4194304)
        self.assertEqual(load(dict(path=str(p),kind='frd',measurement_sample_rate=48000,phase_unwrapped=True),4194304)['kind'],'frd')
        self.assertEqual(load(dict(path=str(p),kind='excess'),4194304)['kind'],'excess')
    def test_duplicate_and_routing_guards(self):
        ch,_,_=self.measurement();r=dict(ch);r['name']='Y'
        path=self.root/'request.json';write_json(path,dict(channels=[ch,r]))
        with self.assertRaises(ValueError):request(path)
        write_json(path,dict(channels=[ch],bass_management=True,mount='shared_sub_output'))
        with self.assertRaises(ValueError):request(path)
    def test_nonfinite_input_is_rejected(self):
        p=self.root/'bad.txt';p.write_text('1,0,0\n2,nan,0\n',encoding='utf-8')
        with self.assertRaises(ValueError):read_numeric(p)
    def test_sparse_wrapped_total_phase_is_rejected(self):
        f=np.geomspace(.5,24000,1500);p=self.root/'sparse.txt'
        phi=ap(f,2100,.707)-2*np.pi*f*.012
        np.savetxt(p,np.c_[f,np.zeros(len(f)),np.rad2deg(np.angle(np.exp(1j*phi)))])
        with self.assertRaisesRegex(ValueError,'混叠'):
            load(dict(path=str(p),kind='frd',measurement_sample_rate=48000),4194304)
        np.savetxt(p,np.c_[f,np.zeros(len(f)),np.rad2deg(phi)])
        source=load(dict(path=str(p),kind='frd',measurement_sample_rate=48000,phase_unwrapped=True),4194304)
        np.testing.assert_allclose(source['phase'],phi,atol=1e-12)
    def test_cropped_frd_boundary_is_rejected(self):
        f=np.geomspace(10,22000,1067);p=self.root/'cropped.txt'
        np.savetxt(p,np.c_[f,-20*np.log10(1+(f/18000)**2),np.rad2deg(ap(f,2100,.707)-2*np.pi*f*.012)])
        with self.assertRaisesRegex(ValueError,'边界'):
            load(dict(path=str(p),kind='frd',measurement_sample_rate=48000,phase_unwrapped=True),4194304)
    def test_rew_ir_header(self):
        ch,_,_=self.measurement();h,fs=sf.read(ch['path']);p=self.root/'rew.txt'
        p.write_text('* Impulse Response data\n'+str(1/fs)+' // Sample interval (seconds)\n'+str(len(h))+' // Response length\n'+'\n'.join(map(str,h)),encoding='utf-8')
        ch['path']=str(p);got=load(ch,4194304)
        self.assertAlmostEqual(got['fs'],48000)
        np.testing.assert_array_equal(got['samples'],h)
    def test_shorter_latency_preserves_taps(self):
        ch,_,_=self.measurement(crossovers=[2000]);c={**self.c,'delay_samples':1920,'max_latency_ms':10.}
        a=design(fit(analyze(ch,c),c),c)
        self.assertEqual(len(a['h']),16384)
        self.assertLessEqual(abs(int(np.argmax(abs(a['h'])))-1920),1)
    def test_head_range_limits_local_detail_but_not_global_trend(self):
        ch,_,_=self.measurement(crossovers=[2000]);a=fit(analyze(ch,self.c),self.c)
        c10={**self.c,'head_half_ranges_cm':[10,10,10]};c5={**self.c,'head_half_ranges_cm':[5,5,5]}
        self.assertAlmostEqual(ranges(a,c10)['local_limit_hz'],247.538927915052)
        self.assertEqual(ranges(a,c5)['local_limit_hz'],300.)
        first=prepare(a.copy(),c10);second=prepare(a.copy(),c5)
        np.testing.assert_allclose(first['global_variants'][0]['smooth_target'],second['global_variants'][0]['smooth_target'],atol=1e-12)
        detail=local_detail(first,first['model_phase'],c10)
        self.assertLess(abs(detail[first['f']>=first['calibration_ranges']['local_limit_hz']]).max(),1e-12)
    def test_broad_residual_is_corrected_across_300hz(self):
        ch,_,_=self.measurement(crossovers=[2000]);c={**self.c,'head_half_ranges_cm':[5,5,5]}
        a=fit(analyze(ch,c),c);f=a['f']
        # 合成分析样例：已知分频模型之外，另有跨越 300 Hz 的缓慢相位隆起。
        bump=.3*(2*np.arctan(f/18)-2*np.arctan(f/850))
        a['phase_raw']=a['model_phase']+bump+a['nuisance_phase']
        a['normalized_phase']=frozen_phase(f,a['phase_raw']-a['nuisance_phase'],a['smoothing_input_valid'],a['smoothing_selection'],c)[0]
        prepare(a,c)
        self.assertGreater(abs(np.interp(500,f,a['global_variants'][0]['smooth_target'])),np.deg2rad(3))
        prior=render(f,a['independent_target'],c);tr=transfer(prior,c)
        prior_phase=frozen_phase(f,a['phase_raw']-a['nuisance_phase']+np.interp(f,tr['f'],tr['phase']),a['smoothing_input_valid'],a['smoothing_selection'],c)[0]
        result=design(a,c);band=(f>=400)&(f<=5000)
        self.assertLess(np.sqrt(np.mean(result['predicted_phase'][band]**2)),.7*np.sqrt(np.mean(prior_phase[band]**2)))
        self.assertLess(abs(result['selected_local_phase'][f>=300]).max(),1e-12)
    def test_single_lfe_uses_only_its_band(self):
        ch,_,_=self.measurement(crossovers=[80]);ch.update(role='lfe',crossover_hz=[],max_usable_hz=120.)
        a=fit(analyze(ch,self.c),self.c)
        self.assertFalse(a['valid'][a['f']>=120].any())
        self.assertLess(abs(a['independent_target'][a['f']>=120]).max(),1e-12)
        self.assertLess(a['model']['rms_deg'],8.)
    def test_complete_stereo_identity_and_tamper_audit(self):
        left,_,_=self.measurement('L',[2000]);right,_,_=self.measurement('R',[2300])
        p=self.root/'request.json';write_json(p,dict(channels=[left,right],pairs=[['L','R']],unprocessed_channels=['C']))
        out=self.root/'result';result=run('design',p,out);self.assertTrue(result['passed'])
        m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(m['files']),2);self.assertEqual(m['status'],'accepted')
        item=next(x for x in m['files'] if x['channels']==['L','R']);h,fs=sf.read(out/item['file'],always_2d=True)
        self.assertEqual(h.shape,(16384,2));self.assertEqual(fs,192000)
        measured=[]
        for j,ch in enumerate([left,right]):
            original,native=sf.read(ch['path']);F=np.fft.rfftfreq(len(original),1/native)
            freq=np.fft.rfftfreq(1048576,1/fs);response=np.fft.rfft(h[:,j],1048576)
            at=np.interp(F,freq,response.real)+1j*np.interp(F,freq,response.imag)
            corrected=np.fft.irfft(np.fft.rfft(original)*at,len(original))
            file=self.root/(ch['name']+'_remeasured.wav');sf.write(file,corrected,native,subtype='FLOAT')
            measured.append({**ch,'path':str(file)})
        rp=self.root/'remeasure.json';write_json(rp,dict(channels=measured,pairs=[['L','R']],existing_phase_correction='this_filter'))
        check=compare(rp,out,self.root/'remeasure_result')
        for name in ['L','R']:
            self.assertLess(check[name]['remeasured_rms_ms'],check[name]['before_rms_ms']*.6)
        sf.write(out/item['file'],h*.9,fs,subtype='FLOAT')
        tampered=audit(out,False);self.assertFalse(tampered['passed'])
        self.assertTrue(any('清单' in x for x in tampered['failures']))

if __name__=='__main__':
    unittest.main(verbosity=2)
