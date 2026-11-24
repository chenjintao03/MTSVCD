"""Original RT, native dimensions and96frame windows, genuine streams."""
from common import *
import argparse,io,fcntl,traceback,math
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[4,5],required=True);p.add_argument('--rates',type=int,nargs='+',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();setup(a.gpu)
assert read(TCM/'complete_audit.json')['passed'],'Do not overlap the matched TCM work'
lock=(ROOT/f'gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import torch,numpy as np
import test_video as rt
from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.utils.common import get_state_dict,set_torch_env
from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE
from src.utils.video_reader import YUV420Reader
from src.utils.stream_helper import SPSHelper,NalType,read_header,read_sps_remaining,read_ip_remaining
from native_window import NativeWindow
set_torch_env();assert CUSTOMIZED_CUDA_INFERENCE;torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid).lower().replace('gpu-','')==UUID[a.gpu].lower().replace('gpu-','')
source_identity();manifest=read(ROOT/'protocol/input_manifest.json')['clips'];clips=manifest
if a.pilot:
    chosen={}
    for c in manifest:chosen.setdefault(c['dataset'],c)
    clips=list(chosen.values());assert len(clips)==4
phase='pilot' if a.pilot else 'full';status=dict(status='running',pid=os.getpid(),gpu=a.gpu,phase=phase,done=0,total=len(clips)*len(a.rates))
def update(**kw):status.update(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(ROOT/f'status/{phase}_gpu{a.gpu}.json',status)
def load():
    nets=[]
    for cls,name in [(DMCI,'image'),(DMC,'video')]:
        path=OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(path)==WEIGHTS[name];net=cls();net.load_state_dict(get_state_dict(str(path)),strict=True);net=net.cuda().eval();net.update(.12);net.half();nets.append(net)
    return nets
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def fresh_decode(c,path,expected,rows,kwargs,official_metric):
    intra,inter=load();dual=c['height']*c['width']>1280*720;intra.set_use_two_entropy_coders(dual);inter.set_use_two_entropy_coders(dual);inter.set_curr_poc(0)
    stream=io.BytesIO(path.read_bytes());helper=SPSHelper();reader=NativeWindow(c,YUV420Reader) if a.pilot else None
    with torch.no_grad():
        for n in range(96):
            h=read_header(stream)
            while h['nal_type']==NalType.NAL_SPS:helper.add_sps_by_id(read_sps_remaining(stream,h['sps_id']));h=read_header(stream)
            s=helper.get_sps_by_id(h['sps_id']);qp,payload=read_ip_remaining(stream)
            assert s['height']==c['height'] and s['width']==c['width'] and bool(s['ec_part'])==dual
            assert (h['nal_type']==NalType.NAL_I)==(n%32==0)
            if n%32==0:dec=intra.decompress(payload,s,qp);inter.clear_dpb();inter.add_ref_frame(None,dec['x_hat'])
            else:
                if s['use_ada_i']:inter.reset_ref_feature()
                dec=inter.decompress(payload,s,qp)
            xhat=dec['x_hat'][:,:,:c['height'],:c['width']];assert digest(xhat)==expected[n],(c['clip'],n)
            if reader:
                y,uv=reader.read_one_frame();ps,ss=official_metric(kwargs,xhat,y[0],uv[0],uv[1],None)
                for k,v in zip(['psnr','psnr_y','psnr_u','psnr_v','ms_ssim','ms_ssim_y','ms_ssim_u','ms_ssim_v'],ps+ss):assert abs(v-rows[n][k])<1e-12,(k,v,rows[n][k])
    assert stream.tell()==len(stream.getbuffer())
    if reader:reader.close()
    del intra,inter;torch.cuda.empty_cache()
def point(c,q,intra,inter):
    qp=[0,21,42,63][q];path=ROOT/f'bitstreams/rt/q{q}/{c["clip"]}.bin';path.parent.mkdir(parents=True,exist_ok=True);rawlog=ROOT/f'logs/raw/q{q}/{c["clip"]}.json';rawlog.parent.mkdir(parents=True,exist_ok=True)
    kwargs=dict(src_type='yuv420',src_path=c['source_path'],src_width=c['width'],src_height=c['height'],frame_num=96,intra_period=32,qp_i=qp,qp_p=qp,reset_interval=64,calc_ssim=True,write_stream=True,check_existing=False,save_decoded_frame=False,verbose=0,verbose_json=True,curr_bin_path=str(path),curr_json_path=str(rawlog))
    rows=[];hashes=[];readers=[];original_reader=rt.get_src_reader;original_metric=rt.get_distortion
    def reader_factory(args):
        assert args['src_path']==c['source_path'] and args['src_width']==c['width'] and args['src_height']==c['height'];r=NativeWindow(c,YUV420Reader);readers.append(r);return r
    def metric_callback(args,xhat,y,u,v,rgb):
        ps,ss=original_metric(args,xhat,y,u,v,rgb);rows.append(dict(frame=len(rows),**dict(zip(['psnr','psnr_y','psnr_u','psnr_v','ms_ssim','ms_ssim_y','ms_ssim_u','ms_ssim_v'],ps+ss))));hashes.append(digest(xhat))
        if len(rows)%16==0:update(current=c['clip'],rate=q,frame=len(rows))
        return ps,ss
    rt.get_src_reader=reader_factory;rt.get_distortion=metric_callback
    try:
        with torch.no_grad():native=rt.run_one_point_with_stream(inter,intra,kwargs)
    finally:rt.get_src_reader=original_reader;rt.get_distortion=original_metric
    assert len(rows)==96 and len(readers)==2 and all(r.closed and r.done==96 and r.hasher.hexdigest()==c['sha256'] for r in readers)
    stream=io.BytesIO(path.read_bytes());helper=SPSHelper();n=0;offset=0;dual=c['height']*c['width']>1280*720
    while stream.tell()<len(stream.getbuffer()):
        h=read_header(stream)
        if h['nal_type']==NalType.NAL_SPS:helper.add_sps_by_id(read_sps_remaining(stream,h['sps_id']));continue
        s=helper.get_sps_by_id(h['sps_id']);coded,payload=read_ip_remaining(stream);is_i=h['nal_type']==NalType.NAL_I;assert is_i==(n%32==0) and s['height']==c['height'] and s['width']==c['width'] and bool(s['ec_part'])==dual
        assert coded==(qp if is_i else int(inter.shift_qp(qp,[0,1,0,2,0,2,0,2][n%8])))
        rows[n].update(type='I' if is_i else 'P',qp=coded,bytes=stream.tell()-offset,use_ada_i=s['use_ada_i']);offset=stream.tell();n+=1
    assert n==96 and sum(row['bytes'] for row in rows)==path.stat().st_size
    bpp=path.stat().st_size*8/(96*c['width']*c['height']);assert abs(native['ave_all_frame_bpp']-bpp)<1e-12
    fresh_decode(c,path,hashes,rows,kwargs,original_metric)
    return dict(method='rt',rate=q,base_qp=qp,clip=c['clip'],camera=c['camera'],dataset=c['dataset'],width=c['width'],height=c['height'],nominal_fps_from_filename=c['nominal_fps_from_filename'],start=c['start'],frames=96,i_positions=[0,32,64],true_bitstream=True,independent_decode_checked=True,independent_metric_check=a.pilot,input_sha256=c['sha256'],source_sha256=c['source_sha256'],protocol_sha256=sha(ROOT/'protocol/experiment.json'),gpu_uuid=UUID[a.gpu],source_window_reads=[r.evidence() for r in readers],dual_entropy_coders=dual,produced_phase=phase,bitstream_bytes=path.stat().st_size,bpp=bpp,per_frame=rows,streams=[dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)],**{k:sum(row[k] for row in rows)/96 for k in ['psnr','psnr_y','psnr_u','psnr_v','ms_ssim','ms_ssim_y','ms_ssim_u','ms_ssim_v']})
try:
    update()
    for q in a.rates:
        assert q in [0,1,2,3];intra,inter=load()
        for c in clips:
            path=ROOT/f'results/rt/q{q}/{c["clip"]}.json'
            if path.exists():
                result=read(path);assert result['protocol_sha256']==sha(ROOT/'protocol/experiment.json') and result['input_sha256']==c['sha256'] and result['independent_decode_checked'] and result['frames']==96
                assert all(sha(s['path'])==s['sha256'] for s in result['streams'])
                if a.pilot:assert result['independent_metric_check']
            else:
                update(current=c['clip'],rate=q,frame=0);began=time.time();result=point(c,q,intra,inter);result['total_seconds']=time.time()-began;atomic(path,result)
            update(done=status['done']+1)
        del intra,inter;torch.cuda.empty_cache()
    source_identity();update(status='complete')
except BaseException:update(status='failed',error=traceback.format_exc());raise
