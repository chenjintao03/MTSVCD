from pathlib import Path
import os,sys,argparse,time,hashlib,traceback,io
from common_yuv import ROOT,atomic,read,sha,job_env,planes,distortion,mean_metrics
job_env()
p=argparse.ArgumentParser();p.add_argument('--method',choices=['rt','tcm'],required=True);p.add_argument('--gpu',type=int,choices=[4,5],required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--rates',type=int,nargs='+',default=[0,1,2,3]);a=p.parse_args()
uuid={4:'@GPU4_UUID@',5:'@GPU5_UUID@'}[a.gpu]
os.environ['CUDA_VISIBLE_DEVICES']=uuid
if a.method=='rt':
 OLD=Path('@WORKSPACE@/rt_original');sys.path[:0]=[str(OLD/'vendor'),str(OLD/'implementation')]
else:
 OLD=Path('@WORKSPACE@/tcm_original');sys.path.insert(0,str(OLD/'implementation/tcm'))
import numpy as np
import torch
torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
assert torch.cuda.device_count()==1
device=torch.device('cuda:0')
assert str(torch.cuda.get_device_properties(device).uuid).lower().replace('gpu-','')==uuid.lower().replace('gpu-','')
if a.method=='rt':
 import test_video as rt
 from src.models.image_model import DMCI
 from src.models.video_model import DMC
 from src.utils.common import set_torch_env,get_state_dict
 from src.utils.stream_helper import SPSHelper,NalType,read_header,read_sps_remaining,read_ip_remaining
 from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE
 set_torch_env();assert CUSTOMIZED_CUDA_INFERENCE
else:
 from src.models.priors import IntraNoAR
 from src.models.video_net_dmc import DMC
 from src.utils.stream_helper import decode_i,decoder_p,get_downsampled_shape
 # Use the frozen RT BT.709 full-code-range transforms for the RGB-only codec.
 import importlib.util
 spec=importlib.util.spec_from_file_location('colour',ROOT/'references/rt_transforms.py');colour=importlib.util.module_from_spec(spec);spec.loader.exec_module(colour)
manifest=read(ROOT/'protocol/input_manifest.json')['clips'];clips=manifest[:1] if a.pilot else manifest
phase='pilot' if a.pilot else 'full';statuspath=ROOT/f'{a.method}_{phase}_gpu{a.gpu}.json'
frozen_sources=read(ROOT/'protocol/source_sha256.json')[a.method] if (ROOT/'protocol/source_sha256.json').exists() else {}
assert all(sha(p)==h for p,h in frozen_sources.items())
status=dict(status='running',pid=os.getpid(),method=a.method,phase=phase,gpu=a.gpu,gpu_uuid=uuid,done=0,total=len(clips)*len(a.rates))
def update(**kw):status.update(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(statuspath,status)
def load(q):
 if a.method=='rt':
  nets=[]
  for cls,name,expected in [(DMCI,'image','555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9'),(DMC,'video','b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6')]:
   w=OLD/'checkpoints'/f'cvpr2025_{name}.pth.tar';assert sha(w)==expected
   net=cls();net.load_state_dict(get_state_dict(str(w)),strict=True);net=net.to(device).eval();net.update(.12);net.half();nets.append(net)
  return nets
 W=Path('@TCM_WEIGHTS@')
 def state(path):
  d=torch.load(path,map_location='cpu',weights_only=False);return d.get('state_dict',d)
 iw=W/f'intra_psnr_TMM_q{q+1}.pth.tar';pw=W/f'inter_psnr_TMM_q{q+1}.pth'
 intra=IntraNoAR.from_state_dict(state(iw)).eval().to(device);inter=DMC();inter.load_dict(state(pw));inter=inter.eval().to(device)
 intra.update(force=True);inter.update(force=True)
 atomic(ROOT/f'protocol/tcm_weights_q{q}.json',dict(i=str(iw),p=str(pw),i_sha256=sha(iw),p_sha256=sha(pw)))
 return intra,inter
def digest(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def independent_rt(bitpath,expected):
 intra,inter=load(0);intra.set_use_two_entropy_coders(False);inter.set_use_two_entropy_coders(False);inter.set_curr_poc(0)
 stream=io.BytesIO(bitpath.read_bytes());helper=SPSHelper()
 with torch.no_grad():
  for n in range(96):
   h=read_header(stream)
   while h['nal_type']==NalType.NAL_SPS:helper.add_sps_by_id(read_sps_remaining(stream,h['sps_id']));h=read_header(stream)
   s=helper.get_sps_by_id(h['sps_id']);qp,payload=read_ip_remaining(stream)
   assert (h['nal_type']==NalType.NAL_I)==(n%32==0)
   if n%32==0:dec=intra.decompress(payload,s,qp);inter.clear_dpb();inter.add_ref_frame(None,dec['x_hat'])
   else:
    if s['use_ada_i']:inter.reset_ref_feature()
    dec=inter.decompress(payload,s,qp)
   assert digest(dec['x_hat'][:,:,:384,:640])==expected[n],n
 assert stream.tell()==len(stream.getbuffer())
 del intra,inter;torch.cuda.empty_cache()
def point_rt(c,q,intra,inter):
 qp=[0,21,42,63][q];bit=ROOT/f'bitstreams/rt/q{q}/{c["clip"]}.bin';bit.parent.mkdir(parents=True,exist_ok=True)
 log=ROOT/f'logs/raw/rt/q{q}/{c["clip"]}.json';log.parent.mkdir(parents=True,exist_ok=True)
 kwargs=dict(src_type='yuv420',src_path=str(ROOT/'inputs/yuv420'/f'{c["clip"]}.yuv'),src_width=640,src_height=384,frame_num=96,intra_period=32,qp_i=qp,qp_p=qp,reset_interval=64,calc_ssim=True,write_stream=True,check_existing=False,save_decoded_frame=False,verbose=0,verbose_json=True,curr_bin_path=str(bit),curr_json_path=str(log))
 rows=[];hashes=[];original=rt.get_distortion
 def callback(cfg,x_hat,y,u,v,rgb):
  ps,ss=original(cfg,x_hat,y,u,v,rgb);rows.append(dict(frame=len(rows),psnr=ps[0],psnr_y=ps[1],psnr_u=ps[2],psnr_v=ps[3],ms_ssim=ss[0],ms_ssim_y=ss[1],ms_ssim_u=ss[2],ms_ssim_v=ss[3]))
  if a.pilot:hashes.append(digest(x_hat))
  if len(rows)%16==0:update(current=c['clip'],rate=q,frame=len(rows))
  return ps,ss
 rt.get_distortion=callback
 try:
  with torch.no_grad():native=rt.run_one_point_with_stream(inter,intra,kwargs)
 finally:rt.get_distortion=original
 assert len(rows)==96 and native['i_frame_num']==3 and native['p_frame_num']==93
 raw=io.BytesIO(bit.read_bytes());helper=SPSHelper();n=0;offset=0
 while raw.tell()<len(raw.getbuffer()):
  h=read_header(raw)
  if h['nal_type']==NalType.NAL_SPS:helper.add_sps_by_id(read_sps_remaining(raw,h['sps_id']));continue
  s=helper.get_sps_by_id(h['sps_id']);coded,payload=read_ip_remaining(raw);is_i=h['nal_type']==NalType.NAL_I
  assert is_i==(n%32==0);expected=qp if is_i else int(inter.shift_qp(qp,[0,1,0,2,0,2,0,2][n%8]));assert coded==expected
  rows[n].update(type='I' if is_i else 'P',qp=coded,bytes=raw.tell()-offset,use_ada_i=s['use_ada_i']);offset=raw.tell();n+=1
 assert n==96 and sum(r['bytes'] for r in rows)==bit.stat().st_size
 assert abs(native['ave_all_frame_bpp']-bit.stat().st_size*8/(96*640*384))<1e-12
 if a.pilot:independent_rt(bit,hashes)
 return rows,[dict(path=str(bit),sha256=sha(bit),bytes=bit.stat().st_size)]
def point_tcm(c,q,intra,inter):
 source=ROOT/'inputs/yuv420'/f'{c["clip"]}.yuv';raw=np.fromfile(source,dtype=np.uint8).reshape(96,-1)
 folder=ROOT/f'bitstreams/tcm/q{q}/{c["clip"]}';folder.mkdir(parents=True,exist_ok=True)
 ref=feature=None;rows=[];streams=[];hashes=[]
 with torch.no_grad():
  for n in range(96):
   y,u,v=planes(raw[n]);uv=np.stack([u,v]);yuv=colour.ycbcr420_to_444_np(y[None],uv)
   t=torch.from_numpy(yuv.copy()).unsqueeze(0).to(device=device,dtype=torch.float32)/255
   x=torch.round(colour.ycbcr2rgb(t)*255).clamp(0,255)/255
   bit=folder/f'{n:03d}.bin'
   if n%32==0:r=intra.encode_decode(x,str(bit),pic_height=384,pic_width=640);feature=None
   else:r=inter.encode_decode(x,ref,feature,str(bit),pic_height=384,pic_width=640);feature=r['feature']
   ref=r['x_hat'].clamp_(0,1);assert torch.isfinite(ref).all()
   yr,uvr=colour.yuv_444_to_420(colour.rgb2ycbcr(ref));yr=(yr*255).clamp(0,255)[0,0].cpu().numpy();uvr=(uvr*255).clamp(0,255)[0].cpu().numpy()
   row=dict(frame=n,type='I' if n%32==0 else 'P',bytes=bit.stat().st_size,**distortion((y,u,v),(yr,uvr[0],uvr[1])))
   assert r['bit']==bit.stat().st_size*8
   rows.append(row);streams.append(dict(path=str(bit),sha256=sha(bit),bytes=bit.stat().st_size))
   if a.pilot:hashes.append(digest(ref))
   if n%16==15:update(current=c['clip'],rate=q,frame=n+1)
 if a.pilot:
  di,dp=load(q);ref=feature=None
  with torch.no_grad():
   for n in range(96):
    bit=folder/f'{n:03d}.bin'
    if n%32==0:
     h,w,ys,zs=decode_i(str(bit));r=di.decompress([[ys],[zs]],get_downsampled_shape(h,w,64));feature=None
    else:r=dp.decompress(ref,feature,decoder_p(str(bit)),384,640);feature=r['feature']
    ref=r['x_hat'].clamp_(0,1);assert digest(ref)==hashes[n],n
  del di,dp;torch.cuda.empty_cache()
 return rows,streams
try:
 update()
 for q in a.rates:
  intra,inter=load(q)
  for c in clips:
   output=ROOT/f'results/{a.method}/q{q}/{c["clip"]}.json'
   if output.exists():
    r=read(output);assert r['input_sha256']==c['sha256'] and r['protocol_sha256']==sha(ROOT/'protocol/benchmark.json')
    assert r['frames']==len(r['per_frame'])==96 and r['i_positions']==[0,32,64]
    assert all(sha(v['path'])==v['sha256'] for v in r['streams'])
    update(done=status['done']+1);continue
   path=ROOT/'inputs/yuv420'/f'{c["clip"]}.yuv';assert sha(path)==c['sha256']
   t=time.time();update(current=c['clip'],rate=q,frame=0)
   rows,streams=(point_rt if a.method=='rt' else point_tcm)(c,q,intra,inter)
   total=sum(v['bytes'] for v in streams)
   r=dict(method=a.method,rate=q,clip=c['clip'],camera=c['camera'],frames=96,i_positions=[0,32,64],true_bitstream=True,independent_decode_checked=a.pilot,input_sha256=c['sha256'],protocol_sha256=sha(ROOT/'protocol/benchmark.json'),gpu_uuid=uuid,total_seconds=time.time()-t,bitstream_bytes=total,bpp=total*8/(96*640*384),per_frame=rows,streams=streams,**mean_metrics(rows))
   atomic(output,r);update(done=status['done']+1);print(a.method,q,c['clip'],round(r['psnr'],3),flush=True)
  del intra,inter;torch.cuda.empty_cache()
 assert all(sha(p)==h for p,h in frozen_sources.items())
 update(status='complete',original_source_unchanged=True)
except Exception:update(status='failed',error=traceback.format_exc());raise
