import os
from pathlib import Path
import math
import numpy as np
import onnxruntime as ort
_H=Path(__file__).resolve().parent
ON=os.environ.get('KT_ON','1')=='1'
AV_D=3.0
AV_REL=0.7
AV_ROT=45.0
AV_HOLD=15
AV_T0=100
AV_TYPES=('forest','city')
CFG={
 'forest':   (350,15,1000000000.0,8.0,5.0,40,120,8.0,0,0.75),
 'city':     (450,25,10.0,6.0,5.0,40,250,4.5,1),
 'mountain': (450, 15, 1000000000.0, 12.0, 8.0, 25, 120, 8.0, 0, 0.75),
 'village':  (450,15,10.0,8.0,5.0,40,120,8.0,0,0.75),
}
MTN_STRONG=(450,15,1000000000.0,12.0,8.0,25,120,8.0,0,0.75)
MTN_THR=0.5
TYPES=tuple(t for t in os.environ.get('KT_TYPES','forest,village,city,mountain').split(',') if t)
LB=('city','open','mountain','village','warehouse','forest')
SIM_DT=0.02
VW=float(os.environ.get('KT_VETO_WARM','0.02'))
RP_ON=os.environ.get('KT_RP','1')=='1'
RP_T=float(os.environ.get('KT_RP_T','0.713'))
RP_HMAX=float(os.environ.get('KT_RP_HMAX','2.0'))  # v2 : 1.0 -> 2.0, le filtre h rejetait 17 vraies détections sur 19
RP_T0=150; RP_EVERY=75; RP_CAP=40; RP_CHASE_MIN=12; RP_CHASE_TICKS=36
# v2 : 20 -> 25 m (les fausses détections sondées sont toutes à plus de 26.8 m)
RP_RANGE_MAX=float(os.environ.get('KT_RP_RANGE','25.0')); RP_IMG_MAX=29.0; RP_CLUE_R=55.0; RP_LATCH_N=int(os.environ.get('KT_RP_N','2')); RP_LATCH_TOL=5.0; RP_EMA=0.3
# v2 : une seule détection très sûre suffit à verrouiller (0 = désactivé)
RP_ONE_P=float(os.environ.get('KT_RP_ONE','0.95'))
RP_TERM_R=6.0; RP_HOVER_UP=float(os.environ.get('KT_HOVER','3.9'));  # v5 : 3.2 -> 3.9 (le point verrouillé est au niveau du sol)
DL_HOVER_UP=float(os.environ.get('KT_DL_HOVER','3.6'))  # v5 : le point profondeur est à mi-hauteur de la victime
# v5 : balayage vertical au-dessus de la cible. La hauteur de la victime est connue à +/- 1 m seulement (fenêtre de 2 m) :
# on part de SW_TOP au-dessus du point verrouillé et on descend à SW_V m/s jusqu'à SW_BOT, ce qui fait passer
# plus de 2 s dans la fenêtre 2-4 m, sous 1 m/s, que la victime soit debout ou allongée.
SW_ON=os.environ.get('KT_SWEEP','1')=='1'; SW_TOP=float(os.environ.get('KT_SW_TOP','5.0')); SW_BOT=2.4; SW_V=float(os.environ.get('KT_SW_V','0.45'))
RP_DUD_TICKS=300; RP_AGL_MIN=2.0; RP_RING=1.6; RP_TERM_MAX=1200
# fix 2: descente rapide au-dessus du verrou RGB (avant : 0.45 m/s, et le minuteur « dud » expirait pendant la descente)
TDESC=os.environ.get('KT_TDESC','1')=='1'
RP_LOG=os.environ.get('KT_RPLOG','')  # sonde : journal de chaque image RGB scorée
TD_DZ=0.8; TD_K=0.8; TD_VMAX=1.5; TD_HK=0.5; TD_HMAX=0.45; TD_DUD_DZ=1.0; V_MAX=3.0
_DMIN,_DMAX=0.5,30.0; _CAMF,_CAMU,_HT=0.13,0.05,1.0
def _sig(x):
 return 1.0/(1.0+math.exp(-max(-40.0,min(40.0,float(x)))))
def _axes(rpy):
 r,p,y=(float(rpy[0]),float(rpy[1]),float(rpy[2]))
 cr,sr,cp,sp,cy,sy=(math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y))
 fwd=np.array([cy*cp,sy*cp,-sp],np.float64); up=np.array([cy*sp*cr+sy*sr,sy*sp*cr-cy*sr,cp*cr],np.float64)
 return fwd,up,np.cross(fwd,up)
def _mworld(u,v,cz,pos,rpy):
 fwd,up,right=_axes(rpy); cam=np.asarray(pos,np.float64)+fwd*_CAMF+up*_CAMU
 return cam+right*(u*_HT*cz)+up*(v*_HT*cz)+fwd*cz
CHAMP_ROUTE=os.environ.get('KT_CHAMP_ROUTE','1')=='1'  # village/warehouse -> UID_169's process end to end
ESC_TERR=os.environ.get('KT_ESC_TERR','1')=='1'        # escape acts ONLY on positively-latched mountain/forest
ESC_ON=os.environ.get('KT_ESC','1')=='1'; ESC_TRAP_M=float(os.environ.get('KT_ESC_TRAP','1.0')); ESC_FREE_M=3.0; ESC_DIST=float(os.environ.get('KT_ESC_DIST','2.0'))
# fix 1: warehouse (départ au sol, z=0.191) sous une étagère visible en haut de l'image -> escape avant de monter
TKO=os.environ.get('KT_TKO','1')=='1'; TKO_Z=(0.17,0.21); TKO_TOP_M=0.6
# v3 : en mountain, plafond d'AGL pendant la recherche (le drone survolait les vallées à 20-40 m au-dessus de la victime)
# v6 : verrou profondeur désactivé par défaut (KT_DL=1 pour l'activer) : net négatif en forest/village sur 360 seeds.
# v4 : verrou sur le détecteur de profondeur interne (sorties /detector et pixel_to_camera_mlp exposées dans policy.onnx),
# hors mountain. La sonde montrait le détecteur juste (< 3 m) dans 11 échecs sur 12, sans que la nav s'engage.
DL_ON=os.environ.get('KT_DL','0')=='1'; DL_P=float(os.environ.get('KT_DL_P','0.95')); DL_D=30.0; DL_R=30.0; DL_TOL=3.0; DL_T0=int(os.environ.get('KT_DL_T0','600')); DL_CLEAR_M=3.0; DL_RECONF=250; DL_RECONF_D=8.0; DL_TERM_MAX=750  # v5b : reconfirmation 5 s, limite 15 s  # v4b : p 0.95, rien avant 12 s, pas de surcharge près d'un obstacle
DL_OUT=('/detector/Concat_output_0','/pixel_to_camera_mlp/pixel_to_camera_mlp.6/Gemm_output_0')
AGL_CAP=float(os.environ.get('KT_AGL_CAP','12.0')); AGL_K=0.4; AGL_VMAX=float(os.environ.get('KT_AGL_VMAX','1.5')); AGL_VMIN=0.3; AGL_T0=300; AGL_CLEAR_M=6.0; AGL_ZONE=float(os.environ.get('KT_AGL_ZONE','30.0'))  # v3b : seulement dans le cercle de recherche
ESC_SPEED=0.3; ESC_CLIMB=0.6; ESC_AGL_TARGET=2.5; ESC_MAX_CREEP=300; ESC_MARGIN=1.0; ESC_BODY_M=1.2; ESC_MAX_CLIMB=150; ESC_CLEAR_TICKS=10; ESC_CREEP_AGL=0.28; ESC_SCAN_TICKS=45
class _Escape:
 def __init__(self):
  self.reset()
 def reset(self):
  self.phase=None; self.heading=None; self.start=None; self.clear=0; self.n=0; self.trap=False; self.cycles=0; self.climb_start=None; self.yaw0=0.0; self.scan_i=0; self.scan_n=0; self.margin_start=None; self.guard_hits=0
 def opening(self,depth,yaw):
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  bands=[float(d[112:128,c0:c0+52].min()) for c0 in (0,51,102,153,204)]
  best=max(bands); k=min((i for i in range(5) if bands[i]>=best-0.05), key=lambda i: abs(i-2))
  if bands[k]<ESC_FREE_M:
   return None
  return float(yaw)-math.radians((k-2)*18.0)
 def decide(self,depth,pos,yaw):
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  if float((d[0:16]<0.8).mean())<0.95:
   return False
  self.trap=True; self.start=np.asarray(pos[0:2],np.float64).copy(); self.yaw0=float(yaw)
  h=self.opening(depth,yaw)
  if h is not None:
   self.heading=h; self.phase='creep'; return True
  self.phase='scan'; self.scan_i=0; self.scan_n=0; return True
 def force(self,depth,pos,yaw):
  """Piège imposé (fix 1) : même manœuvre que decide(), sans son seuil de 95 %."""
  self.trap=True; self.start=np.asarray(pos[0:2],np.float64).copy(); self.yaw0=float(yaw)
  h=self.opening(depth,yaw)
  if h is not None:
   self.heading=h; self.phase='creep'
  else:
   self.phase='scan'; self.scan_i=0; self.scan_n=0
 def act(self,depth,pos,agl,pitch_deg,yaw):
  self.n+=1
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  top=float(d[0:64].min())
  if self.phase=='scan':
   tgt=self.yaw0+math.radians((90.0,-90.0,180.0)[self.scan_i]); self.scan_n+=1
   err=(yaw-tgt+math.pi)%(2*math.pi)-math.pi
   if abs(err)<math.radians(8.0) or self.scan_n>=ESC_SCAN_TICKS:
    h=self.opening(depth,yaw) if abs(err)<math.radians(12.0) else None
    if h is not None:
     self.heading=h; self.phase='creep'; self.n=0; self.clear=0; self.start=np.asarray(pos[0:2],np.float64).copy()
     return np.array([math.cos(h),math.sin(h),0.0],np.float32),ESC_SPEED,None
    self.scan_i+=1; self.scan_n=0
    if self.scan_i>=3:
     self.phase='done'; return None
    tgt=self.yaw0+math.radians((90.0,-90.0,180.0)[self.scan_i])
   yt=(tgt+math.pi)%(2*math.pi)-math.pi
   return np.zeros(3,np.float32),0.0,float(yt/math.pi)
  if self.phase=='creep':
   self.clear=self.clear+1 if (top>=2.0 and abs(pitch_deg)<6.0) else 0
   trav=float(np.linalg.norm(np.asarray(pos[0:2],np.float64)-self.start))
   if self.clear>=ESC_CLEAR_TICKS and self.margin_start is None:
    self.margin_start=np.asarray(pos[0:2],np.float64).copy()
   past=(self.margin_start is not None and float(np.linalg.norm(np.asarray(pos[0:2],np.float64)-self.margin_start))>=ESC_MARGIN)
   rel=(self.heading-yaw+math.pi)%(2*math.pi)-math.pi
   kb=min(4,max(0,int(round(2-math.degrees(rel)/18.0)))); c0=(0,51,102,153,204)[kb]
   if float(d[100:128,c0:c0+52].min())<ESC_BODY_M and self.n>15 and self.guard_hits<3:
    self.guard_hits+=1; self.phase='scan'; self.scan_i=0; self.scan_n=0; self.yaw0=float(yaw); self.clear=0; self.margin_start=None
    return np.zeros(3,np.float32),0.0,None
   if past or self.n>=ESC_MAX_CREEP:
    self.phase='climb'; self.n=0; self.climb_start=np.asarray(pos[0:2],np.float64).copy()
   else:
    vz=float(np.clip((ESC_CREEP_AGL-agl)*3.0,-0.5,0.5))
    v=np.array([math.cos(self.heading),math.sin(self.heading),vz],np.float64); v/=max(1e-6,float(np.linalg.norm(v)))
    return v.astype(np.float32),ESC_SPEED,None
  if self.phase=='climb':
   if agl>=ESC_AGL_TARGET or self.n>=ESC_MAX_CLIMB:
    self.phase='done'; return None
   if top<0.9 and agl<1.0 and self.cycles<2:
    self.cycles+=1; self.phase='creep'; self.n=0; self.clear=0; self.margin_start=None; self.start=np.asarray(pos[0:2],np.float64).copy()
    return np.array([math.cos(self.heading),math.sin(self.heading),0.0],np.float32),ESC_SPEED,None
   return np.array([0.0,0.0,1.0],np.float32),ESC_CLIMB,None
  return None
class _RgbPrimary:
 def __init__(self):
  self.sess=None; self._tried=False
  self.reset()
 def _load(self):
  self._tried=True
  try:
   so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
   self.sess=ort.InferenceSession(str(_H/'victim_rgb_m.onnx'),so,providers=['CPUExecutionProvider']); self.iname=self.sess.get_inputs()[0].name
  except Exception:
   self.sess=None
 def reset(self):
  self.n_req=0; self.last_req=-10**9; self.hit_t=-10**9; self.hits=[]; self.latch=None; self.near_since=None; self.bad=[]; self.clue0=None
  self.n_fix=0; self.n_latch=0; self.n_dud=0; self.latch_t=None
  if not hasattr(self,'hover_up'): self.hover_up=RP_HOVER_UP
  self.sw_t0=None; self.sw_for=None; self.now=0; self.sw_h0=SW_TOP
 def score(self,rgb):
  fr=np.asarray(rgb,np.float32)
  if self.sess is None or fr.size!=256*256*3 or float(np.mean(np.abs(fr)))<0.005:
   return None
  r=np.ascontiguousarray(fr.reshape(256,256,3).reshape(128,2,128,2,3).mean(axis=(1,3)),np.float32)
  o=np.asarray(self.sess.run(None,{self.iname:r})[0],np.float32).reshape(5)
  return _sig(o[0]),float(o[1]),float(o[2]),float(o[4])
 def localise(self,u,v,depth,pos,rpy):
  d=np.asarray(depth,np.float32).reshape(256,256)
  px=min(255,max(0,int(round((u+1.0)*0.5*256-0.5)))); py=min(255,max(0,int(round((1.0-v)*0.5*256-0.5))))
  cz=float(np.median(d[max(0,py-2):py+3,max(0,px-2):px+3]))*(_DMAX-_DMIN)+_DMIN
  if cz>RP_IMG_MAX or cz>RP_RANGE_MAX:
   return None
  q=_mworld(u,v,cz,pos,rpy)
  if not np.all(np.isfinite(q)):
   return None
  if self.clue0 is not None and float(np.linalg.norm(q[0:2]-self.clue0))>RP_CLUE_R:
   return None
  if not (-25.0<=float(q[2]-pos[2])<=1.0):
   return None
  for b in self.bad:
   if float(np.linalg.norm(q[0:2]-b))<6.0:
    return None
  return q
 def observe(self,tick,obs,pos,rpy):
  if self.sess is None and not self._tried:
   self._load()
  rgb=obs.get('rgb') if hasattr(obs,'get') else None
  if rgb is None:
   return
  sc=self.score(rgb)
  if sc is None:
   return
  p,u,v,h=sc
  q=self.localise(u,v,obs['depth'],pos,rpy) if not (p<RP_T or h>RP_HMAX) else None
  if RP_LOG:
   self._log(tick,p,u,v,h,obs['depth'],pos,rpy,q)
  if p<RP_T or h>RP_HMAX:
   return
  if q is None:
   return
  self.hit_t=tick; self.n_fix+=1
  if self.latch is not None:
   if float(np.linalg.norm(q[0:2]-self.latch[0:2]))<=6.0:
    self.latch=(1.0-RP_EMA)*self.latch+RP_EMA*q
   return
  self.hits=[(t,z) for t,z in self.hits if tick-t<=400]+[(tick,q)]
  if RP_ONE_P>0 and p>=RP_ONE_P:
   self.latch=q.copy(); self.near_since=None; self.n_latch+=1; self.latch_t=tick; return
  if len(self.hits)>=RP_LATCH_N:
   pts=np.asarray([z for _,z in self.hits[-RP_LATCH_N:]]); c=pts.mean(0)
   if float(np.max(np.linalg.norm(pts[:,0:2]-c[0:2],axis=1)))<=RP_LATCH_TOL:
    self.latch=c; self.near_since=None; self.n_latch+=1; self.latch_t=tick
 def _log(self,tick,p,u,v,h,depth,pos,rpy,q):
  import json
  d=np.asarray(depth,np.float32).reshape(256,256)
  px=min(255,max(0,int(round((u+1.0)*0.5*256-0.5)))); py=min(255,max(0,int(round((1.0-v)*0.5*256-0.5))))
  cz=float(np.median(d[max(0,py-2):py+3,max(0,px-2):px+3]))*(_DMAX-_DMIN)+_DMIN
  qq=_mworld(u,v,cz,pos,rpy)
  rec=dict(tick=int(tick),p=float(p),u=float(u),v=float(v),h=float(h),cz=cz,pos=[float(x) for x in pos],
           q=[float(x) for x in qq],ok=q is not None,clue0=None if self.clue0 is None else [float(x) for x in self.clue0],
           latched=self.latch is not None,n_hits=len(self.hits),bad=len(self.bad))
  with open(os.path.join(RP_LOG,f"rp_{os.getpid()}.jsonl"),"a") as f:
   f.write(json.dumps(rec)+chr(10))
 def add_point(self,tick,q,tol):
  """Verrou profondeur (v4) : même automate que observe(), à partir d'un point monde déjà calculé."""
  for b in self.bad:
   if float(np.linalg.norm(q[0:2]-b))<6.0:
    return
  self.hit_t=tick; self.n_fix+=1
  if self.latch is not None:
   if float(np.linalg.norm(q[0:2]-self.latch[0:2]))<=6.0:
    self.latch=(1.0-RP_EMA)*self.latch+RP_EMA*q; self.conf_t=tick
   return
  self.hits=[(t,z) for t,z in self.hits if tick-t<=400]
  for _,z in self.hits:
   if float(np.linalg.norm(q[0:2]-z[0:2]))<=tol:
    self.latch=0.5*(q+z); self.near_since=None; self.n_latch+=1; self.latch_t=tick; self.conf_t=tick; return
  self.hits.append((tick,q))
 def want_rgb(self,tick):
  if self.sess is None and not self._tried:
   self._load()
  if self.sess is None or self.n_req>=RP_CAP or tick<RP_T0:
   return False
  chase=(tick-self.hit_t<=RP_CHASE_TICKS) and (tick-self.last_req>=RP_CHASE_MIN)
  paced=(tick-self.last_req>=RP_EVERY)
  if chase or paced:
   self.n_req+=1; self.last_req=tick; return True
  return False
 def h_target(self):
  """Hauteur visée au-dessus du point verrouillé (fixe, ou balayage v5)."""
  if not SW_ON:
   return self.hover_up
  if self.sw_for!=self.latch_t:
   self.sw_t0=None; self.sw_for=self.latch_t
  if self.sw_t0 is None:
   return SW_TOP
  return max(SW_BOT,self.sw_h0-SW_V*(self.now-self.sw_t0)*SIM_DT)
 def tick_end(self,tick,pos):
  self.now=tick
  if self.latch is None:
   return
  near=float(np.linalg.norm(self.latch[0:2]-pos[0:2]))<=RP_RING
  if near and SW_ON and (self.sw_t0 is None or self.h_target()>SW_BOT):
   near=False   # balayage pas terminé : pas de minuteur « dud »
  if near and TDESC and abs(float(self.latch[2])+self.h_target()-float(pos[2]))>TD_DUD_DZ:
   near=False   # encore en descente : ne pas déclarer le verrou « dud »
  if near:
   if self.near_since is None:
    self.near_since=tick
   elif tick-self.near_since>RP_DUD_TICKS:
    self.bad.append(self.latch[0:2].copy()); self.latch=None; self.hits=[]; self.near_since=None; self.n_dud+=1
  else:
   self.near_since=None
  if self.latch is not None and self.latch_t is not None and tick-self.latch_t>RP_TERM_MAX and not (SW_ON and self.sw_t0 is not None and self.h_target()>SW_BOT):   # v5 : pas de coupure en plein balayage
   self.bad.append(self.latch[0:2].copy()); self.latch=None; self.hits=[]; self.near_since=None; self.n_dud+=1
 def terminal(self,pos,agl_m):
  if self.latch is None:
   return None
  off=self.latch[0:2]-pos[0:2]; d=float(np.linalg.norm(off))
  if d>RP_TERM_R:
   return None
  tz=float(self.latch[2])+self.h_target(); dz=tz-float(pos[2])
  if dz<0 and agl_m<RP_AGL_MIN:
   dz=0.3
  if SW_ON and d<RP_RING:
   hz=float(pos[2])-float(self.latch[2])
   if self.sw_t0 is None and d<1.2 and hz<=SW_TOP+0.8:
    # départ du balayage à la hauteur actuelle (jamais de remontée), bornée à [SW_BOT, SW_TOP]
    self.sw_t0=self.now; self.sw_h0=min(SW_TOP,max(SW_BOT,hz))
    tz=float(self.latch[2])+self.h_target(); dz=tz-float(pos[2])
   if self.sw_t0 is not None:
    vh=off*0.5; nh=float(np.linalg.norm(vh))
    if nh>0.3:
     vh=vh*(0.3/nh)
    vz=max(-0.6,min(0.6,1.2*dz))
    v=np.array([vh[0],vh[1],vz],np.float64); n=float(np.linalg.norm(v))
    if n<1e-6:
     return np.zeros(3,np.float32),0.0
    return (v/n).astype(np.float32),float(min(1.0,n/V_MAX))
  if TDESC and d<RP_RING and abs(dz)>TD_DZ:
   vz=max(-TD_VMAX,min(TD_VMAX,TD_K*dz))
   vh=off*TD_HK; nh=float(np.linalg.norm(vh))
   if nh>TD_HMAX:
    vh=vh*(TD_HMAX/nh)
   v=np.array([vh[0],vh[1],vz],np.float64); n=float(np.linalg.norm(v))
   return (v/n).astype(np.float32),float(min(1.0,n/V_MAX))
  dzc=max(-1.5,min(1.5,dz))
  vec=np.array([off[0],off[1],dzc],np.float64); n=float(np.linalg.norm(vec))
  dirn=vec/n if n>1e-6 else np.zeros(3)
  if d<1.0 and abs(dz)<0.7:
   speed=0.04
  elif d<RP_RING:
   speed=0.15
  else:
   speed=min(0.33,max(0.15,0.05*d+0.10))
  return dirn.astype(np.float32),float(speed)
def _dfeat(depth):
 d=np.asarray(depth,np.float32)
 if d.ndim==3 and d.shape[-1]==1:
  d=d[...,0]
 if d.ndim!=2:
  d=np.reshape(d,d.shape[:2])
 d=np.clip(d,0.0,1.0)
 gx=np.abs(np.diff(d,axis=1))
 gy=np.abs(np.diff(d,axis=0))
 pc=np.percentile(d,[1,5,10,25,50,75,90,95,99])
 v=[float(d.min()),float(d.mean()),float(d.std()),float(d.max())]
 v += [float(p) for p in pc]
 v += [float((d<=0.1).mean()),float((d<=0.25).mean()),float((d>=0.95).mean()),
  float((d>=0.999).mean()),float((d<=0.001).mean()),
  float(gx.mean()+gy.mean()),float(((gx>0.05).mean()+(gy>0.05).mean())/2.0)]
 h,w=d.shape
 tm,tn,tx=[],[],[]
 for y0 in np.linspace(0,h,5,dtype=int)[:-1]:
  y1=int(y0+h // 4)
  for x0 in np.linspace(0,w,5,dtype=int)[:-1]:
   x1=int(x0+w // 4)
   t=d[y0:y1,x0:x1]
   tm.append(float(t.mean())); tn.append(float(t.min())); tx.append(float(t.max()))
 return v+tm+tn+tx
class _Net:
 def __init__(self,p):
  b=open(p,'rb').read()
  d,h=(int(x) for x in np.frombuffer(b[:4],np.uint16)); o=4
  n1=d*h; n2=h*6
  self.w1=np.frombuffer(b[o:o+n1],np.int8).reshape(d,h).astype(np.float32); o += n1
  self.w2=np.frombuffer(b[o:o+n2],np.int8).reshape(h,6).astype(np.float32); o += n2
  self.s1,self.s2=np.frombuffer(b[o:o+8],np.float32); o += 8
  self.b1=np.frombuffer(b[o:o+2*h],np.float16).astype(np.float32); o += 2*h
  self.b2=np.frombuffer(b[o:o+12],np.float16).astype(np.float32); o += 12
  self.mu=np.frombuffer(b[o:o+2*d],np.float16).astype(np.float32); o += 2*d
  self.sd=np.frombuffer(b[o:o+2*d],np.float16).astype(np.float32)
  self.w1 *= self.s1; self.w2 *= self.s2
 def probs(self,f):
  x=(f-self.mu)/self.sd
  hd=np.maximum(x@self.w1+self.b1,0)
  z=hd@self.w2+self.b2
  z -= z.max()
  e=np.exp(z)
  return e/e.sum()
class _Cls:
 def __init__(self):
  self.net=_Net(_H/'typenet.bin')
  self.reset()
 def reset(self):
  self.ps=np.zeros(6); self.n=0
  self.latched=set(); self.is_mountain=False; self.p40=None
 def update(self,tick,state,depth):
  if tick>600 or tick%5:
   return
  s=np.asarray(state,np.float32).reshape(-1)
  sf=np.zeros(141,np.float32)
  sf[:min(141,s.size)]=s[:141]
  f=np.asarray([float(tick),float(tick)*SIM_DT]+sf.tolist()+_dfeat(depth),np.float32)
  self.ps += self.net.probs(f); self.n += 1
  avg=self.ps/self.n
  if self.n==40:
   self.p40=float(avg[2])
  i=int(np.argmax(avg)); lab=LB[i]
  bar=0.5 if lab=='warehouse' else 0.7
  if self.n>=3 and avg[i]>=bar:
   self.latched.add(lab)
   if lab=='mountain':
    self.is_mountain=True
  if self.n>=40 and avg[4]>=0.15:
   self.latched.add('warehouse')
 def settled(self,t):
  return t in self.latched
 def mtn_ok(self):
  return bool(self.is_mountain and self.p40 is not None and self.p40>=MTN_THR)
def _warm(rgb):
 im=np.asarray(rgb,np.float32).reshape(256,256,3)[128:]
 r,g,b=im[...,0],im[...,1],im[...,2]
 return float(((r>g+0.03) & (g>=b)).mean())
_BOX=np.array([25.0,30.0,30.0,25.0,12.0,20.0])
CVR=9.0
def _ccore(origin,pad):
 xs=np.arange(-48.0,50.0,2.0)
 X,Y=np.meshgrid(xs,xs)
 pts=np.stack([X.ravel(),Y.ravel()],1)
 dc=np.linalg.norm(pts-origin,axis=1)
 dp=np.linalg.norm(pts-pad,axis=1)
 m=(dc<=33.0)&(dp<=83.0)
 pts,dc,dp=pts[m],dc[m],dp[m]
 ax=np.abs(pts).max(1)
 dens=np.zeros(len(pts))
 for bb in _BOX:
  dens[ax<=bb] += 1.0/(6.0*(2*bb) ** 2)
 w=dens/(1+np.exp(-(30.0-dc)))/(1+np.exp(-(80.0-dp)))
 return pts,w,dc
def _bous(core,lane,spacing,cap,pos):
 core=np.asarray(core,np.float64)
 if core.shape[0]<4:
  return []
 k=np.round(core[:,0]/lane).astype(int)
 o=np.lexsort((np.where(k%2==0,core[:,1],-core[:,1]),k))
 path=core[o]
 sel=[path[0]]
 for q in path[1:]:
  if float(np.linalg.norm(q-sel[-1]))>=spacing:
   sel.append(q)
 w=sel[:cap]
 if len(w)>=2 and np.linalg.norm(w[-1]-pos)<np.linalg.norm(w[0]-pos):
  w=w[::-1]
 return [np.asarray(x,np.float64) for x in w]
def _wps(origin,pad,pos,lane,spacing,cap):
 xs=np.arange(-48.0,50.0,2.0)
 X,Y=np.meshgrid(xs,xs)
 pts=np.stack([X.ravel(),Y.ravel()],1)
 dc=np.linalg.norm(pts-origin,axis=1)
 dp=np.linalg.norm(pts-pad,axis=1)
 m=(dc<=33.0) & (dp<=83.0)
 pts,dc,dp=pts[m],dc[m],dp[m]
 ax=np.abs(pts).max(1)
 dens=np.zeros(len(pts))
 for bb in _BOX:
  dens[ax<=bb] += 1.0/(6.0*(2*bb) ** 2)
 w=dens/(1+np.exp(-(30.0-dc)))/(1+np.exp(-(80.0-dp)))
 core=pts
 if w.sum()>0:
  o=np.argsort(-w)
  cw=np.cumsum(w[o])/w.sum()
  core=pts[o[:int(np.searchsorted(cw,0.9))+1]]
 if core.shape[0]<4:
  core=pts[dc<=30.0]
 if not core.shape[0]:
  return []
 k=np.round(core[:,0]/lane).astype(int)
 o=np.lexsort((np.where(k%2==0,core[:,1],-core[:,1]),k))
 path=core[o]
 sel=[path[0]]
 for q in path[1:]:
  if float(np.linalg.norm(q-sel[-1]))>=spacing:
   sel.append(q)
 w=sel[:cap]
 if len(w)>=2 and np.linalg.norm(w[-1]-pos)<np.linalg.norm(w[0]-pos):
  w=w[::-1]
 return [np.asarray(x,np.float64) for x in w]
class DroneFlightController:
 def __init__(self,*,model_path=None,providers=None):
  so=ort.SessionOptions()
  so.intra_op_num_threads=2
  so.inter_op_num_threads=1
  so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
  so.add_session_config_entry("session.intra_op.allow_spinning","0")
  self.session=ort.InferenceSession(str(model_path or _H/'policy.onnx'),so,
           providers=providers or ["CPUExecutionProvider"])
  self._ms=next(int(i.shape[0]) for i in self.session.get_inputs()
      if i.name=="memory_tensor")
  self._in={i.name for i in self.session.get_inputs()}
  self._cls=_Cls() if ON else None
  self._rp=_RgbPrimary() if (ON and RP_ON) else None
  self._esc=_Escape() if (ON and ESC_ON) else None
  self._dl=_RgbPrimary() if (ON and DL_ON) else None
  if self._dl is not None: self._dl.hover_up=DL_HOVER_UP
  self._outn=[o.name for o in self.session.get_outputs()]
  self._dlok=all(n in self._outn for n in DL_OUT)
  self.reset()
 def reset(self):
  self.memory_tensor=np.zeros(self._ms,np.float32)
  self.tick=0
  self._h={15: [],25: []}
  self._pad=None
  self._on=False
  self._cfg=None
  self._w=None
  self._i=0
  self._adv=0
  self._cv=None
  self._cvt=0
  self._av=0; self._av_side=1.0; self._av_n=0
  self.route='king'
  self._sup=None; self._mass=None; self._w0=None
  self._rp_term=None
  self._tko=False
  self._dl_term=None; self._dl_clue0=None
  if getattr(self,'_dl',None) is not None:
   self._dl.reset()
  if self._rp is not None:
   self._rp.reset()
  if self._esc is not None:
   self._esc.reset()
  if self._cls:
   self._cls.reset()
 def _vbuild(self,pos,clue,c):
  if self._sup is None:
   sup,w,dc=_ccore(pos[0:2]+clue,self._pad)
   self._sup=sup; self._w0=w; self._mass=w.copy(); self._dc=dc
  wps=_bous(self._core(0.9),c[3],c[4],c[5],pos[0:2])
  if not wps:
   self._mass=self._w0.copy()
   wps=_bous(self._core(0.9),c[3]*0.65,c[4]*0.7,c[5],pos[0:2])
  if not wps:
   wps=_bous(self._sup[self._dc<=30.0],c[3],c[4],c[5],pos[0:2])
  return wps
 def _core(self,frac):
  w=self._mass; t=float(w.sum())
  if t<=0:
   return np.zeros((0,2))
  o=np.argsort(-w)
  cw=np.cumsum(w[o])/t
  return self._sup[o[:int(np.searchsorted(cw,frac))+1]]
 def _swept(self,xy):
  if self._mass is not None:
   self._mass[np.linalg.norm(self._sup-np.asarray(xy,np.float64),axis=1)<=CVR]=0.0
 def _stype(self):
  c=self._cls
  if c.is_mountain:
   return 'mountain' if 'mountain' in TYPES else None
  for t in TYPES:
   if t!='mountain' and c.settled(t):
    return t
  return None

 def _dl_on(self):
  """Verrou profondeur actif hors mountain (mountain = verrou RGB), une fois le type estimé."""
  c=self._cls
  return bool(self._dl is not None and c is not None and self.tick>=DL_T0 and c.n>=3 and not c.is_mountain)

 def _champ(self):
  """True on village/warehouse: bypass every added subsystem and run the champion's
  process end to end. Mountain wins the tie (is_mountain is sticky and set first)."""
  if not CHAMP_ROUTE:
   return False
  c=self._cls
  if c is None or c.is_mountain:
   return False
  return bool(c.settled('village') or c.settled('warehouse'))
 def act(self,observation):
  self.tick += 1
  rq=False
  if self.route.endswith('+av'): self.route=self.route[:-3]
  st=np.asarray(observation["state"],np.float32).reshape(-1).copy()
  if ON and self._cls:
   pos=st[0:3].astype(np.float64)
   if self._pad is None:
    self._pad=pos[0:2].copy()
   self._cls.update(self.tick,st,np.asarray(observation["depth"],np.float32))
   for pd,hs in self._h.items():
    if self.tick%pd==0:
     hs.append(pos[0:2].copy())
     if len(hs)>20:
      hs.pop(0)
   if not self._on:
    t=self._stype()
    c=CFG.get(t) if t else None
    if t=='mountain' and self._cls.mtn_ok() and not self._champ():
     c=MTN_STRONG
    if c and self.tick>=c[0]:
     hs=self._h[c[1]]
     if len(hs)>=20:
      hp=np.asarray(hs)
      if float(np.max(np.linalg.norm(hp-hp[0],axis=1)))<c[2]:
       if c[8] and self._cv!='ok':
        if self._cv!='block':
         rgb=observation.get('rgb') if hasattr(observation,'get') else None
         fr=np.asarray(rgb,np.float32) if rgb is not None else None
         if fr is not None and float(fr.max())>0.01:
          self._cv='block' if _warm(fr)>=VW else 'ok'
          if self._cv=='ok':
           self._on=True; self._cfg=c
         elif self._cvt<3:
          rq=True; self._cv='req'; self._cvt += 1
       else:
        self._on=True; self._cfg=c
   if self._on:
    c=self._cfg
    if self._w is None:
     if len(c)>9:
      self._w=self._vbuild(pos,st[-2:].astype(np.float64),c)
     else:
      self._w=_wps(pos[0:2]+st[-2:].astype(np.float64),self._pad,
        pos[0:2],c[3],c[4],c[5])
     self._i=0; self._adv=self.tick
    if len(self._w):
     self._i=min(self._i,len(self._w)-1)
     wp=self._w[self._i]
     if float(np.linalg.norm(wp-pos[0:2]))<c[7] or self.tick-self._adv>c[6]:
      if len(c)>9:
       self._swept(wp)
      nx=self._i+1
      if nx>=len(self._w):
       if len(c)>9:
        rb=self._vbuild(pos,st[-2:].astype(np.float64),c)
        if rb:
         self._w=rb
       nx=0
      self._i=nx
      self._adv=self.tick
      wp=self._w[self._i]
     off=wp-pos[0:2]
     bl=c[9] if len(c)>9 else 1.0
     if bl<1.0:
      off=bl*off+(1.0-bl)*st[-2:].astype(np.float64)
     st[-2]=np.float32(off[0]); st[-1]=np.float32(off[1])
     self.route='king+tour'
   self._rp_term=None
   if self._rp is not None and not self._champ():
    try:
     if self._rp.clue0 is None:
      self._rp.clue0=pos[0:2]+np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
     if self._cls.mtn_ok():
      self._rp.observe(self.tick,observation,pos,st[3:6].astype(np.float64))
      if self._rp.latch is not None:
       lk=self._rp.latch[0:2]-pos[0:2]; st[-2]=np.float32(lk[0]); st[-1]=np.float32(lk[1]); self.route='king+rgb'
       self._rp_term=self._rp.terminal(pos,float(st[162])*20.0 if st.size>162 else 20.0)
      self._rp.tick_end(self.tick,pos)
    except Exception:
     self._rp_term=None
   self._dl_term=None
   if self._dl_on():
    try:
     if self._dl_clue0 is None:
      self._dl_clue0=pos[0:2]+np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
     if self._dl.latch is not None:
      lk=self._dl.latch[0:2]-pos[0:2]; st[-2]=np.float32(lk[0]); st[-1]=np.float32(lk[1]); self.route='king+dl'
      self._dl_term=self._dl.terminal(pos,float(st[162])*20.0 if st.size>162 else 20.0)
     self._dl.tick_end(self.tick,pos)
     _L=self._dl
     if _L.latch is not None:
      # v5b : la cible doit être revue par le détecteur tant qu'on n'est pas au-dessus (sinon faux verrou)
      _far=float(np.linalg.norm(_L.latch[0:2]-pos[0:2]))>DL_RECONF_D
      if (_far and self.tick-getattr(_L,'conf_t',self.tick)>DL_RECONF) or (_L.latch_t is not None and self.tick-_L.latch_t>DL_TERM_MAX and _far):
       _L.bad.append(_L.latch[0:2].copy()); _L.latch=None; _L.hits=[]; _L.near_since=None; _L.n_dud+=1
    except Exception:
     self._dl_term=None
  if AV_D>0.0 and self._cls and self.tick>=AV_T0 and self._stype() in AV_TYPES:
   dm=np.asarray(observation['depth'],np.float32).reshape(256,256)*29.5+0.5
   ctr=float(dm[96:160,96:160].min())
   if ctr<AV_D:
    if self._av==0:
     self._av_side=1.0 if float(dm[64:192,128:224].min())>=float(dm[64:192,32:128].min()) else -1.0
     self._av_n+=1
    self._av=AV_HOLD
   elif ctr>AV_D+AV_REL and self._av>0:
    self._av-=1
   if self._av>0:
    ang=-self._av_side*np.radians(AV_ROT)
    cx,cy=float(st[-2]),float(st[-1]); ca,sa=float(np.cos(ang)),float(np.sin(ang))
    st[-2]=np.float32(ca*cx-sa*cy); st[-1]=np.float32(sa*cx+ca*cy)
    if not self.route.endswith('+av'): self.route=self.route+'+av'
  feed={"depth": np.asarray(observation["depth"],np.float32).reshape(256,256,1),
    "state": st,"memory_tensor": self.memory_tensor}
  if "rgb" in self._in:
   rgb=observation.get("rgb") if hasattr(observation,"get") else None
   feed["rgb"]=(np.asarray(rgb,np.float32).reshape(256,256,3)
      if rgb is not None else np.zeros((256,256,3),np.float32))
  _o=self.session.run(None,feed); a,m=_o[0],_o[1]
  if self._dlok and self._dl_on() and self._dl_clue0 is not None:
   try:
    det=np.asarray(_o[self._outn.index(DL_OUT[0])],np.float64).ravel(); cam=np.asarray(_o[self._outn.index(DL_OUT[1])],np.float64).ravel()
    if det[0]>=DL_P and det[3]<=DL_D:
     fwd,up,right=_axes(st[3:6]); c0=st[0:3].astype(np.float64)+fwd*_CAMF+up*_CAMU
     q=c0+right*cam[0]-up*cam[1]+fwd*cam[2]
     if np.all(np.isfinite(q)) and float(np.linalg.norm(q[0:2]-self._dl_clue0))<=DL_R and -25.0<=float(q[2]-st[2])<=1.0:
      self._dl.add_point(self.tick,q,DL_TOL)
   except Exception:
    pass
  self.memory_tensor=np.asarray(m,np.float32).reshape(self._ms)
  out=np.asarray(a,np.float32).reshape(-1)
  if rq and out.shape[0]>=6:
   out=out.copy(); out[5]=1.0
  if self._esc is not None and out.shape[0]>=4 and (self._tko or not self._champ()):
   try:
    _p=st[0:3].astype(np.float64); _agl=float(st[162])*20.0 if st.size>162 else 20.0
    if self.tick==1:
     if not self._esc.decide(observation['depth'],_p,float(st[5])) and TKO and TKO_Z[0]<=float(_p[2])<=TKO_Z[1]:
      _d=np.asarray(observation['depth'],np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
      if float(_d[0:16].min())<=TKO_TOP_M:
       self._esc.force(observation['depth'],_p,float(st[5]))
       self._tko=True
     elif TKO and self._esc.trap and TKO_Z[0]<=float(_p[2])<=TKO_Z[1]:
      self._tko=True
    _eok=True
    if ESC_TERR:
     _c2=self._cls
     _eok=bool(self._tko or (_c2 is not None and (_c2.is_mountain or _c2.settled('forest'))))
    if _eok and self._esc.phase in ('scan','creep','climb'):
     _e=self._esc.act(observation['depth'],_p,_agl,math.degrees(float(st[4])),float(st[5]))
     if _e is not None:
      out=out.copy(); out[0:3]=_e[0]; out[3]=np.float32(_e[1]); out[4]=np.float32(0.0 if _e[2] is None else _e[2])
   except Exception:
    pass
  if AGL_CAP>0 and out.shape[0]>=4 and self.tick>=AGL_T0 and self._cls is not None and self._cls.mtn_ok() and not self._champ()      and self._rp_term is None and not (self._esc is not None and self._esc.phase in ('scan','creep','climb')):
   try:
    _agl=float(st[162])*20.0 if st.size>162 else 0.0
    _cl=np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
    if _agl>AGL_CAP and float(np.linalg.norm(_cl))<=AGL_ZONE and (self._rp is None or self._rp.latch is None):  # v3c
     _d=np.asarray(observation['depth'],np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
     if float(_d[128:256,64:192].min())>AGL_CLEAR_M:
      dv=np.asarray(out[0:3],np.float64); n=float(np.linalg.norm(dv))
      v=dv/n*float(out[3])*V_MAX if n>1e-6 else np.zeros(3)
      vz=-min(AGL_VMAX,max(AGL_VMIN,AGL_K*(_agl-AGL_CAP+2.0)))
      if v[2]>vz and v[2]<0.3:   # v5b : ne pas contrer une montée de la nav
       v[2]=vz; nv=float(np.linalg.norm(v))
       out=out.copy(); out[0:3]=(v/nv).astype(np.float32); out[3]=np.float32(min(1.0,nv/V_MAX))
   except Exception:
    pass
  if self._dl_term is not None and out.shape[0]>=5 and not (self._esc is not None and self._esc.phase in ('scan','creep','climb')) and float(np.asarray(observation['depth'],np.float32).reshape(256,256)[96:160,96:160].min())*(_DMAX-_DMIN)+_DMIN>=DL_CLEAR_M:
   out=out.copy(); out[0:3]=self._dl_term[0]; out[3]=np.float32(self._dl_term[1]); out[4]=np.float32(0.0)
  if self._rp is not None and out.shape[0]>=6 and self._cls is not None and self._cls.mtn_ok() and not self._champ():
   out=out.copy()
   if self._rp_term is not None:
    out[0:3]=self._rp_term[0]; out[3]=np.float32(self._rp_term[1]); out[4]=np.float32(0.0)
   if self._rp.want_rgb(self.tick):
    out[5]=np.float32(1.0)
  return out

# r20
