import os
import numpy as np, math, torch
from flash_attn.flash_attn_interface import _flash_attn_forward
DD=os.environ.get("FA_DUMP_DIR",".")
for S in (512,1024):
    f=open(f"{DD}/fa_my_{S}.bin","rb")
    B=np.fromfile(f,dtype=np.int32,count=1)[0];H=np.fromfile(f,dtype=np.int32,count=1)[0]
    Sq=np.fromfile(f,dtype=np.int32,count=1)[0];D=np.fromfile(f,dtype=np.int32,count=1)[0]
    n=B*H*Sq*D
    q=np.fromfile(f,dtype=np.float16,count=n).reshape(B,H,Sq,D)
    k=np.fromfile(f,dtype=np.float16,count=n).reshape(B,H,Sq,D)
    v=np.fromfile(f,dtype=np.float16,count=n).reshape(B,H,Sq,D)
    o_c=np.fromfile(f,dtype=np.float16,count=n).reshape(B,H,Sq,D)
    f.close()
    qt=torch.from_numpy(q).cuda();kt=torch.from_numpy(k).cuda();vt=torch.from_numpy(v).cuda()
    sc=1.0/math.sqrt(D)
    _,_,_,_,o_t,_,_,_,_=_flash_attn_forward(qt.transpose(1,2).contiguous(),kt.transpose(1,2).contiguous(),vt.transpose(1,2).contiguous(),0.0,sc,False,(-1,-1))
    o_t=o_t.transpose(1,2).contiguous()
    diff=(o_t.float()-torch.from_numpy(o_c).cuda().float()).abs()
    print(f"S={S}: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f} allclose(1e-2)={torch.allclose(o_t,torch.from_numpy(o_c).cuda(),atol=1e-2,rtol=1e-2)}")
