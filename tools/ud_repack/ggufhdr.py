"""Minimal dependency-free GGUF v3 header parser (tolerates ggml types unknown to the python gguf package)."""
import struct
TYPES={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',
16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16',
34:'TQ1_0',35:'TQ2_0',39:'MXFP4',40:'NVFP4'}
# (elements per block, bytes per block)
BLK={'F32':(1,4),'F16':(1,2),'BF16':(1,2),'Q4_0':(32,18),'Q4_1':(32,20),'Q5_0':(32,22),'Q5_1':(32,24),'Q8_0':(32,34),'Q2_K':(256,84),'Q3_K':(256,110),
'Q4_K':(256,144),'Q5_K':(256,176),'Q6_K':(256,210),'Q8_K':(256,292),'IQ2_XXS':(256,66),'IQ2_XS':(256,74),'IQ3_XXS':(256,98),'IQ1_S':(256,50),'IQ4_NL':(32,18),
'IQ3_S':(256,110),'IQ2_S':(256,82),'IQ4_XS':(256,136),'IQ1_M':(256,56),'TQ1_0':(256,54),'TQ2_0':(256,66),'MXFP4':(32,17),'NVFP4':(64,36),'I8':(1,1),'I16':(1,2),'I32':(1,4),'I64':(1,8),'F64':(1,8)}
def _rd(f,fmt):
    n=struct.calcsize(fmt); return struct.unpack('<'+fmt,f.read(n))
def _str(f):
    (n,)=_rd(f,'Q'); return f.read(n).decode('utf8','replace')
def _val(f,t):
    sc={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    if t in sc: return _rd(f,sc[t])[0]
    if t==8: return _str(f)
    if t==9:
        (et,)=_rd(f,'I'); (n,)=_rd(f,'Q'); return [_val(f,et) for _ in range(n)]
    raise ValueError(t)
def read_header(path):
    """returns dict(kv, tensors=[dict(name,type,type_id,shape,nbytes,offset)], data_start)"""
    with open(path,'rb') as f:
        if f.read(4)!=b'GGUF': raise ValueError('not gguf')
        (ver,)=_rd(f,'I'); nt,nkv=_rd(f,'QQ')
        kv={}
        for _ in range(nkv):
            k=_str(f); (t,)=_rd(f,'I'); v=_val(f,t)
            kv[k]=v if not (isinstance(v,list) and len(v)>64) else f'<list {len(v)}>'
        ts=[]
        for _ in range(nt):
            name=_str(f); (nd,)=_rd(f,'I'); dims=list(_rd(f,'Q'*nd)); (ty,)=_rd(f,'I'); (off,)=_rd(f,'Q')
            tn=TYPES.get(ty,f'TYPE{ty}')
            ne=1
            for d in dims: ne*=d
            be,bb=BLK.get(tn,(0,0))
            nb=(ne//be)*bb if be else 0
            ts.append(dict(name=name,type=tn,type_id=ty,shape=dims,nelem=ne,nbytes=nb,offset=off))
        al=kv.get('general.alignment',32); pos=f.tell(); ds=(pos+al-1)//al*al
    return dict(kv=kv,tensors=ts,data_start=ds,version=ver)
