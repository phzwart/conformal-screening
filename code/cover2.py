import numpy as np
from scipy.stats import betabinom, beta
rng=np.random.default_rng(0); g=0.05; R=200000
def table(n,m,kind):
    lo=np.zeros(n+1); hi=np.zeros(n+1)
    for k in range(n+1):
        if kind=="exact": a1,b1,a2,b2=k,n-k+1,k+1,n-k
        elif kind=="o1": a1,b1,a2,b2=k+1,n-k+1,k+1,n-k+1
        else: a1,b1,a2,b2=k+.5,n-k+.5,k+.5,n-k+.5
        lo[k]=betabinom.ppf(g/2,m,a1,b1) if a1>0 else 0
        hi[k]=betabinom.ppf(1-g/2,m,a2,b2) if b2>0 else m
    return lo,hi
print("n_audit m  rate-prior  | exact  BB(+1)  BB(+1/2) | width exact/BB(+1) | CP->p_F  CP->E[p|Dcal]")
for (n,m,prior) in [(46,100,(1,30)),(46,1000,(1,30)),(100,1000,(1,60)),(235,1000,(2,40)),(500,1000,(2,5))]:
    p=rng.beta(*prior,size=R); k=rng.binomial(n,p); km=rng.binomial(m,p)
    kc=rng.binomial(200,p)
    out=[];W={}
    for kind in ["exact","o1","oh"]:
        lo,hi=table(n,m,kind); out.append(np.mean((km>=lo[k])&(km<=hi[k]))); W[kind]=np.mean(hi[k]-lo[k])
    cl=np.where(k>0,beta.ppf(g/2,np.maximum(k,1),n-k+1),0); cu=np.where(k<n,beta.ppf(1-g/2,k+1,np.maximum(n-k,1)),1)
    pm=(prior[0]+kc)/(sum(prior)+200)
    print(f"{n:5d} {m:5d} {str(prior):10s} | {out[0]:.3f}  {out[1]:.3f}   {out[2]:.3f}   | {W['exact']/W['o1']:.3f}            | {np.mean((p>=cl)&(p<=cu)):.3f}   {np.mean((pm>=cl)&(pm<=cu)):.3f}")
