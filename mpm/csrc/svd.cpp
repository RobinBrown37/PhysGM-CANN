// Project-local Ascend C SIMD kernels. No global OPP package registration.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int L = 128;
using T = LocalTensor<float>;
class Jacobi {
 public:
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf, maskbuf;
  T a[9], v[9], u[9], t[20], sig[3];
  LocalTensor<uint8_t> mask;
  __aicore__ inline void init() {
    pipe.InitBuffer(buf, (9+9+9+20+3)*L*4);
    pipe.InitBuffer(maskbuf, L);
    auto b = buf.Get<float>();
    for(int i=0;i<9;i++){a[i]=b[i*L];v[i]=b[(9+i)*L];u[i]=b[(18+i)*L];}
    for(int i=0;i<20;i++)t[i]=b[(27+i)*L];
    for(int i=0;i<3;i++)sig[i]=b[(47+i)*L];
    mask=maskbuf.Get<uint8_t>();
  }
  __aicore__ inline void dot(T out, T *m, int p, int q) {
    Mul(out,m[p],m[q],L); Mul(t[18],m[3+p],m[3+q],L); Add(out,out,t[18],L);
    Mul(t[18],m[6+p],m[6+q],L); Add(out,out,t[18],L);
  }
  __aicore__ inline void rotate(int p,int q){
    dot(t[0],a,p,p);dot(t[1],a,q,q);dot(t[2],a,p,q);
    Sub(t[3],t[1],t[0],L); // delta
    Mul(t[4],t[3],t[3],L);Mul(t[5],t[2],t[2],L);Muls(t[5],t[5],4.f,L);
    Add(t[4],t[4],t[5],L);Sqrt(t[4],t[4],L);
    CompareScalar(mask,t[3],0.f,CMPMODE::GE,L);
    Muls(t[5],t[4],-1.f,L);
    Select(t[4],mask,t[4],t[5],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
    Add(t[4],t[3],t[4],L); // signed denominator
    Abs(t[5],t[4],L);Maxs(t[5],t[5],1.e-30f,L);
    Muls(t[6],t[2],2.f,L);Div(t[6],t[6],t[5],L);
    Muls(t[7],t[6],-1.f,L);Select(t[6],mask,t[6],t[7],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
    // Stop rotations below relative roundoff, avoiding perpetual equal-value swaps.
    Mul(t[7],t[0],t[1],L);Sqrt(t[7],t[7],L);Muls(t[7],t[7],1.e-7f,L);
    Abs(t[8],t[2],L);Compare(mask,t[8],t[7],CMPMODE::GT,L);
    Duplicate(t[9],0.f,L);Select(t[6],mask,t[6],t[9],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
    Mul(t[7],t[6],t[6],L);Adds(t[7],t[7],1.f,L);Sqrt(t[7],t[7],L);
    Duplicate(t[8],1.f,L);Div(t[7],t[8],t[7],L);Mul(t[6],t[6],t[7],L); // s,c
    for(int row=0;row<3;row++) {
      int x=row*3+p,y=row*3+q;
      Mul(t[10],a[x],t[7],L);Mul(t[11],a[y],t[6],L);Sub(t[10],t[10],t[11],L);
      Mul(t[11],a[x],t[6],L);Mul(a[y],a[y],t[7],L);Add(a[y],a[y],t[11],L);Adds(a[x],t[10],0.f,L);
      Mul(t[10],v[x],t[7],L);Mul(t[11],v[y],t[6],L);Sub(t[10],t[10],t[11],L);
      Mul(t[11],v[x],t[6],L);Mul(v[y],v[y],t[7],L);Add(v[y],v[y],t[11],L);Adds(v[x],t[10],0.f,L);
    }
  }
  __aicore__ inline void swap(int p,int q){
    Compare(mask,sig[p],sig[q],CMPMODE::LT,L);
    Adds(t[0],sig[p],0.f,L);Select(sig[p],mask,sig[q],sig[p],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
    Select(sig[q],mask,t[0],sig[q],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
    for(int r=0;r<3;r++){
      int x=r*3+p,y=r*3+q;
      Adds(t[0],a[x],0.f,L);Select(a[x],mask,a[y],a[x],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);Select(a[y],mask,t[0],a[y],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
      Adds(t[0],v[x],0.f,L);Select(v[x],mask,v[y],v[x],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);Select(v[y],mask,t[0],v[y],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
    }
  }
  __aicore__ inline void cross(T *out,T *m,int p,int q){
    for(int r=0;r<3;r++){
      int r1=(r+1)%3,r2=(r+2)%3;
      Mul(t[16],m[r1*3+p],m[r2*3+q],L);Mul(t[17],m[r2*3+p],m[r1*3+q],L);Sub(out[r],t[16],t[17],L);
    }
  }
  __aicore__ inline void run(GM_ADDR in,GM_ADDR out,int n,int sweeps){
    GlobalTensor<float> src,dst;src.SetGlobalBuffer((__gm__ float*)in);dst.SetGlobalBuffer((__gm__ float*)out);
    for(int off=GetBlockIdx()*L;off<n;off+=GetBlockNum()*L){
      for(int k=0;k<9;k++){DataCopy(a[k],src[k*n+off],L);Duplicate(v[k],(k%4==0)?1.f:0.f,L);}
      PipeBarrier<PIPE_ALL>();
      for(int k=0;k<9;k++)Adds(u[k],a[k],0.f,L); // retain original F
      for(int j=0;j<sweeps;j++){rotate(0,1);rotate(0,2);rotate(1,2);}
      for(int k=0;k<3;k++){dot(sig[k],a,k,k);Sqrt(sig[k],sig[k],L);}
      swap(0,1);swap(1,2);swap(0,1);
      // Reorthogonalize V and recompute F*V from the original input. This
      // prevents repeated plastic projections from accumulating rotation drift.
      dot(t[0],v,0,0);Sqrt(t[0],t[0],L);Maxs(t[0],t[0],1.e-30f,L);
      for(int r=0;r<3;r++)Div(v[r*3],v[r*3],t[0],L);
      dot(t[0],v,0,1);
      for(int r=0;r<3;r++){Mul(t[1],v[r*3],t[0],L);Sub(v[r*3+1],v[r*3+1],t[1],L);}
      dot(t[0],v,1,1);Sqrt(t[0],t[0],L);Maxs(t[0],t[0],1.e-30f,L);
      for(int r=0;r<3;r++)Div(v[r*3+1],v[r*3+1],t[0],L);
      T vcross[3]={t[12],t[13],t[14]};cross(vcross,v,0,1);
      for(int r=0;r<3;r++)Adds(v[r*3+2],vcross[r],0.f,L);
      for(int r=0;r<3;r++)for(int j=0;j<3;j++){
        Mul(a[r*3+j],u[r*3],v[j],L);Mul(t[0],u[r*3+1],v[3+j],L);Add(a[r*3+j],a[r*3+j],t[0],L);
        Mul(t[0],u[r*3+2],v[6+j],L);Add(a[r*3+j],a[r*3+j],t[0],L);
      }
      for(int k=0;k<3;k++){dot(sig[k],a,k,k);Sqrt(sig[k],sig[k],L);}
      // Enforce det(V)=+1; U is completed with a cross product, so the final
      // singular value carries the determinant sign (Warp svd3 convention).
      T cr[3]={t[12],t[13],t[14]};cross(cr,v,0,1);
      Mul(t[0],cr[0],v[2],L);Mul(t[1],cr[1],v[5],L);Add(t[0],t[0],t[1],L);Mul(t[1],cr[2],v[8],L);Add(t[0],t[0],t[1],L);
      CompareScalar(mask,t[0],0.f,CMPMODE::LT,L);
      for(int r=0;r<3;r++){
        Muls(t[0],v[r*3+2],-1.f,L);Select(v[r*3+2],mask,t[0],v[r*3+2],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
        Muls(t[0],a[r*3+2],-1.f,L);Select(a[r*3+2],mask,t[0],a[r*3+2],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
      }
      Maxs(t[0],sig[0],1.e-30f,L);
      for(int r=0;r<3;r++)Div(u[r*3],a[r*3],t[0],L);
      CompareScalar(mask,sig[0],1.e-20f,CMPMODE::GT,L);
      for(int r=0;r<3;r++){Duplicate(t[1],r==0?1.f:0.f,L);Select(u[r*3],mask,u[r*3],t[1],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);}
      // Orthogonalize column 1 and complete the rank-deficient basis.
      Mul(t[0],u[0],a[1],L);Mul(t[1],u[3],a[4],L);Add(t[0],t[0],t[1],L);Mul(t[1],u[6],a[7],L);Add(t[0],t[0],t[1],L);
      for(int r=0;r<3;r++){Mul(t[1],u[r*3],t[0],L);Sub(u[r*3+1],a[r*3+1],t[1],L);}
      dot(t[2],u,1,1);Sqrt(t[2],t[2],L);
      Abs(t[0],u[0],L);CompareScalar(mask,t[0],0.9f,CMPMODE::LT,L);
      Duplicate(t[3],1.f,L);Duplicate(t[4],0.f,L);
      Select(t[5],mask,t[3],t[4],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);Select(t[6],mask,t[4],t[3],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
      Mul(t[7],t[5],u[0],L);Mul(t[8],t[6],u[3],L);Add(t[7],t[7],t[8],L);
      Mul(t[8],t[7],u[0],L);Sub(t[5],t[5],t[8],L);Mul(t[8],t[7],u[3],L);Sub(t[6],t[6],t[8],L);Mul(t[7],t[7],u[6],L);Muls(t[7],t[7],-1.f,L);
      CompareScalar(mask,t[2],1.e-20f,CMPMODE::GT,L);
      T fall[3]={t[5],t[6],t[7]};for(int r=0;r<3;r++)Select(u[r*3+1],mask,u[r*3+1],fall[r],SELMODE::VSEL_TENSOR_TENSOR_MODE,L);
      dot(t[2],u,1,1);Sqrt(t[2],t[2],L);Maxs(t[2],t[2],1.e-30f,L);for(int r=0;r<3;r++)Div(u[r*3+1],u[r*3+1],t[2],L);
      cross(cr,u,0,1);for(int r=0;r<3;r++)Adds(u[r*3+2],cr[r],0.f,L);
      Mul(sig[2],u[2],a[2],L);Mul(t[0],u[5],a[5],L);Add(sig[2],sig[2],t[0],L);Mul(t[0],u[8],a[8],L);Add(sig[2],sig[2],t[0],L);
      PipeBarrier<PIPE_ALL>();
      for(int k=0;k<9;k++){DataCopy(dst[k*n+off],u[k],L);DataCopy(dst[(12+k)*n+off],v[k],L);}
      for(int k=0;k<3;k++)DataCopy(dst[(9+k)*n+off],sig[k],L);
      PipeBarrier<PIPE_ALL>();
    }
  }
};
extern "C" __global__ __aicore__ void physgm_svd(GM_ADDR a,GM_ADDR out,int n,int sweeps){Jacobi k;k.init();k.run(a,out,n,sweeps);}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void physgm_svd_launch(void* a,void* out,int n,int sweeps,void* stream){
  physgm_svd<<<24,nullptr,stream>>>((uint8_t*)a,(uint8_t*)out,n,sweeps);
}
#endif
