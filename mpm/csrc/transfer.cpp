// Operator families 2 and 3: quadratic APIC transfers. IndexAdd and grid gather
// remain built-in TorchNPU operations; no custom atomics or host grid solving.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int L=128;
using T=LocalTensor<float>;
class Transfer {
 public:
  TPipe pipe;TBuf<TPosition::VECCALC> buf,ibuf;
  T p[27],fx[3],base[3],w[9],dw[9],tmp[16],acc[21];
  LocalTensor<int32_t> ids;
  __aicore__ inline void init(){
    pipe.InitBuffer(buf,(27+3+3+9+9+16+21)*L*4);pipe.InitBuffer(ibuf,L*4);
    auto b=buf.Get<float>();int pos=0;
    for(int k=0;k<27;k++)p[k]=b[(pos++)*L];
    for(int k=0;k<3;k++)fx[k]=b[(pos++)*L];
    for(int k=0;k<3;k++)base[k]=b[(pos++)*L];
    for(int k=0;k<9;k++)w[k]=b[(pos++)*L];
    for(int k=0;k<9;k++)dw[k]=b[(pos++)*L];
    for(int k=0;k<16;k++)tmp[k]=b[(pos++)*L];
    for(int k=0;k<21;k++)acc[k]=b[(pos++)*L];
    ids=ibuf.Get<int32_t>();
  }
  __aicore__ inline void weights(float dx){
    for(int a=0;a<3;a++){
      Muls(tmp[0],p[a],1.f/dx,L);Adds(tmp[1],tmp[0],-.5f,L);
      Cast(ids,tmp[1],RoundMode::CAST_TRUNC,L);Cast(base[a],ids,RoundMode::CAST_NONE,L);Sub(fx[a],tmp[0],base[a],L);
      Muls(tmp[0],fx[a],-1.f,L);Adds(tmp[0],tmp[0],1.5f,L);Mul(w[3*a],tmp[0],tmp[0],L);Muls(w[3*a],w[3*a],.5f,L);
      Adds(tmp[0],fx[a],-1.f,L);Mul(w[3*a+1],tmp[0],tmp[0],L);Muls(w[3*a+1],w[3*a+1],-1.f,L);Adds(w[3*a+1],w[3*a+1],.75f,L);
      Adds(tmp[0],fx[a],-.5f,L);Mul(w[3*a+2],tmp[0],tmp[0],L);Muls(w[3*a+2],w[3*a+2],.5f,L);
      Adds(dw[3*a],fx[a],-1.5f,L);Muls(dw[3*a],dw[3*a],1.f/dx,L);
      Adds(dw[3*a+1],fx[a],-1.f,L);Muls(dw[3*a+1],dw[3*a+1],-2.f/dx,L);
      Adds(dw[3*a+2],fx[a],-.5f,L);Muls(dw[3*a+2],dw[3*a+2],1.f/dx,L);
    }
  }
  __aicore__ inline void neighbor(int i,int j,int k){
    Mul(tmp[0],w[i],w[3+j],L);Mul(tmp[0],tmp[0],w[6+k],L);
    Mul(tmp[1],dw[i],w[3+j],L);Mul(tmp[1],tmp[1],w[6+k],L);
    Mul(tmp[2],w[i],dw[3+j],L);Mul(tmp[2],tmp[2],w[6+k],L);
    Mul(tmp[3],w[i],w[3+j],L);Mul(tmp[3],tmp[3],dw[6+k],L);
    Muls(tmp[4],fx[0],-1.f,L);Adds(tmp[4],tmp[4],float(i),L);
    Muls(tmp[5],fx[1],-1.f,L);Adds(tmp[5],tmp[5],float(j),L);
    Muls(tmp[6],fx[2],-1.f,L);Adds(tmp[6],tmp[6],float(k),L);
  }
  __aicore__ inline void p2g(GM_ADDR input,GM_ADDR output,GM_ADDR indices,int n,int grid,float dx,float dt,float rpic){
    GlobalTensor<float> src,dst;GlobalTensor<int32_t> idx;src.SetGlobalBuffer((__gm__ float*)input);dst.SetGlobalBuffer((__gm__ float*)output);idx.SetGlobalBuffer((__gm__ int32_t*)indices);
    for(int off=GetBlockIdx()*L;off<n;off+=GetBlockNum()*L){
      for(int z=0;z<27;z++)DataCopy(p[z],src[z*n+off],L);PipeBarrier<PIPE_ALL>();
      weights(dx);
      if(rpic<-.001f){for(int z=6;z<15;z++)Duplicate(p[z],0.f,L);}
      else {
        for(int z=0;z<9;z++)Adds(acc[z],p[6+z],0.f,L);
        for(int r=0;r<3;r++)for(int c=0;c<3;c++){
          Sub(tmp[0],acc[r*3+c],acc[c*3+r],L);Muls(tmp[0],tmp[0],rpic*.5f,L);
          Muls(p[6+r*3+c],acc[r*3+c],1.f-rpic,L);Add(p[6+r*3+c],p[6+r*3+c],tmp[0],L);
        }
      }
      Mul(p[24],p[24],p[26],L);Mul(p[25],p[25],p[26],L);
      for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++){
        int z=i*9+j*3+k;neighbor(i,j,k);
        Mul(tmp[7],tmp[0],p[24],L);PipeBarrier<PIPE_ALL>();DataCopy(dst[z*n+off],tmp[7],L);PipeBarrier<PIPE_ALL>();
        for(int r=0;r<3;r++){
          Mul(tmp[8],p[6+r*3],tmp[4],L);Mul(tmp[9],p[7+r*3],tmp[5],L);Add(tmp[8],tmp[8],tmp[9],L);Mul(tmp[9],p[8+r*3],tmp[6],L);Add(tmp[8],tmp[8],tmp[9],L);
          Muls(tmp[8],tmp[8],dx,L);Add(tmp[8],tmp[8],p[3+r],L);Mul(tmp[8],tmp[8],tmp[7],L);
          Mul(tmp[9],p[15+r*3],tmp[1],L);Mul(tmp[10],p[16+r*3],tmp[2],L);Add(tmp[9],tmp[9],tmp[10],L);Mul(tmp[10],p[17+r*3],tmp[3],L);Add(tmp[9],tmp[9],tmp[10],L);
          Mul(tmp[9],tmp[9],p[25],L);Muls(tmp[9],tmp[9],dt,L);Sub(tmp[8],tmp[8],tmp[9],L);
          PipeBarrier<PIPE_ALL>();DataCopy(dst[(r+1)*27*n+z*n+off],tmp[8],L);PipeBarrier<PIPE_ALL>();
        }
        Muls(tmp[8],base[0],float(grid*grid),L);Muls(tmp[9],base[1],float(grid),L);Add(tmp[8],tmp[8],tmp[9],L);Add(tmp[8],tmp[8],base[2],L);
        Adds(tmp[8],tmp[8],float(i*grid*grid+j*grid+k),L);Maxs(tmp[8],tmp[8],0.f,L);Mins(tmp[8],tmp[8],float(grid*grid*grid-1),L);
        Cast(ids,tmp[8],RoundMode::CAST_TRUNC,L);PipeBarrier<PIPE_ALL>();DataCopy(idx[z*n+off],ids,L);PipeBarrier<PIPE_ALL>();
      }
      PipeBarrier<PIPE_ALL>();
    }
  }
  __aicore__ inline void g2p(GM_ADDR input,GM_ADDR gathered,GM_ADDR output,int n,float dx,float dt){
    GlobalTensor<float> src,gv,dst;src.SetGlobalBuffer((__gm__ float*)input);gv.SetGlobalBuffer((__gm__ float*)gathered);dst.SetGlobalBuffer((__gm__ float*)output);
    for(int off=GetBlockIdx()*L;off<n;off+=GetBlockNum()*L){
      for(int z=0;z<12;z++)DataCopy(p[z],src[z*n+off],L);PipeBarrier<PIPE_ALL>();weights(dx);
      for(int z=0;z<21;z++)Duplicate(acc[z],0.f,L);
      for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++){
        int z=i*9+j*3+k;neighbor(i,j,k);
        for(int r=0;r<3;r++)DataCopy(p[12+r],gv[r*27*n+z*n+off],L);PipeBarrier<PIPE_ALL>();
        for(int r=0;r<3;r++){
          Mul(tmp[7],p[12+r],tmp[0],L);Add(acc[r],acc[r],tmp[7],L);
          for(int c=0;c<3;c++){
            Mul(tmp[8],tmp[7],tmp[4+c],L);Muls(tmp[8],tmp[8],4.f/dx,L);Add(acc[3+r*3+c],acc[3+r*3+c],tmp[8],L);
            Mul(tmp[8],p[12+r],tmp[1+c],L);Add(acc[12+r*3+c],acc[12+r*3+c],tmp[8],L);
          }
        }
        // The next neighbor overwrites p[12:15] via MTE2. Complete the
        // current vector reads (especially the last velocity component) first.
        PipeBarrier<PIPE_ALL>();
      }
      for(int r=0;r<3;r++){
        Muls(tmp[0],acc[r],dt,L);Add(tmp[0],tmp[0],p[r],L);PipeBarrier<PIPE_ALL>();DataCopy(dst[r*n+off],tmp[0],L);DataCopy(dst[(3+r)*n+off],acc[r],L);PipeBarrier<PIPE_ALL>();
      }
      for(int z=0;z<9;z++){PipeBarrier<PIPE_ALL>();DataCopy(dst[(6+z)*n+off],acc[3+z],L);DataCopy(dst[(24+z)*n+off],acc[12+z],L);}
      for(int r=0;r<3;r++)for(int c=0;c<3;c++){
        Duplicate(tmp[0],0.f,L);
        for(int k=0;k<3;k++){
          Muls(tmp[1],acc[12+r*3+k],dt,L);if(r==k)Adds(tmp[1],tmp[1],1.f,L);
          Mul(tmp[2],tmp[1],p[3+k*3+c],L);Add(tmp[0],tmp[0],tmp[2],L);
        }
        PipeBarrier<PIPE_ALL>();DataCopy(dst[(15+r*3+c)*n+off],tmp[0],L);PipeBarrier<PIPE_ALL>();
      }
      PipeBarrier<PIPE_ALL>();
    }
  }
};
extern "C" __global__ __aicore__ void physgm_p2g(GM_ADDR in,GM_ADDR out,GM_ADDR ids,int n,int grid,float dx,float dt,float rpic){Transfer t;t.init();t.p2g(in,out,ids,n,grid,dx,dt,rpic);}
extern "C" __global__ __aicore__ void physgm_g2p(GM_ADDR in,GM_ADDR gv,GM_ADDR out,int n,float dx,float dt){Transfer t;t.init();t.g2p(in,gv,out,n,dx,dt);}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void physgm_p2g_launch(void* in,void* out,void* ids,int n,int grid,float dx,float dt,float rpic,void* stream){physgm_p2g<<<24,nullptr,stream>>>((uint8_t*)in,(uint8_t*)out,(uint8_t*)ids,n,grid,dx,dt,rpic);}
extern "C" void physgm_g2p_launch(void* in,void* gv,void* out,int n,float dx,float dt,void* stream){physgm_g2p<<<24,nullptr,stream>>>((uint8_t*)in,(uint8_t*)gv,(uint8_t*)out,n,dx,dt);}
#endif
