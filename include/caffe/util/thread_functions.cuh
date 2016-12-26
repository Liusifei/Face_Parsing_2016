#ifndef CAFFE_UTIL_UTIL_THREAD_H_
#define CAFFE_UTIL_UTIL_THREAD_H_

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/gpu_util.cuh"

using namespace std;

namespace caffe {

__device__ void get_gate_idx(int h1,int w1,int h2,int w2, int * out,bool horizontal, bool reverse)
{
	if(horizontal && ! reverse) // left -> right
	{
		if(w1>w2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(horizontal && reverse)  // right -> left
	{
		if(w1<w2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(!horizontal && !reverse)  // top  -> bottom
	{
		if(h1>h2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(!horizontal && reverse)  // bottom -> top
	{
		if(h1<h2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}

}

template <typename Dtype>
__device__ Dtype get_data(Dtype * data, int num, int channels,int height, int width,int n,int c,int h,int w)
{
	if(h<0 || h >=height)
		return 0;
	if(w<0 || w >= width)
		return 0;
	
	return data[n*channels*height*width + c * height*width + h * width + w];
}

template <typename Dtype>
__device__ void set_data(Dtype * data, int num, int channels,int height, int width,int n,int c,int h,int w,Dtype v)
{
	if(h<0 || h >=height)
		return ;
	if(w<0 || w >= width)
		return ;
	
	data[n*channels*height*width + c * height*width + h * width + w]=v;
}

template <typename Dtype>
__device__ Dtype get_gate(Dtype * data, int num, int channels,int height, int width,int n,int c,int h1,int w1,int h2,int w2,bool horizontal,bool reverse)
{
	if(h1<0 || h1 >=height)
		return 0;
	if(w1<0 || w1 >= width)
		return 0;
	if(h2<0 || h2 >=height)
		return 0;
	if(w2<0 || w2 >= width)
		return 0;
	int idx[2];
		
	get_gate_idx(h1,w1,h2,w2, idx,horizontal, reverse);

	int h = idx[0];
	int w = idx[1];
	
	return data[n*channels*height*width + c * height*width + h * width + w];
}
template <typename Dtype>
__device__ void set_gate(Dtype * data, int num, int channels,int height, int width,int n,int c,int h1,int w1,int h2,int w2,bool horizontal,bool reverse,Dtype v)
{
	if(h1<0 || h1 >=height)
		return ;
	if(w1<0 || w1 >= width)
		return ;
	if(h2<0 || h2 >=height)
		return ;
	if(w2<0 || w2 >= width)
		return ;
	int idx[2];
		
	get_gate_idx(h1,w1,h2,w2, idx,horizontal, reverse);

	int h = idx[0];
	int w = idx[1];
	
	data[n*channels*height*width + c * height*width + h * width + w]=v;
}
template <typename Dtype>
__device__ void set_gate_add(Dtype * data, int num, int channels,int height, int width,int n,int c,int h1,int w1,int h2,int w2,bool horizontal,bool reverse,Dtype v)
{
	if(h1<0 || h1 >=height)
		return ;
	if(w1<0 || w1 >= width)
		return ;
	if(h2<0 || h2 >=height)
		return ;
	if(w2<0 || w2 >= width)
		return ;
	int idx[2];
		
	get_gate_idx(h1,w1,h2,w2, idx,horizontal, reverse);

	int h = idx[0];
	int w = idx[1];
	
	atomicAdd((float *)(data + n*channels*height*width + c * height*width + h * width + w),float(v));
}


} // namespace caffe

#endif   // CAFFE_UTIL_UTIL_THREAD_H_
