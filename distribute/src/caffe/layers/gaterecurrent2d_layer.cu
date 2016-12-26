/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gaterecurrent2d_layer.hpp"
#include "caffe/util/io.hpp"

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


template <typename Dtype>
__global__ void forward_one_col_left_right(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3,const Dtype* Idx, Dtype* H,bool horizontal,bool reverse) {
CUDA_KERNEL_LOOP(index, count) {

	
	int hc_count = height * channels;

	int n,c,h,w;
	int temp=index;
	w = T;
	n = temp / hc_count;
	temp = temp % hc_count;
	c = temp / height;
	temp = temp % height;
	h = temp;
	

	Dtype x_data = get_data(X,num,channels,height,width,n,c,h,w);

	
	
	Dtype g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h-1,w-1,horizontal,reverse);
	Dtype g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h,w-1,horizontal,reverse);
	Dtype g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h+1,w-1,horizontal,reverse);

	
	Dtype g_data_1 = get_gate(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
	Dtype h_minus1_data_1 = get_data(H,num,channels,height,width,n,c,h-1,w-1);
	Dtype h1 = (1-g_data_1)*x_data + g_data_1 * h_minus1_data_1;

	Dtype g_data_2 = get_gate(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);
	Dtype h_minus1_data_2 = get_data(H,num,channels,height,width,n,c,h,w-1);
	Dtype h2 = (1-g_data_2)*x_data + g_data_2 * h_minus1_data_2;

	Dtype g_data_3 = get_gate(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
	Dtype h_minus1_data_3 = get_data(H,num,channels,height,width,n,c,h+1,w-1);
	Dtype h3 = (1-g_data_3)*x_data + g_data_3 * h_minus1_data_3;

	Dtype h_data = h1*g1_idx + h2 * g2_idx + h3*g3_idx;

	set_data(H,num,channels,height,width,n,c,h,w,h_data);
	

}
}

template <typename Dtype>
__global__ void forward_one_col_right_left(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3,const Dtype* Idx, Dtype* H,bool horizontal,bool reverse) {
CUDA_KERNEL_LOOP(index, count) {

	
	int hc_count = height * channels;

	int n,c,h,w;
	int temp=index;
	w = T;
	n = temp / hc_count;
	temp = temp % hc_count;
	c = temp / height;
	temp = temp % height;
	h = temp;
	

	Dtype x_data = get_data(X,num,channels,height,width,n,c,h,w);

	
	
	Dtype g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h-1,w+1,horizontal,reverse);
	Dtype g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h,w+1,horizontal,reverse);
	Dtype g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h+1,w+1,horizontal,reverse);

	
	Dtype g_data_1 = get_gate(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
	Dtype h_minus1_data_1 = get_data(H,num,channels,height,width,n,c,h-1,w+1);
	Dtype h1 = (1-g_data_1)*x_data + g_data_1 * h_minus1_data_1;

	Dtype g_data_2 = get_gate(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);
	Dtype h_minus1_data_2 = get_data(H,num,channels,height,width,n,c,h,w+1);
	Dtype h2 = (1-g_data_2)*x_data + g_data_2 * h_minus1_data_2;

	Dtype g_data_3 = get_gate(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
	Dtype h_minus1_data_3 = get_data(H,num,channels,height,width,n,c,h+1,w+1);
	Dtype h3 = (1-g_data_3)*x_data + g_data_3 * h_minus1_data_3;

	Dtype h_data = h1*g1_idx + h2 * g2_idx + h3*g3_idx;

	set_data(H,num,channels,height,width,n,c,h,w,h_data);
	

}
}

template <typename Dtype>
__global__ void forward_one_row_top_bottom(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3,const Dtype* Idx, Dtype* H,bool horizontal,bool reverse) {
CUDA_KERNEL_LOOP(index, count) {

	
	int wc_count = width * channels;

	int n,c,h,w;
	int temp=index;
	h = T;
	n = temp / wc_count;
	temp = temp % wc_count;
	c = temp / width;
	temp = temp % width;
	w = temp;
	

	Dtype x_data = get_data(X,num,channels,height,width,n,c,h,w);

	
	
	Dtype g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h-1,w-1,horizontal,reverse);
	Dtype g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h-1,w,horizontal,reverse);
	Dtype g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h-1,w+1,horizontal,reverse);

	
	Dtype g_data_1 = get_gate(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
	Dtype h_minus1_data_1 = get_data(H,num,channels,height,width,n,c,h-1,w-1);
	Dtype h1 = (1-g_data_1)*x_data + g_data_1 * h_minus1_data_1;

	Dtype g_data_2 = get_gate(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);
	Dtype h_minus1_data_2 = get_data(H,num,channels,height,width,n,c,h-1,w);
	Dtype h2 = (1-g_data_2)*x_data + g_data_2 * h_minus1_data_2;

	Dtype g_data_3 = get_gate(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
	Dtype h_minus1_data_3 = get_data(H,num,channels,height,width,n,c,h-1,w+1);
	Dtype h3 = (1-g_data_3)*x_data + g_data_3 * h_minus1_data_3;

	Dtype h_data = h1*g1_idx + h2 * g2_idx + h3*g3_idx;

	set_data(H,num,channels,height,width,n,c,h,w,h_data);
	

}
}



template <typename Dtype>
__global__ void forward_one_row_bottom_top(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3,const Dtype* Idx, Dtype* H,bool horizontal,bool reverse) {
CUDA_KERNEL_LOOP(index, count) {

	
	int wc_count = width * channels;

	int n,c,h,w;
	int temp=index;
	h = T;
	n = temp / wc_count;
	temp = temp % wc_count;
	c = temp / width;
	temp = temp % width;
	w = temp;
	

	Dtype x_data = get_data(X,num,channels,height,width,n,c,h,w);

	
	
	Dtype g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h+1,w-1,horizontal,reverse);
	Dtype g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h+1,w,horizontal,reverse);
	Dtype g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h+1,w+1,horizontal,reverse);

	
	Dtype g_data_1 = get_gate(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
	Dtype h_minus1_data_1 = get_data(H,num,channels,height,width,n,c,h+1,w-1);
	Dtype h1 = (1-g_data_1)*x_data + g_data_1 * h_minus1_data_1;

	Dtype g_data_2 = get_gate(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);
	Dtype h_minus1_data_2 = get_data(H,num,channels,height,width,n,c,h+1,w);
	Dtype h2 = (1-g_data_2)*x_data + g_data_2 * h_minus1_data_2;

	Dtype g_data_3 = get_gate(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
	Dtype h_minus1_data_3 = get_data(H,num,channels,height,width,n,c,h+1,w+1);
	Dtype h3 = (1-g_data_3)*x_data + g_data_3 * h_minus1_data_3;

	Dtype h_data = h1*g1_idx + h2 * g2_idx + h3*g3_idx;

	set_data(H,num,channels,height,width,n,c,h,w,h_data);
	

}
}


template <typename Dtype>
__global__ void backward_one_col_left_right(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3,const Dtype* Idx, const Dtype* H, Dtype * X_diff, Dtype * G1_diff,Dtype* G2_diff,Dtype * G3_diff,Dtype * Idx_diff,  Dtype * Hdiff,bool horizontal,bool reverse) {
CUDA_KERNEL_LOOP(index, count) {

	
	int hc_count = height * channels;

	int n,c,h,w;
	int temp=index;



	w = T;
	n = temp / hc_count;
	temp = temp % hc_count;
	c = temp / height;
	temp = temp % height;
	h = temp;


	Dtype x_data = get_data(X,num,channels,height,width,n,c,h,w);

	//h(t)_diff = top(t)_diff
	Dtype h_diff = get_data(Hdiff,num,channels,height,width,n,c,h,w); 

	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	Dtype add1_g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h-1,w+1,horizontal,reverse);
	Dtype add1_h3_diff = get_data(Hdiff,num,channels,height,width,n,c,h-1,w+1);
	Dtype add1_g3_data = get_gate(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);

	Dtype add1_g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h,w+1,horizontal,reverse);
	Dtype add1_h2_diff = get_data(Hdiff,num,channels,height,width,n,c,h,w+1);
	Dtype add1_g2_data = get_gate(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);

	Dtype add1_g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h+1,w+1,horizontal,reverse);
	Dtype add1_h1_diff = get_data(Hdiff,num,channels,height,width,n,c,h+1,w+1);
	Dtype add1_g1_data = get_gate(G1,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);

	h_diff = h_diff + add1_g3_idx * add1_h3_diff * add1_g3_data + add1_g2_idx * add1_h2_diff * add1_g2_data + add1_g1_idx * add1_h1_diff * add1_g1_data ;

	
	//Hdiff[n*channels*height*width + c*height*width + h*width + w]=0;
	set_data(Hdiff,num,channels,height,width,n,c,h,w,h_diff); 


	//x(t)_diff=(1-g(t))*h(t)_diff
	Dtype g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h-1,w-1,horizontal,reverse);
	Dtype g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h,w-1,horizontal,reverse);
	Dtype g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h+1,w-1,horizontal,reverse);

	Dtype g1_data =  get_gate(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
	Dtype g2_data =  get_gate(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);
	Dtype g3_data =  get_gate(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);

	Dtype x_diff = (1-g1_data)*h_diff*g1_idx + (1-g2_data)*h_diff*g2_idx + (1-g3_data)*h_diff*g3_idx;
	set_data(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	

	//g(t)_diff = h(t)_diff * x(t) * -1
	//g(t)_diff+=h(t)_diff * h(t-1)if t>0
	Dtype g1_diff = h_diff * g1_idx * x_data * -1;
	Dtype h1_minus1_data = get_data(H,num,channels,height,width,n,c,h-1,w-1); 
	g1_diff = g1_diff + h_diff * g1_idx*h1_minus1_data;
	set_gate(G1_diff,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse,g1_diff);

	Dtype g2_diff = h_diff * g2_idx * x_data * -1;
	Dtype h2_minus1_data = get_data(H,num,channels,height,width,n,c,h,w-1); 
	g2_diff = g2_diff + h_diff * g2_idx*h2_minus1_data;
	set_gate(G2_diff,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse,g2_diff);

	Dtype g3_diff = h_diff * g3_idx * x_data * -1;
	Dtype h3_minus1_data = get_data(H,num,channels,height,width,n,c,h+1,w-1); 
	g3_diff = g3_diff + h_diff * g3_idx*h3_minus1_data;
	set_gate(G3_diff,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse,g3_diff);
	
	//idx_diff = h_diff*( (1-g(t))*x(t) + g(t)*h(t-1)  )
	Dtype g1_idx_diff = h_diff * (  (1-g1_data)*x_data + g1_data*h1_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,0,h,w,h-1,w-1,horizontal,reverse,g1_idx_diff);

	Dtype g2_idx_diff = h_diff * (  (1-g2_data)*x_data + g2_data*h2_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,1,h,w,h,w-1,horizontal,reverse,g2_idx_diff);

	Dtype g3_idx_diff = h_diff * (  (1-g3_data)*x_data + g3_data*h3_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,2,h,w,h+1,w-1,horizontal,reverse,g3_idx_diff);
	
	
}
}




template <typename Dtype>
__global__ void backward_one_col_right_left(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3,const Dtype* Idx, const Dtype* H, Dtype * X_diff, Dtype * G1_diff,Dtype* G2_diff,Dtype * G3_diff,Dtype * Idx_diff,  Dtype * Hdiff,bool horizontal,bool reverse) {
CUDA_KERNEL_LOOP(index, count) {

	
	int hc_count = height * channels;

	int n,c,h,w;
	int temp=index;



	w = T;
	n = temp / hc_count;
	temp = temp % hc_count;
	c = temp / height;
	temp = temp % height;
	h = temp;

	
	

	Dtype x_data = get_data(X,num,channels,height,width,n,c,h,w);

	
	

	//h(t)_diff = top(t)_diff
	Dtype h_diff = get_data(Hdiff,num,channels,height,width,n,c,h,w); 

	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	Dtype add1_g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h-1,w-1,horizontal,reverse);
	Dtype add1_h3_diff = get_data(Hdiff,num,channels,height,width,n,c,h-1,w-1);
	Dtype add1_g3_data = get_gate(G3,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);

	Dtype add1_g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h,w-1,horizontal,reverse);
	Dtype add1_h2_diff = get_data(Hdiff,num,channels,height,width,n,c,h,w-1);
	Dtype add1_g2_data = get_gate(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);

	Dtype add1_g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h+1,w-1,horizontal,reverse);
	Dtype add1_h1_diff = get_data(Hdiff,num,channels,height,width,n,c,h+1,w-1);
	Dtype add1_g1_data = get_gate(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);

	h_diff = h_diff + add1_g3_idx * add1_h3_diff * add1_g3_data + add1_g2_idx * add1_h2_diff * add1_g2_data + add1_g1_idx * add1_h1_diff * add1_g1_data ;

	
	//Hdiff[n*channels*height*width + c*height*width + h*width + w]=0;
	set_data(Hdiff,num,channels,height,width,n,c,h,w,h_diff); 


	//x(t)_diff=(1-g(t))*h(t)_diff
	Dtype g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h-1,w+1,horizontal,reverse);
	Dtype g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h,w+1,horizontal,reverse);
	Dtype g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h+1,w+1,horizontal,reverse);

	Dtype g1_data =  get_gate(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
	Dtype g2_data =  get_gate(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);
	Dtype g3_data =  get_gate(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);

	Dtype x_diff = (1-g1_data)*h_diff*g1_idx + (1-g2_data)*h_diff*g2_idx + (1-g3_data)*h_diff*g3_idx;
	set_data(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	

	//g(t)_diff = h(t)_diff * x(t) * -1
	//g(t)_diff+=h(t)_diff * h(t-1)if t>0
	Dtype g1_diff = h_diff * g1_idx * x_data * -1;
	Dtype h1_minus1_data = get_data(H,num,channels,height,width,n,c,h-1,w+1); 
	g1_diff = g1_diff + h_diff * g1_idx*h1_minus1_data;
	set_gate(G1_diff,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse,g1_diff);

	Dtype g2_diff = h_diff * g2_idx * x_data * -1;
	Dtype h2_minus1_data = get_data(H,num,channels,height,width,n,c,h,w+1); 
	g2_diff = g2_diff + h_diff * g2_idx*h2_minus1_data;
	set_gate(G2_diff,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse,g2_diff);

	Dtype g3_diff = h_diff * g3_idx * x_data * -1;
	Dtype h3_minus1_data = get_data(H,num,channels,height,width,n,c,h+1,w+1); 
	g3_diff = g3_diff + h_diff * g3_idx*h3_minus1_data;
	set_gate(G3_diff,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse,g3_diff);
	
	//idx_diff = h_diff*( (1-g(t))*x(t) + g(t)*h(t-1)  )
	Dtype g1_idx_diff = h_diff * (  (1-g1_data)*x_data + g1_data*h1_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,0,h,w,h-1,w+1,horizontal,reverse,g1_idx_diff);

	Dtype g2_idx_diff = h_diff * (  (1-g2_data)*x_data + g2_data*h2_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,1,h,w,h,w+1,horizontal,reverse,g2_idx_diff);

	Dtype g3_idx_diff = h_diff * (  (1-g3_data)*x_data + g3_data*h3_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,2,h,w,h+1,w+1,horizontal,reverse,g3_idx_diff);
	
	
}
}



template <typename Dtype>
__global__ void backward_one_row_top_bottom(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3,const Dtype* Idx, const Dtype* H, Dtype * X_diff, Dtype * G1_diff,Dtype* G2_diff,Dtype * G3_diff,Dtype * Idx_diff,  Dtype * Hdiff,bool horizontal,bool reverse) {
CUDA_KERNEL_LOOP(index, count) {

	
	int wc_count = width * channels;

	int n,c,h,w;
	int temp=index;
	h = T;
	n = temp / wc_count;
	temp = temp % wc_count;
	c = temp / width;
	temp = temp % width;
	w = temp;

	Dtype x_data = get_data(X,num,channels,height,width,n,c,h,w);

	//h(t)_diff = top(t)_diff
	Dtype h_diff = get_data(Hdiff,num,channels,height,width,n,c,h,w); 

	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	Dtype add1_g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h+1,w-1,horizontal,reverse);
	Dtype add1_h3_diff = get_data(Hdiff,num,channels,height,width,n,c,h+1,w-1);
	Dtype add1_g3_data = get_gate(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);

	Dtype add1_g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h+1,w,horizontal,reverse);
	Dtype add1_h2_diff = get_data(Hdiff,num,channels,height,width,n,c,h+1,w);
	Dtype add1_g2_data = get_gate(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);

	Dtype add1_g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h+1,w+1,horizontal,reverse);
	Dtype add1_h1_diff = get_data(Hdiff,num,channels,height,width,n,c,h+1,w+1);
	Dtype add1_g1_data = get_gate(G1,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);

	h_diff = h_diff + add1_g3_idx * add1_h3_diff * add1_g3_data + add1_g2_idx * add1_h2_diff * add1_g2_data + add1_g1_idx * add1_h1_diff * add1_g1_data ;

	
	//Hdiff[n*channels*height*width + c*height*width + h*width + w]=0;
	set_data(Hdiff,num,channels,height,width,n,c,h,w,h_diff); 


	//x(t)_diff=(1-g(t))*h(t)_diff
	Dtype g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h-1,w-1,horizontal,reverse);
	Dtype g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h-1,w,horizontal,reverse);
	Dtype g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h-1,w+1,horizontal,reverse);

	Dtype g1_data =  get_gate(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
	Dtype g2_data =  get_gate(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);
	Dtype g3_data =  get_gate(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);

	Dtype x_diff = (1-g1_data)*h_diff*g1_idx + (1-g2_data)*h_diff*g2_idx + (1-g3_data)*h_diff*g3_idx;
	set_data(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	

	//g(t)_diff = h(t)_diff * x(t) * -1
	//g(t)_diff+=h(t)_diff * h(t-1)if t>0
	Dtype g1_diff = h_diff * g1_idx * x_data * -1;
	Dtype h1_minus1_data = get_data(H,num,channels,height,width,n,c,h-1,w-1); 
	g1_diff = g1_diff + h_diff * g1_idx*h1_minus1_data;
	set_gate(G1_diff,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse,g1_diff);

	Dtype g2_diff = h_diff * g2_idx * x_data * -1;
	Dtype h2_minus1_data = get_data(H,num,channels,height,width,n,c,h-1,w); 
	g2_diff = g2_diff + h_diff * g2_idx*h2_minus1_data;
	set_gate(G2_diff,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse,g2_diff);

	Dtype g3_diff = h_diff * g3_idx * x_data * -1;
	Dtype h3_minus1_data = get_data(H,num,channels,height,width,n,c,h-1,w+1); 
	g3_diff = g3_diff + h_diff * g3_idx*h3_minus1_data;
	set_gate(G3_diff,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse,g3_diff);
	
	//idx_diff = h_diff*( (1-g(t))*x(t) + g(t)*h(t-1)  )
	Dtype g1_idx_diff = h_diff * (  (1-g1_data)*x_data + g1_data*h1_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,0,h,w,h-1,w-1,horizontal,reverse,g1_idx_diff);

	Dtype g2_idx_diff = h_diff * (  (1-g2_data)*x_data + g2_data*h2_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,1,h,w,h-1,w,horizontal,reverse,g2_idx_diff);

	Dtype g3_idx_diff = h_diff * (  (1-g3_data)*x_data + g3_data*h3_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,2,h,w,h-1,w+1,horizontal,reverse,g3_idx_diff);
	
	
}
}



template <typename Dtype>
__global__ void backward_one_row_bottom_top(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3,const Dtype* Idx, const Dtype* H, Dtype * X_diff, Dtype * G1_diff,Dtype* G2_diff,Dtype * G3_diff,Dtype * Idx_diff,  Dtype * Hdiff,bool horizontal,bool reverse) {
CUDA_KERNEL_LOOP(index, count) {

	
	int wc_count = width * channels;

	int n,c,h,w;
	int temp=index;
	h = T;
	n = temp / wc_count;
	temp = temp % wc_count;
	c = temp / width;
	temp = temp % width;
	w = temp;

	Dtype x_data = get_data(X,num,channels,height,width,n,c,h,w);

	//h(t)_diff = top(t)_diff
	Dtype h_diff = get_data(Hdiff,num,channels,height,width,n,c,h,w); 

	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	Dtype add1_g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h-1,w-1,horizontal,reverse);
	Dtype add1_h3_diff = get_data(Hdiff,num,channels,height,width,n,c,h-1,w-1);
	Dtype add1_g3_data = get_gate(G3,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);

	Dtype add1_g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h-1,w,horizontal,reverse);
	Dtype add1_h2_diff = get_data(Hdiff,num,channels,height,width,n,c,h-1,w);
	Dtype add1_g2_data = get_gate(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);

	Dtype add1_g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h-1,w+1,horizontal,reverse);
	Dtype add1_h1_diff = get_data(Hdiff,num,channels,height,width,n,c,h-1,w+1);
	Dtype add1_g1_data = get_gate(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);

	h_diff = h_diff + add1_g3_idx * add1_h3_diff * add1_g3_data + add1_g2_idx * add1_h2_diff * add1_g2_data + add1_g1_idx * add1_h1_diff * add1_g1_data ;

	
	//Hdiff[n*channels*height*width + c*height*width + h*width + w]=0;
	set_data(Hdiff,num,channels,height,width,n,c,h,w,h_diff); 


	//x(t)_diff=(1-g(t))*h(t)_diff
	Dtype g1_idx = get_gate(Idx,num,3,height,width,n,0,h,w,h+1,w-1,horizontal,reverse);
	Dtype g2_idx = get_gate(Idx,num,3,height,width,n,1,h,w,h+1,w,horizontal,reverse);
	Dtype g3_idx = get_gate(Idx,num,3,height,width,n,2,h,w,h+1,w+1,horizontal,reverse);

	Dtype g1_data =  get_gate(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
	Dtype g2_data =  get_gate(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);
	Dtype g3_data =  get_gate(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);

	Dtype x_diff = (1-g1_data)*h_diff*g1_idx + (1-g2_data)*h_diff*g2_idx + (1-g3_data)*h_diff*g3_idx;
	set_data(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	

	//g(t)_diff = h(t)_diff * x(t) * -1
	//g(t)_diff+=h(t)_diff * h(t-1)if t>0
	Dtype g1_diff = h_diff * g1_idx * x_data * -1;
	Dtype h1_minus1_data = get_data(H,num,channels,height,width,n,c,h+1,w-1); 
	g1_diff = g1_diff + h_diff * g1_idx*h1_minus1_data;
	set_gate(G1_diff,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse,g1_diff);

	Dtype g2_diff = h_diff * g2_idx * x_data * -1;
	Dtype h2_minus1_data = get_data(H,num,channels,height,width,n,c,h+1,w); 
	g2_diff = g2_diff + h_diff * g2_idx*h2_minus1_data;
	set_gate(G2_diff,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse,g2_diff);

	Dtype g3_diff = h_diff * g3_idx * x_data * -1;
	Dtype h3_minus1_data = get_data(H,num,channels,height,width,n,c,h+1,w+1); 
	g3_diff = g3_diff + h_diff * g3_idx*h3_minus1_data;
	set_gate(G3_diff,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse,g3_diff);
	
	//idx_diff = h_diff*( (1-g(t))*x(t) + g(t)*h(t-1)  )
	Dtype g1_idx_diff = h_diff * (  (1-g1_data)*x_data + g1_data*h1_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,0,h,w,h+1,w-1,horizontal,reverse,g1_idx_diff);

	Dtype g2_idx_diff = h_diff * (  (1-g2_data)*x_data + g2_data*h2_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,1,h,w,h+1,w,horizontal,reverse,g2_idx_diff);

	Dtype g3_idx_diff = h_diff * (  (1-g3_data)*x_data + g3_data*h3_minus1_data);
	set_gate_add(Idx_diff,num,3,height,width,n,2,h,w,h+1,w+1,horizontal,reverse,g3_idx_diff);
	
	
}
}

template <typename Dtype>
void GateRecurrent2dLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
	const Dtype* X = bottom[0]->gpu_data();
	const Dtype* G1 = bottom[1]->gpu_data();
	const Dtype* G2 = bottom[2]->gpu_data();
	const Dtype* G3 = bottom[3]->gpu_data();
	const Dtype* Idx = bottom[4]->gpu_data();
	Dtype * H = top[0]->mutable_gpu_data();

	if(horizontal_ && !reverse_) // left to right
	{
		const int count = height_ * channels_ * num_;

		for(int t=0;t<width_;t++)
		{
			
	
			forward_one_col_left_right<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,Idx,H,horizontal_,reverse_);

			CUDA_POST_KERNEL_CHECK;
		}
	}
	else if(horizontal_ && reverse_) // right to left
	{
		const int count = height_ * channels_ * num_;

		for(int t=width_ - 1; t>=0; t--)
		{
			
	
			forward_one_col_right_left<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,Idx,H,horizontal_,reverse_);
			CUDA_POST_KERNEL_CHECK;
		}
	}
	else if(!horizontal_ && !reverse_) // top to bottom
	{
		const int count = width_ * channels_ * num_;

		for(int t=0; t< height_; t++)
		{
			
	
			forward_one_row_top_bottom<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,Idx,H,horizontal_,reverse_);
			CUDA_POST_KERNEL_CHECK;
		}
	}
	else  //bottom to top
	{
		const int count = width_ * channels_ * num_;

		for(int t=height_-1; t>=0; t--)
		{
			
	
			forward_one_row_bottom_top<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,Idx,H,horizontal_,reverse_);
			CUDA_POST_KERNEL_CHECK;
		}
	}


}

template <typename Dtype>
void GateRecurrent2dLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    	const Dtype* X = bottom[0]->gpu_data();
	const Dtype* G1 = bottom[1]->gpu_data();
	const Dtype* G2 = bottom[2]->gpu_data();
	const Dtype* G3 = bottom[3]->gpu_data();
	const Dtype* Idx = bottom[4]->gpu_data();
	const Dtype * H = top[0]->gpu_data();

	Dtype * H_diff = H_.mutable_gpu_diff();
	caffe_copy(top[0]->count(),top[0]->gpu_diff(),H_diff);

	Dtype * X_diff = bottom[0]->mutable_gpu_diff();
	Dtype * G1_diff = bottom[1]->mutable_gpu_diff();
	Dtype * G2_diff = bottom[2]->mutable_gpu_diff();
	Dtype * G3_diff = bottom[3]->mutable_gpu_diff();
	Dtype * Idx_diff = bottom[4]->mutable_gpu_diff();

	Dtype * H_cpudiff = H_.mutable_cpu_diff();

	SaveArray("topdiff.txt", H_.mutable_cpu_diff(),top[0]->count());

	if(horizontal_ && ! reverse_) //left to right
	{
		const int count =  height_ * channels_ * num_;

		for(int t = width_ -1; t>=0; t--)
		{
			backward_one_col_left_right<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,Idx,H,X_diff,G1_diff,G2_diff,G3_diff,Idx_diff,H_diff,horizontal_, reverse_);

			CUDA_POST_KERNEL_CHECK;

		}
	}
	else if(horizontal_ &&  reverse_) //right to left
	{
		const int count =  height_ * channels_ * num_;

		for(int t = 0; t<width_; t++)
		{
			backward_one_col_right_left<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,Idx,H,X_diff,G1_diff,G2_diff,G3_diff,Idx_diff,H_diff,horizontal_, reverse_);

			CUDA_POST_KERNEL_CHECK;

		}
	}
	else if(!horizontal_ &&  !reverse_) //top to bottom
	{
		const int count =  width_ * channels_ * num_;
		for(int t = height_-1; t>=0; t--)
		{
			backward_one_row_top_bottom<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,Idx,H,X_diff,G1_diff,G2_diff,G3_diff,Idx_diff,H_diff,horizontal_, reverse_);

			CUDA_POST_KERNEL_CHECK;

		}
	}
	else  //bottom to top
	{
		const int count =  width_ * channels_ * num_;
		for(int t = 0; t<height_; t++)
		{
			backward_one_row_bottom_top<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,Idx,H,X_diff,G1_diff,G2_diff,G3_diff,Idx_diff,H_diff,horizontal_, reverse_);

			CUDA_POST_KERNEL_CHECK;

		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(GateRecurrent2dLayer);

}  // namespace caffe
