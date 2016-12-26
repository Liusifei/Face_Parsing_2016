/*
 * Author: Sifei Liu
*/
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gaterecurrent2dmax_layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

__device__ void get_gate_idx_mx(int h1,int w1,int h2,int w2, int * out,bool horizontal, bool reverse)
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
__device__ Dtype get_data_mx(Dtype * data, int num, int channels,int height, int width,int n,int c,int h,int w)
{
	if(h<0 || h >=height)
		return 0;
	if(w<0 || w >= width)
		return 0;
	
	return data[n*channels*height*width + c * height*width + h * width + w];
}

template <typename Dtype>
__device__ void set_data_mx(Dtype * data, int num, int channels,int height, int width,int n,int c,int h,int w,Dtype v)
{
	if(h<0 || h >=height)
		return ;
	if(w<0 || w >= width)
		return ;
	
	data[n*channels*height*width + c * height*width + h * width + w]=v;
}

template <typename Dtype>
__device__ Dtype get_gate_mx(Dtype * data, int num, int channels,int height, int width,int n,int c,int h1,int w1,int h2,int w2,bool horizontal,bool reverse)
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
		
	get_gate_idx_mx(h1,w1,h2,w2, idx,horizontal, reverse);

	int h = idx[0];
	int w = idx[1];
	
	return data[n*channels*height*width + c * height*width + h * width + w];
}
template <typename Dtype>
__device__ void set_gate_mx(Dtype * data, int num, int channels,int height, int width,int n,int c,int h1,int w1,int h2,int w2,bool horizontal,bool reverse,Dtype v)
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
		
	get_gate_idx_mx(h1,w1,h2,w2, idx,horizontal, reverse);

	int h = idx[0];
	int w = idx[1];
	
	data[n*channels*height*width + c * height*width + h * width + w]=v;
}
template <typename Dtype>
__device__ void set_gate_add_mx(Dtype * data, int num, int channels,int height, int width,int n,int c,int h1,int w1,int h2,int w2,bool horizontal,bool reverse,Dtype v)
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
		
	get_gate_idx_mx(h1,w1,h2,w2, idx,horizontal, reverse);

	int h = idx[0];
	int w = idx[1];
	
	atomicAdd((float *)(data + n*channels*height*width + c * height*width + h * width + w),float(v));
}

template <typename Dtype>
__global__ void get_allgates_mx(const int count, Dtype* G1, Dtype* G2, Dtype* G3) {
CUDA_KERNEL_LOOP(index, count) {
	Dtype g[4];
	int maxid;
	g[0] = G1[index];
	g[1] = G2[index];
	g[2] = G3[index];
	g[3] = 1 - g[0] - g[1] - g[2];
	maxid = g[0] >= g[1] ? 0:1;
	maxid = g[2] >= g[maxid] ? 2:maxid;
	maxid = g[3] >= g[maxid] ? 3:maxid;
	if(maxid == 0){
		G1[index] = 1;
		G2[index] = 0;
		G3[index] = 0;
	}
	if(maxid == 1){
		G1[index] = 0;
		G2[index] = 1;
		G3[index] = 0;
	}
	if(maxid == 2){
		G1[index] = 0;
		G2[index] = 0;
		G3[index] = 1;
	}
	if(maxid == 3){
		G1[index] = 0;
		G2[index] = 0;
		G3[index] = 0;
	}	
}
}


// modified to h(t) = (1-sum(g_i(t))) * x_i(t) + sum(g_i(t) * h_i(t-1))
template <typename Dtype>
__global__ void forward_one_col_left_right(const int count, int T, int num,int channels, int height,  int width,const Dtype* X, const Dtype* G1, const Dtype* G2,const Dtype* G3, Dtype* H, bool horizontal, bool reverse) {
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
	

	Dtype x_data = get_data_mx(X,num,channels,height,width,n,c,h,w);
	

	
	Dtype g_data_1 = get_gate_mx(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
	Dtype h_minus1_data_1 = get_data_mx(H,num,channels,height,width,n,c,h-1,w-1);
	Dtype h1_minus1 = g_data_1 * h_minus1_data_1;

	Dtype g_data_2 = get_gate_mx(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);
	Dtype h_minus1_data_2 = get_data_mx(H,num,channels,height,width,n,c,h,w-1);
	Dtype h2_minus1 = g_data_2 * h_minus1_data_2;

	Dtype g_data_3 = get_gate_mx(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
	Dtype h_minus1_data_3 = get_data_mx(H,num,channels,height,width,n,c,h+1,w-1);
	Dtype h3_minus1 = g_data_3 * h_minus1_data_3;

	Dtype h_hype = h1_minus1 + h2_minus1 + h3_minus1;
	Dtype x_hype = (1 - g_data_1 - g_data_2 - g_data_3) * x_data;

	Dtype h_data = x_hype + h_hype;

	set_data_mx(H,num,channels,height,width,n,c,h,w,h_data);
	

}
}

// modified
template <typename Dtype>
__global__ void forward_one_col_right_left(const int count, int T, int num,int channels, int height,  int width,const Dtype* X, const Dtype* G1, const Dtype* G2,const Dtype* G3, Dtype* H,bool horizontal,bool reverse) {
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
	

	Dtype x_data = get_data_mx(X,num,channels,height,width,n,c,h,w);
	

	Dtype g_data_1 = get_gate_mx(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
	Dtype h_minus1_data_1 = get_data_mx(H,num,channels,height,width,n,c,h-1,w+1);
	Dtype h1_minus1 = g_data_1 * h_minus1_data_1;

	Dtype g_data_2 = get_gate_mx(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);
	Dtype h_minus1_data_2 = get_data_mx(H,num,channels,height,width,n,c,h,w+1);
	Dtype h2_minus1 = g_data_2 * h_minus1_data_2;

	Dtype g_data_3 = get_gate_mx(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
	Dtype h_minus1_data_3 = get_data_mx(H,num,channels,height,width,n,c,h+1,w+1);
	Dtype h3_minus1 = g_data_3 * h_minus1_data_3;

	Dtype h_hype = h1_minus1 + h2_minus1 + h3_minus1;
	Dtype x_hype = (1 - g_data_1 - g_data_2 - g_data_3) * x_data;

	Dtype h_data = x_hype + h_hype;

	set_data_mx(H,num,channels,height,width,n,c,h,w,h_data);
	

}
}

// modified
template <typename Dtype>
__global__ void forward_one_row_top_bottom(const int count, int T, int num,int channels, int height,  int width,const Dtype* X, const Dtype* G1, const Dtype* G2,const Dtype* G3, Dtype* H,bool horizontal,bool reverse) {
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
	

	Dtype x_data = get_data_mx(X,num,channels,height,width,n,c,h,w);
	
	
	Dtype g_data_1 = get_gate_mx(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
	Dtype h_minus1_data_1 = get_data_mx(H,num,channels,height,width,n,c,h-1,w-1);
	Dtype h1_minus1 = g_data_1 * h_minus1_data_1;

	Dtype g_data_2 = get_gate_mx(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);
	Dtype h_minus1_data_2 = get_data_mx(H,num,channels,height,width,n,c,h-1,w);
	Dtype h2_minus1 = g_data_2 * h_minus1_data_2;

	Dtype g_data_3 = get_gate_mx(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
	Dtype h_minus1_data_3 = get_data_mx(H,num,channels,height,width,n,c,h-1,w+1);
	Dtype h3_minus1 = g_data_3 * h_minus1_data_3;

	Dtype h_hype = h1_minus1 + h2_minus1 + h3_minus1;
	Dtype x_hype = (1 - g_data_1 - g_data_2 - g_data_3) * x_data;

	Dtype h_data = x_hype + h_hype;

	set_data_mx(H,num,channels,height,width,n,c,h,w,h_data);
	

}
}


// modified
template <typename Dtype>
__global__ void forward_one_row_bottom_top(const int count, int T, int num,int channels, int height,  int width,const Dtype* X, const Dtype* G1, const Dtype* G2,const Dtype* G3, Dtype* H,bool horizontal,bool reverse) {
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
	

	Dtype x_data = get_data_mx(X,num,channels,height,width,n,c,h,w);
 
	
	Dtype g_data_1 = get_gate_mx(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
	Dtype h_minus1_data_1 = get_data_mx(H,num,channels,height,width,n,c,h+1,w-1);
	Dtype h1_minus1 = g_data_1 * h_minus1_data_1;


	Dtype g_data_2 = get_gate_mx(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);
	Dtype h_minus1_data_2 = get_data_mx(H,num,channels,height,width,n,c,h+1,w);
	Dtype h2_minus1 = g_data_2 * h_minus1_data_2;

	Dtype g_data_3 = get_gate_mx(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
	Dtype h_minus1_data_3 = get_data_mx(H,num,channels,height,width,n,c,h+1,w+1);
	Dtype h3_minus1 = g_data_3 * h_minus1_data_3;

	Dtype h_hype = h1_minus1 + h2_minus1 + h3_minus1;
	Dtype x_hype = (1 - g_data_1 - g_data_2 - g_data_3) * x_data;

	Dtype h_data = x_hype + h_hype;

	set_data_mx(H,num,channels,height,width,n,c,h,w,h_data);
	
}
}

// modified
template <typename Dtype>
__global__ void backward_one_col_left_right(const int count, int T, int num,int channels, int height,  int width,const Dtype* X, const Dtype* G1, const Dtype* G2,const Dtype* G3, const Dtype* H, Dtype * X_diff, Dtype * G1_diff,Dtype* G2_diff,Dtype * G3_diff, Dtype * Hdiff,bool horizontal,bool reverse) {
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


	Dtype x_data = get_data_mx(X,num,channels,height,width,n,c,h,w);
	

	//h(t)_diff = top(t)_diff
	Dtype h_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h,w); 

	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	Dtype add1_h3_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h-1,w+1);
	Dtype add1_g3_data = get_gate_mx(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);

	Dtype add1_h2_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h,w+1);
	Dtype add1_g2_data = get_gate_mx(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);

	Dtype add1_h1_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h+1,w+1);
	Dtype add1_g1_data = get_gate_mx(G1,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
 
	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;

	
	//Hdiff[n*channels*height*width + c*height*width + h*width + w]=0;
	set_data_mx(Hdiff,num,channels,height,width,n,c,h,w,h_diff); 


	//x(t)_diff=(1-sum(g_date))*h(t)_diff
    Dtype g1_data =  get_gate_mx(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
	Dtype g2_data =  get_gate_mx(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);
	Dtype g3_data =  get_gate_mx(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
	
	Dtype x_diff = (1- g1_data -g2_data -g3_data) * h_diff;
	set_data_mx(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	

	// g_diff = h_diff * (h_data(t-1) - x_data)
	Dtype h1_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h-1,w-1); 
	Dtype g1_diff = h_diff * (h1_minus1_data - x_data);
	set_gate_mx(G1_diff,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse,g1_diff);

	Dtype h2_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h,w-1); 
	Dtype g2_diff = h_diff * (h2_minus1_data - x_data);
	set_gate_mx(G2_diff,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse,g2_diff);

	Dtype h3_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h+1,w-1); 
	Dtype g3_diff = h_diff * (h3_minus1_data - x_data);
	set_gate_mx(G3_diff,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse,g3_diff);
	
 
}
}



// modified
template <typename Dtype>
__global__ void backward_one_col_right_left(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3, const Dtype* H, Dtype * X_diff, Dtype * G1_diff,Dtype* G2_diff,Dtype * G3_diff, Dtype * Hdiff,bool horizontal,bool reverse) {
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

	
	Dtype x_data = get_data_mx(X,num,channels,height,width,n,c,h,w);	

	//h(t)_diff = top(t)_diff
	Dtype h_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h,w); 

	///h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	Dtype add1_h3_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h-1,w-1);
	Dtype add1_g3_data = get_gate_mx(G3,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);

	Dtype add1_h2_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h,w-1);
	Dtype add1_g2_data = get_gate_mx(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);

	Dtype add1_h1_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h+1,w-1);
	Dtype add1_g1_data = get_gate_mx(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);

	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;

	
	set_data_mx(Hdiff,num,channels,height,width,n,c,h,w,h_diff); 

    Dtype g1_data =  get_gate_mx(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
	Dtype g2_data =  get_gate_mx(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);
	Dtype g3_data =  get_gate_mx(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
	Dtype x_diff = (1- g1_data -g2_data -g3_data) * h_diff;
	set_data_mx(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	
    // g_diff = h_diff * (h_data(t-1) - x_data)
	Dtype h1_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h-1,w+1); 
	Dtype g1_diff = h_diff * (h1_minus1_data - x_data);
	set_gate_mx(G1_diff,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse,g1_diff);

	
	Dtype h2_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h,w+1); 
	Dtype g2_diff = h_diff * (h2_minus1_data - x_data);
	set_gate_mx(G2_diff,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse,g2_diff);

	Dtype h3_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h+1,w+1); 
	Dtype g3_diff = h_diff * (h3_minus1_data - x_data);
	set_gate_mx(G3_diff,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse,g3_diff);
	
	
}
}


// modified
template <typename Dtype>
__global__ void backward_one_row_top_bottom(const int count, int T, int num,int channels, int height,  int width,const Dtype* X,const Dtype* G1, const Dtype* G2,const Dtype* G3, const Dtype* H, Dtype * X_diff, Dtype * G1_diff,Dtype* G2_diff,Dtype * G3_diff, Dtype * Hdiff,bool horizontal,bool reverse) {
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

	Dtype x_data = get_data_mx(X,num,channels,height,width,n,c,h,w);

	Dtype h_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h,w); 

	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	Dtype add1_h3_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h+1,w-1);
	Dtype add1_g3_data = get_gate_mx(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);

	Dtype add1_h2_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h+1,w);
	Dtype add1_g2_data = get_gate_mx(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);

	Dtype add1_h1_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h+1,w+1);
	Dtype add1_g1_data = get_gate_mx(G1,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);

	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;

	
	set_data_mx(Hdiff,num,channels,height,width,n,c,h,w,h_diff); 


	//x(t)_diff=(1-g(t))*h(t)_diff
	Dtype g1_data =  get_gate_mx(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
	Dtype g2_data =  get_gate_mx(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);
	Dtype g3_data =  get_gate_mx(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
	Dtype x_diff = (1- g1_data -g2_data -g3_data) * h_diff;
	set_data_mx(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	


	// g_diff = h_diff * (h_data(t-1) - x_data)
	Dtype h1_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h-1,w-1); 
	Dtype g1_diff = h_diff * (h1_minus1_data - x_data);
	set_gate_mx(G1_diff,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse,g1_diff);

	Dtype h2_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h-1,w); 
	Dtype g2_diff = h_diff * (h2_minus1_data - x_data);
	set_gate_mx(G2_diff,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse,g2_diff);

	Dtype h3_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h-1,w+1); 
	Dtype g3_diff = h_diff * (h3_minus1_data - x_data);
	set_gate_mx(G3_diff,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse,g3_diff);
	
}
}


//modified
template <typename Dtype>
__global__ void backward_one_row_bottom_top(const int count, int T, int num,int channels, int height,  int width, const Dtype* X, const Dtype* G1, const Dtype* G2,const Dtype* G3, const Dtype* H, Dtype * X_diff, Dtype * G1_diff,Dtype* G2_diff,Dtype * G3_diff, Dtype * Hdiff,bool horizontal,bool reverse) {
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

	Dtype x_data = get_data_mx(X,num,channels,height,width,n,c,h,w);

	//h(t)_diff = top(t)_diff
	Dtype h_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h,w); 

	//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
	Dtype add1_h3_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h-1,w-1);
	Dtype add1_g3_data = get_gate_mx(G3,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);

	Dtype add1_h2_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h-1,w);
	Dtype add1_g2_data = get_gate_mx(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);

	Dtype add1_h1_diff = get_data_mx(Hdiff,num,channels,height,width,n,c,h-1,w+1);
	Dtype add1_g1_data = get_gate_mx(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);

	h_diff = h_diff + add1_h3_diff * add1_g3_data + add1_h2_diff * add1_g2_data + add1_h1_diff * add1_g1_data;

	
	set_data_mx(Hdiff,num,channels,height,width,n,c,h,w,h_diff); 


	//x(t)_diff=(1-g(t))*h(t)_diff
	Dtype g1_data =  get_gate_mx(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
	Dtype g2_data =  get_gate_mx(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);
	Dtype g3_data =  get_gate_mx(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
 	Dtype x_diff = (1- g1_data -g2_data -g3_data) * h_diff;
	set_data_mx(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	

	// g_diff = h_diff * (h_data(t-1) - x_data)
	Dtype h1_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h+1,w-1); 
	Dtype g1_diff = h_diff * (h1_minus1_data - x_data);
	set_gate_mx(G1_diff,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse,g1_diff);

	//Dtype g2_diff = h_diff * g2_idx * x_data * -1;
	Dtype h2_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h+1,w); 
	Dtype g2_diff = h_diff * (h2_minus1_data - x_data);
	set_gate_mx(G2_diff,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse,g2_diff);

	//Dtype g3_diff = h_diff * g3_idx * x_data * -1;
	Dtype h3_minus1_data = get_data_mx(H,num,channels,height,width,n,c,h+1,w+1); 
	Dtype g3_diff = h_diff * (h3_minus1_data - x_data);
	set_gate_mx(G3_diff,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse,g3_diff);
	

}
}

//modified
template <typename Dtype>
void GateRecurrent2dmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    if(maxidpool_) {

		const int count = height_ * width_ * channels_ * num_;
		Dtype* G1_m = bottom[1]->mutable_gpu_data();
		Dtype* G2_m = bottom[2]->mutable_gpu_data();
		Dtype* G3_m = bottom[3]->mutable_gpu_data();
		get_allgates_mx<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, G1_m, G2_m, G3_m);

		CUDA_POST_KERNEL_CHECK;
	}

	const Dtype* X = bottom[0]->gpu_data();
	const Dtype* G1 = bottom[1]->gpu_data();
	const Dtype* G2 = bottom[2]->gpu_data();
	const Dtype* G3 = bottom[3]->gpu_data();
	Dtype * H = top[0]->mutable_gpu_data();

	if(horizontal_ && !reverse_) // left to right
	{
		const int count = height_ * channels_ * num_;

		for(int t=0;t<width_;t++)
		{
			
	
			forward_one_col_left_right<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,H,horizontal_,reverse_);

			CUDA_POST_KERNEL_CHECK;
		}
	}
	else if(horizontal_ && reverse_) // right to left
	{
		const int count = height_ * channels_ * num_;

		for(int t=width_ - 1; t>=0; t--)
		{
			
	
			forward_one_col_right_left<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,H,horizontal_,reverse_);
			CUDA_POST_KERNEL_CHECK;
		}
	}
	else if(!horizontal_ && !reverse_) // top to bottom
	{
		const int count = width_ * channels_ * num_;

		for(int t=0; t< height_; t++)
		{
			
	
			forward_one_row_top_bottom<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,H,horizontal_,reverse_);
			CUDA_POST_KERNEL_CHECK;
		}
	}
	else  //bottom to top
	{
		const int count = width_ * channels_ * num_;

		for(int t=height_-1; t>=0; t--)
		{
			
	
			forward_one_row_bottom_top<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,H,horizontal_,reverse_);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}

template <typename Dtype>
void GateRecurrent2dmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	//if(maxidpool_) {

	//	const int count = height_ * channels_ * num_;
	//	Dtype* G1_m = bottom[1]->mutable_gpu_data();
	//	Dtype* G2_m = bottom[2]->mutable_gpu_data();
	//	Dtype* G3_m = bottom[3]->mutable_gpu_data();
	//	get_allgates_mx<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, G1_m, G2_m, G3_m);

	//	CUDA_POST_KERNEL_CHECK;
	//}


    const Dtype* X = bottom[0]->gpu_data();
	const Dtype* G1 = bottom[1]->gpu_data();
	const Dtype* G2 = bottom[2]->gpu_data();
	const Dtype* G3 = bottom[3]->gpu_data();
	const Dtype * H = top[0]->gpu_data();

    // SaveArray("G1.txt", G1->gpu_data(), G1->count());

	Dtype * H_diff = H_.mutable_gpu_diff();
	caffe_copy(top[0]->count(),top[0]->gpu_diff(),H_diff);

	Dtype * X_diff = bottom[0]->mutable_gpu_diff();
	Dtype * G1_diff = bottom[1]->mutable_gpu_diff();
	Dtype * G2_diff = bottom[2]->mutable_gpu_diff();
	Dtype * G3_diff = bottom[3]->mutable_gpu_diff();

	Dtype * H_cpudiff = H_.mutable_cpu_diff();

	//SaveArray("topdiff.txt", H_.mutable_cpu_diff(),top[0]->count());

	if(horizontal_ && ! reverse_) //left to right
	{
		const int count =  height_ * channels_ * num_;

		for(int t = width_ -1; t>=0; t--)
		{
			backward_one_col_left_right<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,H,X_diff,G1_diff,G2_diff,G3_diff,H_diff,horizontal_, reverse_);

			CUDA_POST_KERNEL_CHECK;

		}
	}
	else if(horizontal_ &&  reverse_) //right to left
	{
		const int count =  height_ * channels_ * num_;

		for(int t = 0; t<width_; t++)
		{
			backward_one_col_right_left<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,H,X_diff,G1_diff,G2_diff,G3_diff,H_diff,horizontal_, reverse_);

			CUDA_POST_KERNEL_CHECK;

		}
	}
	else if(!horizontal_ &&  !reverse_) //top to bottom
	{
		const int count =  width_ * channels_ * num_;
		for(int t = height_-1; t>=0; t--)
		{
			backward_one_row_top_bottom<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,H,X_diff,G1_diff,G2_diff,G3_diff,H_diff,horizontal_, reverse_);

			CUDA_POST_KERNEL_CHECK;

		}
	}
	else  //bottom to top
	{
		const int count =  width_ * channels_ * num_;
		for(int t = 0; t<height_; t++)
		{
			backward_one_row_bottom_top<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t, num_,channels_,height_,width_,X,G1,G2,G3,H,X_diff,G1_diff,G2_diff,G3_diff,H_diff,horizontal_, reverse_);

			CUDA_POST_KERNEL_CHECK;

		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(GateRecurrent2dmaxLayer);

}  // namespace caffe
