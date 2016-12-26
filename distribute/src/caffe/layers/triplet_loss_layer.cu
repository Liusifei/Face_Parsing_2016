/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"
#include "time.h"

namespace caffe {

template <typename Dtype>
__global__ void TripletLossCalc_1(const int n, const int step,const int datacount, const Dtype* bottom_data, Dtype* bottom_diff,const Dtype* label,const Dtype margin,const Dtype * lossweight,Dtype* loss,Dtype* losscount, Dtype * validlosscount) {
  CUDA_KERNEL_LOOP(index, n) {
	int v=index;
	int ida,idp,idn;
	float dist_sq_ap, dist_sq_an;
	float diff_ap,diff_an,diff_pn;
	float mdist;
	float la,lp,ln;
	int idx;

	ida = v%step;
	
	for(idx=0;idx<step*step;idx++)
	{
		idp= idx/step;
		idn= idx%step;

		if(ida == idp)
			continue;
		la = label[ida];
		lp = label[idp];
		ln = label[idn];

		const Dtype * a = bottom_data + ida*datacount;
		const Dtype * p = bottom_data + idp*datacount;
		const Dtype * n = bottom_data + idn*datacount;
		Dtype * a_diff = bottom_diff + ida*datacount;
		Dtype * p_diff = bottom_diff + idp*datacount;
		Dtype * n_diff = bottom_diff + idn*datacount;
		float ai,pi,ni;
		
		if((int)la == (int)lp && (int)la != (int)ln)
		{//find one triplet

		

			dist_sq_ap = 0;
			dist_sq_an = 0;
			for(int i=0;i<datacount;i++)
			{
				ai=a[i];
				pi=p[i];
				ni=n[i];
				dist_sq_ap += (ai-pi)*(ai-pi);
				dist_sq_an += (ai-ni)*(ai-ni);
			}
		
			mdist = margin + dist_sq_ap- dist_sq_an;
			if(mdist <0)
				mdist = 0;
			
			if(mdist > 0)
			{
				for(int i=0;i<datacount;i++)
				{
					ai=a[i];
					pi=p[i];
					ni=n[i];
					diff_ap = ai-pi;
					diff_an = ai-ni;
					diff_pn = pi-ni;
					atomicAdd((float *)(a_diff+i),float(-1*lossweight[0]*diff_pn));
					atomicAdd((float *)(p_diff+i),float(-1*lossweight[0]*diff_ap));
					atomicAdd((float *)(n_diff+i),float(lossweight[0]*diff_an));
				}
				int id_loss = index % (step*step);
				atomicAdd((float *)(validlosscount+id_loss),float(1));	
			}

			int id_loss = index % (step*step);	
			atomicAdd((float *)(loss+id_loss),float(mdist));
			atomicAdd((float *)(losscount+id_loss),float(1));
	
		}//end find one triplet

	}//end for idn
   
  }//end loop
}


template <typename Dtype>
__global__ void TripletLossCalc_2(const int n, const int step,const int datacount, const Dtype* bottom_data, Dtype* bottom_diff,const Dtype* label,const Dtype margin,const Dtype * lossweight,Dtype* loss,Dtype* losscount, Dtype * validlosscount) {
  CUDA_KERNEL_LOOP(index, n) {
	int v=index;
	int ida,idp,idn;
	float dist_sq_ap, dist_sq_an;
	float diff_ap,diff_an,diff_pn;
	float mdist;
	float la,lp,ln;

	ida = (int)(v/step);
	idp = v%step;

	for(idn=0;idn<step && ida!=idp ;idn++)
	{
		la = label[ida];
		lp = label[idp];
		ln = label[idn];

		const Dtype * a = bottom_data + ida*datacount;
		const Dtype * p = bottom_data + idp*datacount;
		const Dtype * n = bottom_data + idn*datacount;
		Dtype * a_diff = bottom_diff + ida*datacount;
		Dtype * p_diff = bottom_diff + idp*datacount;
		Dtype * n_diff = bottom_diff + idn*datacount;
		float ai,pi,ni;
		
		if((int)la == (int)lp && (int)la != (int)ln)
		{//find one triplet

		

			dist_sq_ap = 0;
			dist_sq_an = 0;
			for(int i=0;i<datacount;i++)
			{
				ai=a[i];
				pi=p[i];
				ni=n[i];
				dist_sq_ap += (ai-pi)*(ai-pi);
				dist_sq_an += (ai-ni)*(ai-ni);
			}
		
			mdist = margin + dist_sq_ap- dist_sq_an;
			if(mdist <0)
				mdist = 0;
			
			if(mdist > 0)
			{
				for(int i=0;i<datacount;i++)
				{
					ai=a[i];
					pi=p[i];
					ni=n[i];
					diff_ap = ai-pi;
					diff_an = ai-ni;
					diff_pn = pi-ni;
					atomicAdd((float *)(a_diff+i),float(-1*lossweight[0]*diff_pn));
					atomicAdd((float *)(p_diff+i),float(-1*lossweight[0]*diff_ap));
					atomicAdd((float *)(n_diff+i),float(lossweight[0]*diff_an));
				}
				int id_loss = index % (step*step);
				atomicAdd((float *)(validlosscount+id_loss),float(1));	
			}

			int id_loss = index % (step*step);	
			atomicAdd((float *)(loss+id_loss),float(mdist));
			atomicAdd((float *)(losscount+id_loss),float(1));
	
		}//end find one triplet

	}//end for idn
   
  }//end loop
}


template <typename Dtype>
__global__ void TripletLossCalc_3(const int n, const int step,const int datacount, const Dtype* bottom_data, Dtype* bottom_diff,const Dtype* label,const Dtype margin,const Dtype * lossweight,Dtype* loss,Dtype* losscount, Dtype * validlosscount) {
  CUDA_KERNEL_LOOP(index, n) {
	int v=index;
	int ida,idp,idn;
	float dist_sq_ap, dist_sq_an;
	float diff_ap,diff_an,diff_pn;
	float mdist;

	idn = v%step;
	v = v - idn;
	idp = v%(step*step)/step;
	v = v -idp*step;
	ida = v/(step*step);

	float la,lp,ln;
	la = label[ida];
	lp = label[idp];
	ln = label[idn];

	const Dtype * a = bottom_data + ida*datacount;
	const Dtype * p = bottom_data + idp*datacount;
	const Dtype * n = bottom_data + idn*datacount;
	Dtype * a_diff = bottom_diff + ida*datacount;
	Dtype * p_diff = bottom_diff + idp*datacount;
	Dtype * n_diff = bottom_diff + idn*datacount;
	float ai,pi,ni;

	if((int)la == (int)lp && (int)la != (int)ln && ida!=idp)
	{//find one triplet

		

		dist_sq_ap = 0;
		dist_sq_an = 0;
		for(int i=0;i<datacount;i++)
		{
			ai=a[i];
			pi=p[i];
			ni=n[i];
			dist_sq_ap += (ai-pi)*(ai-pi);
			dist_sq_an += (ai-ni)*(ai-ni);
		}
		
		mdist = margin + dist_sq_ap- dist_sq_an;
		if(mdist <0)
			mdist = 0;
			
		if(mdist > 0)
		{
			for(int i=0;i<datacount;i++)
			{
				ai=a[i];
				pi=p[i];
				ni=n[i];
				diff_ap = ai-pi;
				diff_an = ai-ni;
				diff_pn = pi-ni;
				atomicAdd((float *)(a_diff+i),float(-1*lossweight[0]*diff_pn));
				atomicAdd((float *)(p_diff+i),float(-1*lossweight[0]*diff_ap));
				atomicAdd((float *)(n_diff+i),float(lossweight[0]*diff_an));
			}
			int id_loss = index % (step*step);
			atomicAdd((float *)(validlosscount+id_loss),float(1));	
		}

		int id_loss = index % (step*step);	
		atomicAdd((float *)(loss+id_loss),float(mdist));
		atomicAdd((float *)(losscount+id_loss),float(1));
	
	}//end find one triplet
   
  }//end loop
}



template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	if(this->layer_param_.triplet_loss_param().use_cpu())
	{
		this->Forward_cpu(bottom,top);
		return;
	}
	
	double dur;
	clock_t start,end;
	start = clock();

	//LOG(INFO)<<"--> TripletLossLayer<Dtype>::Forward_gpu";
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	caffe_gpu_set(bottom[0]->count(),Dtype(0.0),bottom_diff);
	Dtype * diff_ap = diff_ap_.mutable_gpu_data();
	Dtype * diff_an = diff_an_.mutable_gpu_data();
	Dtype * diff_pn = diff_pn_.mutable_gpu_data();
	Dtype * samplecount = samplecount_.mutable_gpu_data();
	//LOG(INFO)<<"before set";
	caffe_gpu_set(loss_.count(),Dtype(0),loss_.mutable_gpu_data());
	caffe_gpu_set(loss_.count(),Dtype(0),loss_.mutable_gpu_diff());
	caffe_gpu_set(valid_loss_.count(),Dtype(0),valid_loss_.mutable_gpu_data());
	//LOG(INFO)<<"after set";
	Dtype * loss = loss_.mutable_gpu_data();
	Dtype * losscount = loss_.mutable_gpu_diff();
        Dtype * validlosscount = valid_loss_.mutable_gpu_data();
	const Dtype* label = bottom[1]->gpu_data();
	//LOG(INFO)<<"after 1";
	Dtype margin = this->layer_param_.triplet_loss_param().margin();// alpha
	//LOG(INFO)<<"after 1-1";
	const Dtype * lossweight = top[0]->gpu_diff();
	//const Dtype lossweight =  topdiff[0];
	//LOG(INFO)<<"after 1-2";
	int datacount = bottom[0]->count() / bottom[0]->num();
	const int step = bottom[0]->num();
	//LOG(INFO)<<"after 2";

	if(this->layer_param_.triplet_loss_param().step()==3)
	{
		const int count = step * step * step;
	
		TripletLossCalc_3<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, step, datacount,bottom_data, bottom_diff,label,margin,lossweight,loss,losscount,validlosscount);
		CUDA_POST_KERNEL_CHECK;

	}
	else if(this->layer_param_.triplet_loss_param().step()==2)
	{
		const int count = step * step;
	
		TripletLossCalc_2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, step, datacount,bottom_data, bottom_diff,label,margin,lossweight,loss,losscount,validlosscount);
		CUDA_POST_KERNEL_CHECK;
	}
	else if(this->layer_param_.triplet_loss_param().step()==1)
	{
		const int count = step;
	
		TripletLossCalc_1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, step, datacount,bottom_data, bottom_diff,label,margin,lossweight,loss,losscount,validlosscount);
		CUDA_POST_KERNEL_CHECK;
	}
	else
		LOG(FATAL)<<"TripletLoss step error! Must be 1~3, now get:"<<this->layer_param_.triplet_loss_param().step();



	Dtype totalcount = caffe_cpu_asum(loss_.count(),loss_.cpu_diff());
	Dtype asumloss = caffe_cpu_asum(loss_.count(),loss_.cpu_data());
	Dtype valid_loss_count = caffe_cpu_asum(valid_loss_.count(),valid_loss_.cpu_data());
	if(totalcount>0)
		top[0]->mutable_cpu_data()[0] = asumloss/totalcount/Dtype(2);
	else
		top[0]->mutable_cpu_data()[0] = 0;

	if(this->layer_param_.triplet_loss_param().norm_diff() && valid_loss_count>0)
	{       LOG(INFO)<<"valid loss count = "<<(int)valid_loss_count<<"; total loss count = "<<totalcount<<"; ratio = "<<valid_loss_count/totalcount;
		caffe_gpu_scal(bottom[0]->count(),Dtype(1.0/valid_loss_count), bottom[0]->mutable_gpu_diff());
	}
	
	

	end = clock();
	dur = (double)(end - start);
	if(this->layer_param_.triplet_loss_param().print_time())
		LOG(INFO)<<"TripletLossLaye GPU Use Time: "<<(dur/CLOCKS_PER_SEC);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	this->Backward_cpu(top,propagate_down,bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);
}  // namespace caffe
