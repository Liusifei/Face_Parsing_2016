/*
* Author: Sifei Liu 
* 3 bottoms, no hard mining
*/
#include <algorithm>  
#include <vector>  

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/triplet01_loss_layer.hpp"
#include "time.h"

namespace caffe {

template <typename Dtype>
void Triplet01LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top); 

	CHECK(bottom[0]->num() == bottom[1]->num());  
	CHECK(bottom[1]->num() == bottom[2]->num());  
	CHECK(bottom[0]->channels() == bottom[1]->channels());  
	CHECK(bottom[1]->channels() == bottom[2]->channels());  
	CHECK(bottom[0]->height() == 1);  
	CHECK(bottom[0]->width() == 1);  
	CHECK(bottom[1]->height() == 1);  
	CHECK(bottom[1]->width() == 1);  
	CHECK(bottom[2]->height() == 1);  
	CHECK(bottom[2]->width() == 1);  

	diff_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
	diff_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
	diff_pn_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  

	diff_sq_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
	diff_sq_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
	dist_sq_ap_.Reshape(bottom[0]->num(), 1, 1, 1);  
	dist_sq_an_.Reshape(bottom[0]->num(), 1, 1, 1); 

	summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
	for (int i = 0; i < bottom[0]->channels(); ++i)  
	summer_vec_.mutable_cpu_data()[i] = Dtype(1);  
/*	dist_binary_.Reshape(bottom[0]->num(), 1, 1, 1);  
	for (int i = 0; i < bottom[0]->num(); ++i)  
	dist_binary_.mutable_cpu_data()[i] = Dtype(1);*/
}



template <typename Dtype>
void Triplet01LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
    LossLayer<Dtype>::Reshape(bottom, top);

    CHECK(bottom[0]->height()== 1);
    CHECK(bottom[0]->width()== 1);
    CHECK(bottom[1]->channels()== 1);
    CHECK(bottom[1]->height()== 1);
    CHECK(bottom[1]->width()== 1);
    CHECK(bottom[1]->num() == bottom[0]->num());

	diff_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
	diff_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
	diff_pn_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  

	diff_sq_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
	diff_sq_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
	dist_sq_ap_.Reshape(bottom[0]->num(), 1, 1, 1);  
	dist_sq_an_.Reshape(bottom[0]->num(), 1, 1, 1); 

	summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
	for (int i = 0; i < bottom[0]->channels(); ++i)  
	summer_vec_.mutable_cpu_data()[i] = Dtype(1);  
/*	dist_binary_.Reshape(bottom[0]->num(), 1, 1, 1);  
	for (int i = 0; i < bottom[0]->num(); ++i)  
	dist_binary_.mutable_cpu_data()[i] = Dtype(1);*/
  
}

template <typename Dtype>  
void Triplet01LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	double dur;
	clock_t start,end;
	start = clock();

	int count = bottom[0]->count();  
	// a_i-p_i for diff_ap_
	caffe_sub(  
		count,  
		bottom[0]->cpu_data(),  // a  
		bottom[1]->cpu_data(),  // p  
		diff_ap_.mutable_cpu_data());   
	// a_i-n_i for diff_an_
	caffe_sub(  
		count,  
		bottom[0]->cpu_data(),  // a  
		bottom[2]->cpu_data(),  // n  
		diff_an_.mutable_cpu_data());   
	// p_i-n_i for diff_pn_
	caffe_sub(  
		count,  
		bottom[1]->cpu_data(),  // p  
		bottom[2]->cpu_data(),  // n  
		diff_pn_.mutable_cpu_data());    
	
	const int channels = bottom[0]->channels();  
	Dtype margin = this->layer_param_.triplet01_loss_param().margin();  

	Dtype loss(0.0);  
	for (int i = 0; i < bottom[0]->num(); ++i) {
		// (a_i - p_i)^2  
		dist_sq_ap_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,  
			diff_ap_.cpu_data() + (i*channels), diff_ap_.cpu_data() + (i*channels));  
		// (a_i - n_i)^2
		dist_sq_an_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,  
			diff_an_.cpu_data() + (i*channels), diff_an_.cpu_data() + (i*channels));  
		// 0.2 + (a_i - p_i)^2 - (a_i - n_i)^2 over all channels
		Dtype mdist = std::max(margin + dist_sq_ap_.cpu_data()[i] - dist_sq_an_.cpu_data()[i], Dtype(0.0));  
		
		loss += mdist;  
		
		if(mdist==Dtype(0)){  
			//dist_binary_.mutable_cpu_data()[i] = Dtype(0);  
			//prepare for backward pass  
			caffe_set(channels, Dtype(0), diff_ap_.mutable_cpu_data() + (i*channels));  
			caffe_set(channels, Dtype(0), diff_an_.mutable_cpu_data() + (i*channels));  
			caffe_set(channels, Dtype(0), diff_pn_.mutable_cpu_data() + (i*channels));  
		}  
	}  
	loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);  
	top[0]->mutable_cpu_data()[0] = loss; 

	end = clock();
	dur = (double)(end - start);
	if(this->layer_param_.triplet01_loss_param().print_time())
		LOG(INFO)<<"Triplet01LossLaye CPU Use Time: "<<(dur/CLOCKS_PER_SEC);
}

template <typename Dtype>  
void Triplet01LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

for (int i = 0; i < 3; ++i) {  
    if (propagate_down[i]) {  

      const Dtype sign = (i < 2) ? -1 : 1;  
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /  
          static_cast<Dtype>(bottom[i]->num());  

      int num = bottom[i]->num();  
      int channels = bottom[i]->channels();  

      for (int j = 0; j < num; ++j) {

        Dtype* bout = bottom[i]->mutable_cpu_diff();  

        if (i==0) {  // a  
              caffe_cpu_axpby(  
                  channels,  
                  alpha,  
                  diff_pn_.cpu_data() + (j*channels),  
                  Dtype(0.0),  
                  bout + (j*channels));
        } else if (i==1) {  // p  
              caffe_cpu_axpby(  
                  channels,  
                  alpha,  
                  diff_ap_.cpu_data() + (j*channels),  
                  Dtype(0.0),  
                  bout + (j*channels));   
        } else if (i==2) {  // n  
              caffe_cpu_axpby(  
                  channels,  
                  alpha,  
                  diff_an_.cpu_data() + (j*channels),  
                  Dtype(0.0),  
                  bout + (j*channels));  
        } 
      } // for batch num 
    } //if propagate_down[i]  
  } //for bottom  
}

#ifdef CPU_ONLY
STUB_GPU(Triplet01LossLayer);
#endif

INSTANTIATE_CLASS(Triplet01LossLayer);
REGISTER_LAYER_CLASS(Triplet01Loss);

}  // namespace caffe