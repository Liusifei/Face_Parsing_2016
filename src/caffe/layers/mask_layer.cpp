/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/mask_layer.hpp"

namespace caffe {
template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
   CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->height(), bottom[1]->height())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->width(), bottom[1]->width())<< "Inputs must have the same dimension.";
   CHECK(bottom[0]->channels() == bottom[1]->channels())<< "Inputs must have the same dimension.";
   
   top[0]->ReshapeLike(*bottom[0]);
  
}

template <typename Dtype>
void MaskLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
   //CHECK_EQ(bottom[0]->count(), bottom[1]->count())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->height(), bottom[1]->height())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->width(), bottom[1]->width())<< "Inputs must have the same dimension.";
   CHECK(bottom[0]->channels() == bottom[1]->channels() )<< "Inputs must have the same dimension.";
   
   
   top[0]->ReshapeLike(*bottom[0]);
   
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(),bottom[1]->cpu_data(),top[0]->mutable_cpu_data());
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        caffe_mul(bottom[0]->count(), top[0]->cpu_diff(),bottom[1]->cpu_data(),bottom[0]->mutable_cpu_diff());
    }
}

#ifdef CPU_ONLY
STUB_GPU(MaskLayer);
#endif

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
