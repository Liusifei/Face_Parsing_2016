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
#include "caffe/layers/pad_layer.hpp"

namespace caffe {

template <typename Dtype>
void PadLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 
  pad_h_ = this->layer_param_.pad_param().pad_h();
  pad_w_ = this->layer_param_.pad_param().pad_w();

}

template <typename Dtype>
void PadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

   top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height()+2*pad_h_,bottom[0]->width()+2*pad_w_);
      
   start_w_ = pad_w_;
   start_h_ = pad_h_;
  
}

template <typename Dtype>
void PadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
    

    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_set(top[0]->count(), static_cast<Dtype>(0), top_data);

    for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int c = 0; c < bottom[0]->channels(); ++c) {
          for (int h = 0; h < bottom[0]->height(); ++h) {
            caffe_copy(bottom[0]->width(),
                bottom_data + bottom[0]->offset(n, c, h),
		top_data + top[0]->offset(n, c, start_h_ + h, start_w_));
          }
        }
    }
           
}

template <typename Dtype>
void PadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);

    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        for (int h = 0; h < bottom[0]->height(); ++h) {
          caffe_copy(bottom[0]->width(),
              top_diff + top[0]->offset(n, c, start_h_ + h, start_w_),
              bottom_diff + bottom[0]->offset(n, c, h));
        }
      }
    }
  }     
}

#ifdef CPU_ONLY
STUB_GPU(PadLayer);
#endif

INSTANTIATE_CLASS(PadLayer);
REGISTER_LAYER_CLASS(Pad);

}  // namespace caffe
