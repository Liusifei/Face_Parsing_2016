/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/layers/mask_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_data(),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    if (propagate_down[0]) {
        caffe_gpu_mul(bottom[0]->count(), top[0]->gpu_diff(),bottom[1]->gpu_data(),bottom[0]->mutable_gpu_diff());
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);

}  // namespace caffe
