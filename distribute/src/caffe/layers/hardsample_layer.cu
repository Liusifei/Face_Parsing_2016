/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/hardsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void HardSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if(bottom.size()>2)
    {
        caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_data(),bottom[2]->gpu_data(), top[0]->mutable_gpu_data());
    }
    else
    {
        caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
    }
}

template <typename Dtype>
void HardSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    Backward_cpu( top, propagate_down, bottom) ;
}

INSTANTIATE_LAYER_GPU_FUNCS(HardSampleLayer);

}  // namespace caffe
