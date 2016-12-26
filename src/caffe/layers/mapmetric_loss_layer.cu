/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/mapmetricloss_layer.hpp"


namespace caffe {

template <typename Dtype>
void MapMetricLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    this->Forward_cpu(bottom,top);
}



template <typename Dtype>
void MapMetricLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    this->Backward_cpu(top,propagate_down,bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(MapMetricLossLayer);

}  // namespace caffe
