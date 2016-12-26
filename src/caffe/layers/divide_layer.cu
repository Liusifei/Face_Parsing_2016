#include <algorithm>
#include <vector>
#include "caffe/layers/divide_layer.hpp"
//#include "caffe/layer.hpp"
//#include "caffe/layers/neuron_layer.hpp"

// out = 1./in
namespace caffe {

template <typename Dtype>
__global__ void DividForward(const int n, Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
  	if (in[index] == 0)
  	{
  		in[index] = Dtype(0.0001);
  	}
    out[index] = Dtype(1) / in[index];
  }
}

template <typename Dtype>
__global__ void DividBackward(const int n, const Dtype* in_diff, Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
  	if (in_data[index] == 0)
  	{
  		in_data[index] = Dtype(0.0001);
  	}
    out_diff[index] = (Dtype(-1) / (in_data[index] * in_data[index])) * in_diff[index];
  }
}

template <typename Dtype>
void DivideLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  //caffe_gpu_add_scalar(count, eps_, bottom_data);
  //caffe_gpu_div(count, Dtype(1), bottom_data, top_data);
  DividForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void DivideLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    DividBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DivideLayer);
} // namespace caffe