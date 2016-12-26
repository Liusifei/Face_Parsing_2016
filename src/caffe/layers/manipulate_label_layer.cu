/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/manipulatelabel_layer.hpp"

namespace caffe {

/// @brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void ManipulateLabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		
		Forward_cpu(bottom,top);
}

/// @brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void ManipulateLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		Backward_cpu(top,propagate_down,bottom);
  
}


INSTANTIATE_LAYER_GPU_FUNCS(ManipulateLabelLayer);

}  // namespace caffe
