#include <algorithm>
#include <vector>

//#include "caffe/util_other_layers.hpp"
#include"caffe/layers/l2norm_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int num = bottom[0]->num();
  // feature的所有维度
  const int all_fea_count = bottom[0]->count();
  const int dim = all_fea_count / num;

  // 先算出每个数的平方
  caffe_gpu_mul(all_fea_count, bottom_data, bottom_data, temp_.mutable_gpu_data());

  // 然后把得到同一个sample的features的平方和
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, temp_.gpu_data(),
      sum_multiplier_.gpu_data(), 0., norm_.mutable_gpu_data());

  // 开方
  caffe_gpu_powx(num, norm_.gpu_data(), Dtype(0.5),
      norm_.mutable_gpu_data());

  // 加上eps
  caffe_gpu_add_scalar(num, eps_, norm_.mutable_gpu_data());

  // 得到最终norm矩阵
  // 前面算出来的norm矩阵是n*1，为了计算方便，我们需要得到n*d的norm矩阵
  // 其中同一行的d个元素的值都一样，为该sample的归一化银子
  // 所以直接利用让norm矩阵乘以一个1*d维的矩阵（全1）即可得到最终的norm矩阵
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      norm_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());

  // 进行norm
  caffe_gpu_div(all_fea_count, bottom_data, temp_.gpu_data(), top_data);
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  const int num = bottom[0]->num();
  // feature的所有维度
  const int all_fea_count = bottom[0]->count();
  const int dim = all_fea_count / num;

  // d_i = k_i / r - x_i * \sum_j{x_j * k_j} * r^{-3}

  /* cal -x_i * \sum_j{x_j * k_j} * r^{-3} */

  // x_j * k_j，点乘
  caffe_gpu_mul(all_fea_count, bottom_data, top_diff, temp_.mutable_gpu_data());

  // -\sum_j{x_j * k_j}，矩阵乘法
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, temp_.gpu_data(),
      sum_multiplier_.gpu_data(), 0., temp1_.mutable_gpu_data());
  // 变成跟feature同样维度的矩阵（详细看forward函数）
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Dtype(-1),
      temp1_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());
  // -x_i * \sum_j{x_j * k_j}
  caffe_gpu_mul(all_fea_count, bottom_data, temp_.gpu_data(), bottom_diff);
  // r^{-3}
  caffe_gpu_powx(num, norm_.gpu_data(), Dtype(-3), temp_.mutable_gpu_data());
  // 变成跟feature同样维度的矩阵（详细看forward函数）
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      temp_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp1_.mutable_gpu_data());

  // -x_i * \sum_j{x_j * k_j} / r^3
  caffe_gpu_mul(all_fea_count, bottom_diff, temp1_.gpu_data(), bottom_diff);

  /* end x_i * \sum_j{x_j * k_j} * r^{-3} */



  /* cal k_i / r */

  // 变成跟feature同样维度的矩阵（详细看forward函数）
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
      norm_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());
  caffe_gpu_div(all_fea_count, top_diff, temp_.gpu_data(), temp_.mutable_gpu_data());

  /* end k_i / r */


  // final diff
  caffe_gpu_add(all_fea_count, bottom_diff, temp_.gpu_data(), bottom_diff);

}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormLayer);

}  // namespace caffe
