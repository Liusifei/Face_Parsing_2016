#ifndef CAFFE_CONCAT_LAYER_HPP_
#define CAFFE_CONCAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief 对输入进行l2的归一化（按sample来）
 * 		  Forward:
 * 		  	假设输入的某个sample的feature是[x_1, ..., x_n]，
 * 		  	另r = sqrt(x_1^2 + ... + x_n^2)
 * 		  	那么输出的feature是[x_1 / r, ..., x_n / r]
 *
 * 		  	注意，x_1^2 + ... + x_n^2的和有可能是0，
 * 		  	所以应该加上一个很微小的数，防止除0
 *
 *
 * 		  Backward:
 * 		  	d_i = k_i / r - x_i * \sum_j{x_j * k_j} * r^{-3}
 *
 * 		  	k_i是残差
 */
template <typename Dtype>
class L2NormLayer : public Layer<Dtype> {
 public:
  explicit L2NormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "L2Norm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  //virtual inline DiagonalAffineMap<Dtype> coord_map() {
  //  return DiagonalAffineMap<Dtype>::identity(2);
  //}

  virtual inline pair<Dtype, Dtype> receptive_field(
      const pair<Dtype, Dtype>& output_size, bool flag = true) {
    return output_size;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> norm_, temp_, temp1_;

  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
  Dtype eps_;
};


}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
