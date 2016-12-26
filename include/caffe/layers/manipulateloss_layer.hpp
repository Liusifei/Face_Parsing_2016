/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_MANIPULATELOSS_LAYER_HPP_
#define CAFFE_MANIPULATELOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ManipulateLossLayer : public Layer<Dtype> {
 public:
  explicit ManipulateLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ManipulateLoss"; }
 
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  //do not auto set top
  virtual inline bool AutoTopBlobs() const { return false; }
  
 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> mask_;
  bool use_balancesample_;
  vector<Dtype> per_class_statistic;
  vector<Dtype> per_class_balanceRate;
};

}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
