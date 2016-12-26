/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_TRIPLET2_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET2_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe {


template <typename Dtype>
class Triplet2LossLayer : public LossLayer<Dtype> {
 public:
  
  explicit Triplet2LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Triplet2Loss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
 
 protected:
  /// @copydoc SoftmaxWithLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void get_hardest_ignore(const Dtype * fea,const Dtype * label,Dtype* ignore_flag,const int feaNum,const int feaDim);
  Blob<Dtype> diff_ap_; // cached for backward pass. anchor - positive
  Blob<Dtype> diff_an_; //cached for backward pass. anchor - negative
  Blob<Dtype> diff_pn_; //cached for backward pass. positive - negative

  Blob<Dtype> loss_;
  Blob<Dtype> valid_loss_;

  Blob<Dtype> dist_matrix_;
  Blob<Dtype> ignore_flag_;
  Blob<Dtype> temp_;
  
  
  Blob<Dtype> samplecount_;
};


}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
