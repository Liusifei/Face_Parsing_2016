/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_TEMPORALLSTM_LAYER_HPP_
#define CAFFE_TEMPORALLSTM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {
template <typename Dtype>
class TemporalLstmLayer : public Layer<Dtype> {/// Spatial Recurrent Layer, add by liangji, 20150112
public:
  explicit TemporalLstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TemporalLstm"; }
  virtual void PreStartSequence();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
   int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;

  Dtype clipping_threshold_; // threshold for clipped gradient
  Blob<Dtype> pre_gate_;  // gate values before nonlinearity
  Blob<Dtype> gate_;      // gate values after nonlinearity
  Blob<Dtype> cell_;      // memory cell;

  Blob<Dtype> prev_cell_; // previous cell state value
  Blob<Dtype> prev_out_;  // previous hidden activation value
  Blob<Dtype> next_cell_; // next cell state value
  Blob<Dtype> next_out_;  // next hidden activation value

  // intermediate values
  Blob<Dtype> fdc_;
  Blob<Dtype> ig_;
  Blob<Dtype> tanh_cell_;

};

}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
