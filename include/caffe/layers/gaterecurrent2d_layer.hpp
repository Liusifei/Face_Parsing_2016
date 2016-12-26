/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_GATERECURRENT2D_LAYER_HPP_
#define CAFFE_GATERECURRENT2D_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GateRecurrent2dLayer : public Layer<Dtype> {/// Gate Recurrent Layer, add by liangji, 20150905
public:
  explicit GateRecurrent2dLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GateRecurrent2d"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  Blob<Dtype> H_;
  
  int num_;
  int height_;
  int width_;
  int channels_;
  bool horizontal_;
  bool reverse_;

};


}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
