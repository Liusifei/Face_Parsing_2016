#ifndef CAFFE_DIVIDE_LAYER_HPP_
#define CAFFE_DIVIDE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Tests whether the input exceeds a threshold: outputs 1 for inputs
 *        above threshold; 0 otherwise.
 */
template <typename Dtype>
class DivideLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * out = 1./in;
   * dout/din = -1./(in)^2
   */
  explicit DivideLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  //virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Divide"; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //const Dtype eps_ = Dtype(0.0001);
};

}  // namespace caffe

#endif  // CAFFE_DIVIDE_LAYER_HPP_
