#ifndef CAFFE_UPSCALE_LAYER_HPP_
#define CAFFE_UPSCALE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class UpscaleLayer : public Layer<Dtype>{
public:
	explicit UpscaleLayer(const LayerParameter& param)
	: Layer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Upscale"; }

	virtual inline bool EqualNumBottomTopBlobs() const { return true; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		      const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};





}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
