#include <algorithm>
#include <vector>
#include "caffe/layers/divide_layer.hpp"
//#include "caffe/layer.hpp"
//#include "caffe/layers/neuron_layer.hpp"

namespace caffe {


template <typename Dtype>
void DivideLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	for (int i = 0; i < count; ++i)
	{
		if (bottom_data[i]==0)
		{
			bottom_data[i] = Dtype(0.0001);
		}
		top_data[i] = Dtype(1) / bottom_data[i];
	}
}

template <typename Dtype>
void DivideLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_data = bottom[0]->mutable_cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i)
	{
		if (bottom_data[i]==0)
		{
			bottom_data[i] = Dtype(0.0001);
		}
		bottom_diff[i] = (Dtype(-1) / (bottom_data[i] * bottom_data[i])) * top_diff[i];
	}
}

#ifdef CPU_ONLY
STUB_GPU(DivideLayer);
#endif

INSTANTIATE_CLASS(DivideLayer);
REGISTER_LAYER_CLASS(Divide);

}// namespace caffe