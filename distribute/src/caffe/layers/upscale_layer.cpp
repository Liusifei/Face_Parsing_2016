#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/upscale_layer.hpp"
#include "caffe/util/upscale.hpp"
namespace caffe {

template <typename Dtype>
void UpscaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void UpscaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    Blob<Dtype>& src = *(bottom[i]);
    Blob<Dtype>& dst = *(top[i]);
    std::vector<int> infered_dst_shape = src.shape();
    CHECK_GE(infered_dst_shape.size(), 2);
    int last_dim = infered_dst_shape.size() -1;
    infered_dst_shape[last_dim] *= 2;
    infered_dst_shape[last_dim-1] *= 2;
    dst.Reshape(infered_dst_shape);
  }
}

template <typename Dtype>
void UpscaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    Blob<Dtype>& src = *(bottom[i]);
    Blob<Dtype>& dst = *(top[i]);
//    upscale_2x_cpu(src, dst);
    Blob2xUpscaler<Dtype>::Forward_cpu(src, dst);
  }
}

template <typename Dtype>
void UpscaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    Blob<Dtype>& src = *(bottom[i]);
    Blob<Dtype>& dst = *(top[i]);
    caffe::caffe_set(src.count(), Dtype(0), src.mutable_cpu_diff());
//    upscale_2x_grad_cpu(dst, src);
    Blob2xUpscaler<Dtype>::Backward_cpu(dst, src);
  }
}






#ifdef CPU_ONLY
STUB_GPU(UpscaleLayer);
#else

template <typename Dtype>
void UpscaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    Blob<Dtype>& src = *(bottom[i]);
    Blob<Dtype>& dst = *(top[i]);
    Blob2xUpscaler<Dtype>::Forward_gpu(src, dst);
  }
}

template <typename Dtype>
void UpscaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    Blob<Dtype>& src = *(bottom[i]);
    Blob<Dtype>& dst = *(top[i]);
    caffe::caffe_gpu_set(src.count(), Dtype(0), src.mutable_gpu_diff());
    Blob2xUpscaler<Dtype>::Backward_gpu(dst, src);
  }
}


#endif

INSTANTIATE_CLASS(UpscaleLayer);
REGISTER_LAYER_CLASS(Upscale);

}  // namespace caffe
