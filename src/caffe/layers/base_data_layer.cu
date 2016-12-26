#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}


template <typename Dtype>
void BasePrefetchingArbitraryDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LiangjiBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  for(int i=0;i<batch->blobs_.size();i++)
  {
      CHECK(top[i]->ShapeEquals(batch->blobs_[i]));
      //top[i]->ReshapeLike(batch->blobs_[i]);
      caffe_copy(batch->blobs_[i]->count(), batch->blobs_[i]->gpu_data(),
             top[i]->mutable_gpu_data());
  }
  
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingArbitraryDataLayer);

}  // namespace caffe
