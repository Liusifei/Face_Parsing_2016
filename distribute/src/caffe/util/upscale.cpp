/*
 * upscale.cpp
 *
 *      Author: Alan_Huang
 */


#include "caffe/util/upscale.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
std::ostream &operator<<(std::ostream &os,const Blob<Dtype> &inst) {
  os << "shape: " << inst.shape_string() << " data: [" << std::endl;
  std::vector<int> inst_shape = inst.shape();
  CHECK_GT(inst_shape.size(), 0);
  CHECK_GE(inst_shape.size(), 4);
  for (int n = 0; n < inst.num(); ++n) {
    os << "[ ";
    for (int c = 0; c < inst.channels(); ++c) {
      os << "[ ";
      for (int h = 0; h < inst.height(); ++h) {
        for (int w = 0; w < inst.width(); ++w) {
          os << inst.data_at(n,c,h,w) << " ";
        }
        os << std::endl;
      }
      os << " ]" << std::endl;
    }
    os << " ]" << std::endl;
  }
  os <<" ] diff: [" ;
  for (int n = 0; n < inst.num(); ++n) {
    os << "[ ";
    for (int c = 0; c < inst.channels(); ++c) {
      os << "[ ";
      for (int h = 0; h < inst.height(); ++h) {
        for (int w = 0; w < inst.width(); ++w) {
          os << inst.diff_at(n,c,h,w) << " ";
        }
        os << std::endl;
      }
      os << " ]" << std::endl;
    }
    os << " ]" << std::endl;
  }
  os <<" ]";
  return os;
}

template std::ostream &operator<<(std::ostream &os,const Blob<float> &inst);
template std::ostream &operator<<(std::ostream &os,const Blob<double> &inst);


template <typename Dtype>
void Blob2xUpscaler<Dtype>::Check(const Blob<Dtype>& src_blob,
    const Blob<Dtype>& dst_blob) {
  std::vector<int> infered_dst_shape = src_blob.shape();
  CHECK_GE(infered_dst_shape.size(), 2);
  int last_dim = infered_dst_shape.size() -1;
  infered_dst_shape[last_dim] *= 2;
  infered_dst_shape[last_dim-1] *= 2;
  CHECK(infered_dst_shape == dst_blob.shape());
}


template <typename Dtype>
void Blob2xUpscaler<Dtype>::Forward_cpu(const Blob<Dtype>& src_blob,
    Blob<Dtype>& dst_blob) {
  Blob2xUpscaler<Dtype>::Check(src_blob, dst_blob);
  int last_dim = src_blob.shape().size() - 1;
  int total_channel_num = src_blob.count(0, src_blob.shape().size() - 2);
  int src_spatial_dim = src_blob.count(last_dim - 1);
  int dst_spatial_dim = dst_blob.count(last_dim - 1);
  int src_h = src_blob.shape(last_dim - 1);
  int src_w = src_blob.shape(last_dim);

  const Dtype* src_data = src_blob.cpu_data();
  Dtype* dst_data = dst_blob.mutable_cpu_data();
  for (int i = 0; i < total_channel_num; ++i) {
    upscale_2x_cpu<PointInterpolateForward>(
        const_cast<Dtype*>(src_data), src_h, src_w, dst_data);
    src_data += src_spatial_dim;
    dst_data += dst_spatial_dim;
  }
}

//template <typename Dtype>
//void upscale_2x_cpu(const Blob<Dtype>& src_blob, Blob<Dtype>& dst_blob) {
//  std::vector<int> infered_dst_shape = src_blob.shape();
//  CHECK_GE(infered_dst_shape.size(), 2);
//  int last_dim = infered_dst_shape.size() -1;
//  infered_dst_shape[last_dim] *= 2;
//  infered_dst_shape[last_dim-1] *= 2;
//  CHECK(infered_dst_shape == dst_blob.shape());
//
//  int total_channel_num = src_blob.count(0, infered_dst_shape.size() - 2);
//  int src_spatial_dim = src_blob.count(last_dim - 1);
//  int dst_spatial_dim = dst_blob.count(last_dim - 1);
//  int src_h = src_blob.shape(last_dim - 1);
//  int src_w = src_blob.shape(last_dim);
//
//  const Dtype* src_data = src_blob.cpu_data();
//  Dtype* dst_data = dst_blob.mutable_cpu_data();
//
//  for (int i = 0; i < total_channel_num; ++i) {
//    upscale_2x_cpu<PointInterpolateForward>(
//        const_cast<Dtype*>(src_data), src_h, src_w, dst_data);
//    src_data += src_spatial_dim;
//    dst_data += dst_spatial_dim;
//  }
//}
//
//template void upscale_2x_cpu(const Blob<float>& src_blob, Blob<float>& dst_blob);
//template void upscale_2x_cpu(const Blob<double>& src_blob, Blob<double>& dst_blob);
////
//
//template <typename Dtype>
//void upscale_2x_grad_cpu(const Blob<Dtype>& dst_blob, Blob<Dtype>& src_blob) {
//  std::vector<int> infered_dst_shape = src_blob.shape();
//  CHECK_GE(infered_dst_shape.size(), 2);
//  int last_dim = infered_dst_shape.size() -1;
//  infered_dst_shape[last_dim] *= 2;
//  infered_dst_shape[last_dim-1] *= 2;
//  CHECK(infered_dst_shape == dst_blob.shape());
//
//  int total_channel_num = src_blob.count(0, infered_dst_shape.size() - 2);
//  int src_spatial_dim = src_blob.count(last_dim - 1);
//  int dst_spatial_dim = dst_blob.count(last_dim - 1);
//  int src_h = src_blob.shape(last_dim - 1);
//  int src_w = src_blob.shape(last_dim);
//
//  const Dtype* dst_diff = dst_blob.cpu_diff();
//  Dtype* src_diff = src_blob.mutable_cpu_diff();
//  for (int i = 0; i < total_channel_num; ++i) {
//    upscale_2x_cpu<PointInterpolateBackward>(
//        src_diff, src_h, src_w, const_cast<Dtype*>(dst_diff));
//    src_diff += src_spatial_dim;
//    dst_diff += dst_spatial_dim;
//  }
//}
//
//template void upscale_2x_grad_cpu(const Blob<float>& dst_blob, Blob<float>& src_blob);
//template void upscale_2x_grad_cpu(const Blob<double>& dst_blob, Blob<double>& src_blob);



template <typename Dtype>
void Blob2xUpscaler<Dtype>::Backward_cpu(const Blob<Dtype>& dst_blob,
    Blob<Dtype>& src_blob) {

  Blob2xUpscaler<Dtype>::Check(src_blob, dst_blob);
  int last_dim = src_blob.shape().size() - 1;
  int total_channel_num = src_blob.count(0, src_blob.shape().size() - 2);
  int src_spatial_dim = src_blob.count(last_dim - 1);
  int dst_spatial_dim = dst_blob.count(last_dim - 1);
  int src_h = src_blob.shape(last_dim - 1);
  int src_w = src_blob.shape(last_dim);

  const Dtype* dst_diff = dst_blob.cpu_diff();
  Dtype* src_diff = src_blob.mutable_cpu_diff();
  for (int i = 0; i < total_channel_num; ++i) {
    upscale_2x_cpu<PointInterpolateBackward>(
        src_diff, src_h, src_w, const_cast<Dtype*>(dst_diff));
    src_diff += src_spatial_dim;
    dst_diff += dst_spatial_dim;
  }
}

template class Blob2xUpscaler<float>;
template class Blob2xUpscaler<double>;

}  // namespace caffe
