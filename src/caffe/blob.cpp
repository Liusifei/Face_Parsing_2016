#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_MATIO
#include "matio.h"
#endif

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike( shared_ptr<Blob<Dtype> > other) {
  Reshape(other->shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(shared_ptr< Blob<Dtype> > other) {
  vector<int> othershape = other->shape();
  if(othershape.size() != shape_.size())
    return false;
  for(int i=0;i<shape_.size();i++)
  {
    if(shape_[i]!=othershape[i])
       return false;
  }
  return true;
}


template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

//following add by liangji20040249@gmail.com
template <typename Dtype>
void Blob<Dtype>::run_statistics(){
  const Dtype * data = cpu_data();
  const Dtype * diff = cpu_diff();
  
   data_min_ = data_at(0,0,0,0);
   data_max_ = data_at(0,0,0,0);
   diff_min_ = diff_at(0,0,0,0);
   diff_max_ = diff_at(0,0,0,0);
   data_std_ = 0;
   diff_std_ = 0;
   data_var_ = 0;
   diff_var_ = 0;
   data_mean_ = 0;
   diff_mean_ = 0;
   data_nan_ = false;
   diff_nan_ = false;
  
  for(int i=0;i < count_; i++)
  {
        if(data_min_ > data[i])  data_min_ = data[i];
        
        if(data_max_ < data[i])  data_max_ = data[i];
        
        if(diff_min_ > diff[i])  diff_min_ = diff[i];
        
        if(diff_max_ < diff[i])  diff_max_ = diff[i];
        
        data_mean_ += data[i];
        diff_mean_ += diff[i];
        data_var_ += data[i]*data[i];
        diff_var_ += diff[i]*diff[i];
        
        if ( isnan( data[i] ) || isinf( data[i] ) )  data_nan_ = true;
        if ( isnan( diff[i] ) || isinf( diff[i] ) )  diff_nan_ = true;
        
  }
  data_mean_ = data_mean_ / count_;
  data_var_ = data_var_ / count_ - data_mean_ * data_mean_;
  if(data_var_ < 0)
        data_std_ = 0;
  else
        data_std_ = sqrt(data_var_);
  
  diff_mean_ = diff_mean_ / count_;
  diff_var_ = diff_var_ / count_ - diff_mean_ * diff_mean_;
  if(diff_var_ < 0)
        diff_std_ = 0;
  else
        diff_std_ = sqrt(diff_var_);
}

template <typename Dtype>
bool Blob<Dtype>::nan_diff() const{
  return diff_nan_;
}

template <typename Dtype>
bool Blob<Dtype>::nan_data() const{
  return data_nan_;
}

template <typename Dtype>
Dtype Blob<Dtype>::std_data() const{
  return data_std_;
}

template <typename Dtype>
Dtype Blob<Dtype>::std_diff() const{
  return diff_std_;
}

template <typename Dtype>
Dtype Blob<Dtype>::mean_diff() const{
  return diff_mean_;
}

template <typename Dtype>
Dtype Blob<Dtype>::mean_data() const{
  return data_mean_;
}

template <typename Dtype>
Dtype Blob<Dtype>::min_data() const{
  return data_min_;
}

template <typename Dtype>
Dtype Blob<Dtype>::min_diff() const{
   return diff_min_;
}

template <typename Dtype>
Dtype Blob<Dtype>::max_data() const{
   return data_max_;
}

template <typename Dtype>
Dtype Blob<Dtype>::max_diff() const{
   return diff_max_;
}

template <typename Dtype>
void Blob<Dtype>::check_data() const{
    int num_ = shape_[0],channels_=shape_[1],height_=shape_[2],width_=shape_[3]; 
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          Dtype dt = data_at(n, c, h, w);
          if ( isnan( dt ) || isinf( dt ) )
            LOG(FATAL) << n << "," << c << "," << h << "," << w << ": " << dt;
        }
      }
    }
  }
}

template <typename Dtype>
void Blob<Dtype>::check_diff() const{
    int num_ = shape_[0],channels_=shape_[1],height_=shape_[2],width_=shape_[3]; 
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          Dtype df = diff_at(n, c, h, w);
          if ( isnan( df ) || isinf( df ) )
            LOG(FATAL) << n << "," << c << "," << h << "," << w << ": " << df;
        }
      }
    }
  }
}

template <> void Blob<unsigned int>::AddDiffFrom(const Blob& other) { NOT_IMPLEMENTED; }
template <> void Blob<int>::AddDiffFrom(const Blob& other) { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::AddDiffFrom(const Blob& other)
{
	  switch (Caffe::mode()) {
	  case Caffe::GPU:
		caffe::caffe_gpu_axpy<Dtype>(count_,Dtype(1),other.gpu_diff(),
				static_cast<Dtype*>(diff_->mutable_gpu_data()));
	    break;
	  case Caffe::CPU:
		caffe::caffe_axpy<Dtype>(count_,Dtype(1),other.cpu_diff(),
		  				static_cast<Dtype*>(diff_->mutable_cpu_data()));
	    break;
	  default:
	    LOG(FATAL) << "Unknown caffe mode.";
	  }
}


template <> void Blob<unsigned int>::ClearDiff() { NOT_IMPLEMENTED; }
template <> void Blob<int>::ClearDiff() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::ClearDiff()
{
	  switch (Caffe::mode()) {
	  case Caffe::GPU:
	    caffe::caffe_gpu_set(count_,Dtype(0), static_cast<Dtype*>(diff_->mutable_gpu_data()));
	    break;
	  case Caffe::CPU:
		caffe::caffe_set(count_,Dtype(0), static_cast<Dtype*>(diff_->mutable_cpu_data()));
	    break;
	  default:
	    LOG(FATAL) << "Unknown caffe mode.";
	  }
}

// jay add
template <typename Dtype>
void Blob<Dtype>::WriteToBinaryFile(std::string& fn) {
  std::ofstream ofs(fn.c_str(), std::ios_base::binary | std::ios_base::out);

  if (!ofs.is_open()) {
    LOG(FATAL) << "Failt to open " << fn;
  }
  
  ofs.write((char*)&shape_[2], sizeof(shape_[2]));
  ofs.write((char*)&shape_[3], sizeof(shape_[3]));
  ofs.write((char*)&shape_[1], sizeof(shape_[1]));
  ofs.write((char*)&shape_[0], sizeof(shape_[0]));

  Dtype val;

  for (int n = 0; n < shape_[0]; ++n) {
    for (int c = 0; c < shape_[1]; ++c) {
      for (int w = 0; w < shape_[3]; ++w) {
	for (int h = 0; h < shape_[2]; ++h) {
	  val = data_at(n, c, h, w);
	  ofs.write((char*)&val, sizeof(val));
	}
      }
    }
  }


  ofs.close();
}
// end jay

#ifdef USE_MATIO
template <typename Dtype> enum matio_types matio_type_map();
template <> enum matio_types matio_type_map<float>() { return MAT_T_SINGLE; }
template <> enum matio_types matio_type_map<double>() { return MAT_T_DOUBLE; }
template <> enum matio_types matio_type_map<int>() { return MAT_T_INT32; }
template <> enum matio_types matio_type_map<unsigned int>() { return MAT_T_UINT32; }

template <typename Dtype> enum matio_classes matio_class_map();
template <> enum matio_classes matio_class_map<float>() { return MAT_C_SINGLE; }
template <> enum matio_classes matio_class_map<double>() { return MAT_C_DOUBLE; }
template <> enum matio_classes matio_class_map<int>() { return MAT_C_INT32; }
template <> enum matio_classes matio_class_map<unsigned int>() { return MAT_C_UINT32; }

template <typename Dtype>
void Blob<Dtype>::FromMat(const char *fname) {
  mat_t *matfp;
  matfp = Mat_Open(fname, MAT_ACC_RDONLY);
  CHECK(matfp) << "Error opening MAT file " << fname;
  // Read data
  matvar_t *matvar;
  matvar = Mat_VarReadInfo(matfp,"data");
  CHECK(matvar) << "Field 'data' not present in MAT file " << fname;
  {
    CHECK_EQ(matvar->class_type, matio_class_map<Dtype>())
      << "Field 'data' must be of the right class (single/double) in MAT file " << fname;
    CHECK(matvar->rank < 5) << "Field 'data' cannot have ndims > 4 in MAT file " << fname;
    Reshape((matvar->rank > 3) ? matvar->dims[3] : 1,
	    (matvar->rank > 2) ? matvar->dims[2] : 1,
	    (matvar->rank > 1) ? matvar->dims[1] : 1,
	    (matvar->rank > 0) ? matvar->dims[0] : 0);
    Dtype* data = mutable_cpu_data();
    int ret = Mat_VarReadDataLinear(matfp, matvar, data, 0, 1, count());	 
    CHECK(ret == 0) << "Error reading array 'data' from MAT file " << fname;
    Mat_VarFree(matvar);
  }
  // Read diff, if present
  matvar = Mat_VarReadInfo(matfp,"diff");
  if (matvar && matvar -> data_size > 0) {
    CHECK_EQ(matvar->class_type, matio_class_map<Dtype>())
      << "Field 'diff' must be of the right class (single/double) in MAT file " << fname;
    Dtype* diff = mutable_cpu_diff();
    int ret = Mat_VarReadDataLinear(matfp, matvar, diff, 0, 1, count());	 
    CHECK(ret == 0) << "Error reading array 'diff' from MAT file " << fname;
    Mat_VarFree(matvar);
  }
  Mat_Close(matfp);
}

template <typename Dtype>
void Blob<Dtype>::ToMat(const char *fname, bool write_diff) {
  mat_t *matfp;
  matfp = Mat_Create(fname, 0);
  //matfp = Mat_CreateVer(fname, 0, MAT_FT_MAT73);
  CHECK(matfp) << "Error creating MAT file " << fname;
  size_t dims[4];
  dims[0] = shape_[3]; dims[1] = shape_[2]; dims[2] = shape_[1]; dims[3] = shape_[0];
  matvar_t *matvar;
  // save data
  {
    matvar = Mat_VarCreate("data", matio_class_map<Dtype>(), matio_type_map<Dtype>(),
			   4, dims, mutable_cpu_data(), 0);
    CHECK(matvar) << "Error creating 'data' variable";
    CHECK_EQ(Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE), 0) 
      << "Error saving array 'data' into MAT file " << fname;
    Mat_VarFree(matvar);
  }
  // save diff
  if (write_diff) {
    matvar = Mat_VarCreate("diff", matio_class_map<Dtype>(), matio_type_map<Dtype>(),
			   4, dims, mutable_cpu_diff(), 0);
    CHECK(matvar) << "Error creating 'diff' variable";
    CHECK_EQ(Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE), 0)
      << "Error saving array 'diff' into MAT file " << fname;
    Mat_VarFree(matvar);
  }
  Mat_Close(matfp);
}

#endif



INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

