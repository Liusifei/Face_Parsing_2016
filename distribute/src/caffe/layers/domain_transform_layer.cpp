/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/domaintransform_layer.hpp"
#include "caffe/util/util_img.hpp"
#include "opencv2/opencv.hpp"

namespace caffe {



template <typename Dtype>
void DomainTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
	CHECK(bottom.size()==2)<<"DomainTransformLayer need two bottom.";
  
}

template <typename Dtype>
void DomainTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
	CHECK(bottom[0]->num() == bottom[1]->num())<<"DomainTransformLayer need two qeual bottom.";
	CHECK(bottom[0]->channels() == bottom[1]->channels())<<"DomainTransformLayer need two qeual bottom.";
	CHECK(bottom[0]->height() == bottom[1]->height())<<"DomainTransformLayer need two qeual bottom.";
	CHECK(bottom[0]->width() == bottom[1]->width())<<"DomainTransformLayer need two qeual bottom.";
	top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
}

template <typename Dtype>
void DomainTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	int n = bottom[0]->num();
	int c = bottom[0]->channels();
	int h = bottom[0]->height();
	int w = bottom[0]->width();
	const int count = h*w*c;	
	cv::Mat I = cv::Mat::eye(c,c,CV_32FC1);
	int dim = h*w;	

	Dtype * bottom1_data = bottom[0]->mutable_cpu_data();
	Dtype * bottom2_data = bottom[1]->mutable_cpu_data();
	Dtype * top_data = top[0]->mutable_cpu_data();
	COVs_inv_.clear();
	A_.clear();
	
	//loop for each sample
	for(int i=0;i<n;i++)
	{
		Dtype * b1 = bottom1_data + i * count;
		Dtype * b2 = bottom2_data + i * count;
		Dtype * topdata = top_data + i * count;
		
		//set Fs, Ft, Fs_out
		cv::Mat Fs(c, dim, CV_32FC1, b1);
		cv::Mat Ft(c, dim, CV_32FC1, b2);
		cv::Mat Fs_out(c, dim, CV_32FC1,topdata);
		
		//get inv(COVs)
		cv::Mat COVs(c, c, CV_32FC1);
		cv::Mat MEANs(1, dim, CV_32FC1);
		cv::calcCovarMatrix(Fs,COVs,MEANs,CV_COVAR_NORMAL+CV_COVAR_COLS,CV_32F);
		COVs = COVs * (1.0/(dim-1));
		COVs = COVs + I;
		cv::Mat COVs_inv = COVs.inv();
		COVs_inv_.push_back(COVs_inv);

		//get inv(sqrt(COVs))
		cv::Mat us,ss,vs;
		cv::SVD::compute(COVs,ss,us,vs);
		cv::Mat sis = cv::Mat::eye(c,c,CV_32FC1);
		for(int k=0;k<c;k++)
			sis.at<float>(k,k) = ss.at<float>(k);
		cv::Mat s_sqrt;
		cv::sqrt(sis, s_sqrt);		
		cv::Mat COVs_sqrt = us * s_sqrt * vs ;
		cv::Mat COVs_sqrt_inv = COVs_sqrt.inv() ;

		//get COVt and sqrt(COVt)
		cv::Mat COVt(c, c, CV_32FC1);
		cv::Mat MEANt(1, dim, CV_32FC1);
		cv::calcCovarMatrix(Ft,COVt,MEANt,CV_COVAR_NORMAL+CV_COVAR_COLS,CV_32F);
		COVt = COVt * (1.0/(dim-1));
		COVt = COVt + I;
		
		cv::Mat ut,st,vt;
		cv::SVD::compute(COVt,st,ut,vt);
		cv::Mat sit = cv::Mat::eye(c,c,CV_32FC1);
		for(int k=0;k<c;k++)
			sit.at<float>(k,k) = st.at<float>(k);
		cv::Mat t_sqrt;
		cv::sqrt(sit, t_sqrt);		
		cv::Mat COVt_sqrt = ut * t_sqrt * vt;
		
		//get A = inv(sqrt(COVs)) * sqrt(COVt)
		cv::Mat A = COVs_sqrt_inv * COVt_sqrt;
		A_.push_back(A);

		//Fs_out = ((Fs - mean(Fs))' * A )' + mean(Ft)
		cv::Mat Fs_submean = Fs;
		for(int k=0;k<c;k++)
			Fs_submean.row(k) -= MEANs.at<float>(k);
		
		cv::Mat temp = Fs_submean.t() * A;
		Fs_out = temp.t();
		for(int k=0;k<c;k++)
			Fs_out.row(k) += MEANt.at<float>(k);
	}
	
}

template <typename Dtype>
void DomainTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	//clear bottom diff
	caffe_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_cpu_diff());
	
	Dtype * top_diff = top[0]->mutable_cpu_diff();
	Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
	
	int n = bottom[0]->num();
	int c = bottom[0]->channels();
	int h = bottom[0]->height();
	int w = bottom[0]->width();
	const int count = h*w*c;	
	cv::Mat I = cv::Mat::eye(c,c,CV_32FC1);
	int dim = h*w;	
	
	//loop for each sample
	for(int i=0;i<n;i++)
	{	
		Dtype * b_diff = bottom_diff + i * count;
		Dtype * t_diff = top_diff + i * count;
		
		cv::Mat B_diff(c, dim, CV_32FC1, b_diff);
		cv::Mat T_diff(c, dim, CV_32FC1, t_diff);

		//delta = 0.5*(eye + inv(COVs))*A
		cv::Mat delta = I + COVs_inv_[i];
		delta = delta * A_[i];
		delta = delta * 0.5;
		
		//bottom_diff = delta * top_diff
		B_diff = delta * T_diff; 	
	}
}

#ifdef CPU_ONLY
STUB_GPU(DomainTransformLayer);
#endif

INSTANTIATE_CLASS(DomainTransformLayer);
REGISTER_LAYER_CLASS(DomainTransform);

}  // namespace caffe
