#ifndef CAFFE_UTIL_UTIL_IMG_H_
#define CAFFE_UTIL_UTIL_IMG_H_



#include <vector>

#include "caffe/proto/caffe.pb.h"

#include "caffe/blob.hpp"

#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define SQR(a) ((a)*(a))

using cv::Point_;
using cv::Mat_;
using cv::Mat;
using cv::vector;

#define PI 3.14159265

namespace caffe {




template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,
		Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,
		Blob<Dtype>* dst);



template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c);


template <typename Dtype>
void ResizeBlob(const Blob<Dtype>* src,
		Blob<Dtype>* dst);

template <typename Dtype>
void BiLinearResizeMat_cpu(const Dtype* src, const int src_h, const int src_w,
		Dtype* dst, const int dst_h, const int dst_w);

template <typename Dtype>
void RuleBiLinearResizeMat_cpu(const Dtype* src,Dtype* dst, const int dst_h, const int dst_w,
		const Dtype* loc1, const Dtype* weight1, const Dtype* loc2,const Dtype* weight2,
		const	Dtype* loc3,const Dtype* weight3,const Dtype* loc4, const Dtype* weight4);


template <typename Dtype>
void GetBiLinearResizeMatRules_cpu(  const int src_h, const int src_w,
		  const int dst_h, const int dst_w,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4);

template <typename Dtype>
void GetBiLinearResizeMatRules_gpu(  const int src_h, const int src_w,
		  const int dst_h, const int dst_w,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4);





/**
 * be careful, this function is valid only when src and dst are GPU memory
 *
 */
template <typename Dtype>
void BiLinearResizeMat_gpu(const Dtype* src, const int src_h, const int src_w,
		Dtype* dst, const int dst_h, const int dst_w);
//
//template <typename Dtype>
//void BlobAddPadding_cpu(const Blob<Dtype>* src,
//		Blob<Dtype>* dst, const int pad_h,const int pad_w);

template <typename Dtype>
void GenerateSubBlobs_cpu(const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);

template <typename Dtype>
void GenerateSubBlobs_gpu(const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);


template <typename Dtype>
void CropBlobs_cpu( const Blob<Dtype>&src, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst);

template <typename Dtype>
void CropBlobs_gpu( const Blob<Dtype>&src, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst);

/**
 * @brief  crop blob. The destination blob will be reshaped.
 */
template <typename Dtype>
void CropBlobs( const Blob<Dtype>&src, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst);



template <typename Dtype>
void CropBlobs_cpu( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h = 0, const int dst_start_w = 0);

template <typename Dtype>
void CropBlobs_gpu( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h = 0, const int dst_start_w = 0);
/**
 * @brief  crop blob. The destination blob will not be reshaped.
 */
template <typename Dtype>
void CropBlobs( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h = 0, const int dst_start_w = 0);

/**
 * src(n,c,h,w)  ===>   dst(n_ori,c,new_h,new_w)
 * n contains sub images from (0,0),(0,1),....(nh,nw)
 */
template <typename Dtype>
void ConcateSubImagesInBlobs_cpu( const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);

template <typename Dtype>
void ConcateSubImagesInBlobs_gpu( const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);


/*
 * @brief 得到从目标图像到源图像的二维旋转变换矩阵，是一个2x3的矩阵M。
 *        dst(x, y) = src(M_00 * x + M_01 * y + M_02,
 *                      M_10 * y + M_11 * y + M_12)
 *
 * @param srcCenter 源图像的旋转中心
 *        dstCenter 目标图像的旋转中心
 *        alpha 旋转角度(弧度)。角度为正值表示向逆时针旋转，坐标轴跟像素的index一致。
 *        scale 缩放系数。表示从源图像到目标图像缩小的倍数。
 *
 */
template<typename Dtype>
Mat_<Dtype> Get_Affine_matrix(const Point_<Dtype>& srcCenter,
    const Point_<Dtype>& dstCenter,
    const Dtype alpha, const Dtype scale);


/*
 * @brief 对源图像进行仿射变换
 *        dst(i, j) = src(M_00 * x + M_01 * y + M_02,
 *                      M_10 * y + M_11 * y + M_12)
 *        插值方法是最近邻
 *
 *        每个dst(i, j)都会对应与一个src(a, b)，
 *        当(a, b)不在源图像上的时候（超出源图像）：
 *        (1) fill_type为true：用value来填充dst(i, j)的值
 *        (2) fill_type为false：用源图像上距离(a, b)的点的值来填充dst(i, j)的值
 *
 */
template<typename Dtype>
void mAffineWarp(const Mat_<Dtype>& M, const Mat& srcImg, Mat& dstImg,
    const bool fill_type = true, const uchar value = 0);


void EncodeImage(const cv::Mat& img, string& result_str,
    const string format = ".jpg");

//enum AffineImage_Norm_Mode {
//  // 以左眼到左嘴角以及右眼到右嘴角的平均距离作为标准距离
//  AVE_LE2LM_RE2RM = 0,
//
//  // 以包含左右眼左右嘴角的最小矩形的对角线作为标准距离
//  RECT_LE_RE_LM_RM,
//};

/*
 * @brief 对输入图像进行affine变换(主要用于reid)
 *        landmark顺序：
 *        0: 左眼
 *        1: 右眼
 *        2: 鼻尖
 *        3: 左嘴角
 *        4: 右嘴角
 *
 *        对于输入图像，会根据norm_mode进行不同规格进行缩放，
 *        缩放的时候会存在一个标准距离，然后标准距离会占图像的norm_ratio
 *
 *        接着目标图像会以下标为center_ind的landmark为中心，
 *        center_ind为负数表示以左右眼左右嘴角的四个点的中点作为目标图像的中心
 *
 *        目标图像是两眼水平头朝上
 */
void GetAffineImage(const cv::Mat& src, cv::Mat& dst,
    const vector<float>& landmark,
    const AffineImageParameter& affine_param);


void ReadAnnotations(std::string sourcefile, vector<pair<string, vector<float> > > & samples ,int key_point_count);
void ReadAnnotations(std::string sourcefile, std::string lmkfile, vector<pair<string, vector<float> > > & samples ,int key_point_count);

float GetROCData(vector<pair<float, bool> >& pred_results,
    vector<vector<float> >& tp_fp_rates) ;

pair<float, float> GetBestAccuracy(vector<pair<float, bool> >& pred_results) ;

float GetL2Distance(const vector<float>& fea1, const vector<float>& fea2) ;


template<typename T>  inline Mat_<float> Get_Affine_Matrix(T &srcCenter, T &dstCenter,float alpha, float scale)
{
    Mat_<float> M(2,3);
    M(0,0) = scale*cos(alpha);
    M(0,1) = scale*sin(alpha);
    M(1,0) = -M(0,1);
    M(1,1) =  M(0,0);

    M(0,2) = srcCenter.x - M(0,0)*dstCenter.x - M(0,1)*dstCenter.y;
    M(1,2) = srcCenter.y - M(1,0)*dstCenter.x - M(1,1)*dstCenter.y;
    return M;
}
inline Mat_<float> inverseMatrix(Mat_<float>& M)
{
    double D = M(0,0)*M(1,1) - M(0,1)*M(1,0);
    D = D != 0 ? 1./D : 0;

    Mat_<float> inv_M(2,3);

    inv_M(0,0) = M(1,1)*D;
    inv_M(0,1) = M(0,1)*(-D);
    inv_M(1,0) = M(1,0)*(-D);
    inv_M(1,1) = M(0,0)*D;

    inv_M(0,2) = -inv_M(0,0)*M(0,2) - inv_M(0,1)*M(1,2);
    inv_M(1,2) = -inv_M(1,0)*M(0,2) - inv_M(1,1)*M(1,2);
    return inv_M;
}
void mAffineWarp(const Mat_<float> M, const Mat& srcImg,Mat& dstImg,int interpolation=INTER_LINEAR);
template<typename T>  inline void Affine_Point(const Mat_<float> &M,T& srcPt, T &dstPt)
{
    float x = M(0,0)*srcPt.x + M(0,1)*srcPt.y + M(0,2);
    float y = M(1,0)*srcPt.x + M(1,1)*srcPt.y + M(1,2);
    dstPt.x = x;
    dstPt.y = y;
}
template<typename T>  inline void Affine_Shape(const Mat_<float>& M, const Mat_<T>& srcShape, Mat_<T>& dstShape)
{
    int N = srcShape.rows/2;
    if (dstShape.rows != srcShape.rows || dstShape.cols != srcShape.cols)
        dstShape.create(srcShape.rows, srcShape.cols);
    for (int i = 0; i<N; ++i)
    {
        Point_<float> pt(srcShape(i,0), srcShape(i+N,0));
        Affine_Point(M, pt, pt);
        dstShape(i,0) = (float)pt.x;
        dstShape(i+N,0) = (float)pt.y;
    }
}
//grid based warp disturbance
struct Triangle
{
    Triangle(){}
    Triangle(int first,int second,int third)
    {
        i = first;
        j = second;
        k = third;
    }
    int i,j,k;      //Fist, Second and Third  Vertex Indexs 
};

inline void TriWarp(float x, float y,  float x1, float y1, float x2, float y2, float x3, float y3, float& X, float& Y,  float X1, float Y1, float X2, float Y2, float X3, float Y3)
{
    float c = 1.0/(+x2*y3-x2*y1-x1*y3-x3*y2+x3*y1+x1*y2);
    float alpha = (y*x3-y3*x+x*y2-x2*y+x2*y3-x3*y2)*c;
    float belta  = (-y*x3+x1*y+x3*y1+y3*x-x1*y3-x*y1)*c;
    float gamma = 1.0 - alpha - belta; 

    X = alpha*X1 + belta*X2 + gamma*X3;
    Y = alpha*Y1 + belta*Y2 + gamma*Y3;
}

void disturb_Triangle(Mat& img,Mat& dstImg, Mat& labelImg, Mat& dstlabelImg,int blockx, int blocky, float distrub_radius_ratio=0.3,  int interpolation=CV_INTER_LINEAR);
template<typename T>
void getBoundingBox(vector<T>& shape, T& TL, T& BR)
{
    float min_x = 10e10;
    float min_y = 10e10;
    float max_x = 0;
    float max_y = 0;
    for (int i = 0; i < shape.size(); ++i)
    {
        if (shape[i].x < min_x)
            min_x = shape[i].x;
        if (shape[i].y < min_y)
            min_y = shape[i].y;
        if (shape[i].x > max_x)
            max_x = shape[i].x;
        if (shape[i].y > max_y)
            max_y = shape[i].y;
    }
    TL.x = min_x - 1;
    TL.y = min_y - 1;
    BR.x = max_x + 1;
    BR.y = max_y + 1;
}


}  // namespace caffe




#endif   // CAFFE_UTIL_UTIL_IMG_H_
