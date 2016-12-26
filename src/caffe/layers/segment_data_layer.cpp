/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  
#include <iostream> 
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/segment_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/util_img.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
SegmentDataLayer<Dtype>::~SegmentDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void SegmentDataLayer<Dtype>::ImageDataToBlob(shared_ptr<Blob<Dtype> > top, const int item_id, cv::Mat image, vector<float> mean, float datascale)
{
	CHECK(top->height() == image.rows);
	CHECK(top->width() == image.cols);
	CHECK(top->channels() == image.channels());
	
	Dtype * topdata = top->mutable_cpu_data();
	int idcount=0;
	const int spsize = image.rows* image.cols *image.channels();
	const int channel = image.channels();
        
        CHECK(mean.size() == channel);

	if(channel ==3)
	{
		for (int c = 0; c < channel; ++c) 
		{                            
			for (int h = 0; h < image.rows; ++h) {
			    for (int w = 0; w < image.cols; ++w) {

				float v =static_cast<Dtype>(static_cast<unsigned char>(image.at<cv::Vec3b>(h, w)[c]));
				topdata[idcount + item_id * spsize] = (v - mean[c])*datascale;
				idcount++;
			
			    }
			}
		}
	}
	else
	{
		for (int h = 0; h < image.rows; ++h) {
		    for (int w = 0; w < image.cols; ++w) {

			float v =static_cast<Dtype>(static_cast<unsigned char>(image.at<unsigned char>(h, w)));
			topdata[idcount + item_id * spsize] = (v - mean[0])*datascale;
      //LOG(INFO)<<"v:"<<v<<", top:"<<topdata[idcount + item_id * spsize]<<", mean:"<<mean[0]<<", scale:"<<datascale;
			idcount++;
		    }
		}
	}
}


template <typename Dtype>
void SegmentDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  param_ = this->layer_param_.segment_data_param();
  top_size_ = param_.topinfo_size();
  const int top_size = top_size_;
  CHECK(top_size == top.size());
  batch_size_ = param_.batch_size();
  const int batch_size = batch_size_;
  
  //use_flip_ = param_.use_flip();
  use_distrib_ = param_.use_distrib();
  //distrib_scale_ = param_.distrib_scale();
  
  
  for(int i=0;i<this->PREFETCH_COUNT;i++)
  {
     this->prefetch_[i].blobs_.clear();
  }
  for (int top_id = 0; top_id < top_size; ++top_id) {
    

    for(int i=0;i<this->PREFETCH_COUNT;i++)
    {
       this->prefetch_[i].blobs_.push_back(shared_ptr<Blob<Dtype> >());
    }

    const TopInfo& top_info = param_.topinfo(top_id);
    const int channel = top_info.channels();
    const int width = top_info.width();
    const int height = top_info.height();
    

    for(int i=0;i<this->PREFETCH_COUNT;i++)
    {
       this->prefetch_[i].blobs_[top_id].reset( new Blob<Dtype>(batch_size, channel,height, width));
    }

    
    top[top_id]->Reshape(this->prefetch_[0].blobs_[top_id].get()->shape());

    LOG(INFO) << "output data #" << top_id
        << " size: " << top[top_id]->num() << ","
        << top[top_id]->channels() << "," << top[top_id]->height() << ","
        << top[top_id]->width();
  }
  string source = param_.source();
  data_folder_ = param_.data_folder();
  LOG(INFO)<<"opening file: "<<source;
  ifstream infile(source.c_str());
  string line;
  lines_.clear();
  while(infile >> line)
  {
    lines_.push_back(line);
  }
  infile.close();
  LOG(INFO)<<"A total of "<<lines_.size()<<" samples.";
  lines_id_ = 0;
  if(param_.shuffle())
  {
    const unsigned int seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(seed));
    ShuffleImages();
  }
}

template <typename Dtype>
void SegmentDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}
template <typename Dtype>
void SegmentDataLayer<Dtype>::GetDistribParam() {
  
  if(!use_distrib_)
  {
    distrib_angle_ = 0.0f;
    distrib_scale_ = 1.0f; 
    distrib_x_ = 0.0f;
    distrib_y_ = 0.0f;
    use_flip_ = false;
  }
  else
  {
    float range;
    range = ((caffe::caffe_rng_rand()%INT_MAX)*1.0f /(INT_MAX -1) -0.5);//range (-0.5,0.5)
    distrib_angle_ = param_.distrib_angle() * range; // rotation -0.5* angle ~ 0.5*angle
    range = ((caffe::caffe_rng_rand()%INT_MAX)*1.0f /(INT_MAX -1) -0.5);
    distrib_scale_ = 1+ param_.distrib_scale() * range; //scale 1- 0.5*scale ~ 1+ 0.5*scale
    range = ((caffe::caffe_rng_rand()%INT_MAX)*1.0f /(INT_MAX -1) -0.5);
    distrib_x_ = param_.distrib_x() * range;
    range = ((caffe::caffe_rng_rand()%INT_MAX)*1.0f /(INT_MAX -1) -0.5);
    distrib_y_ = param_.distrib_y() * range;
    if(param_.use_flip())
    {
      range = ((caffe::caffe_rng_rand()%INT_MAX)*1.0f /(INT_MAX -1) -0.5);
      use_flip_ = range>0?true:false;
    }
  }
}
template <typename Dtype>
void SegmentDataLayer<Dtype>::GetHeatmapTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo)
{
;
}

template <typename Dtype>
void SegmentDataLayer<Dtype>::GetLaneTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo)
{ 
  CHECK(topinfo.channels() == 4); // mask,  a,b,c
  CHECK(!use_flip_);// do not support flip yet
  CHECK(fabs(distrib_angle_)<1e-10);// do not support rotation distrib
  CHECK(top->channels()==topinfo.channels());
  CHECK(top->width()==topinfo.width());
  CHECK(top->height()==topinfo.height());

  const int perpixel_labelnum = topinfo.channels() + 2; // x +y + channels
  std::ifstream inputfile(src.c_str());
  if(!inputfile.is_open())
    LOG(FATAL)<<"can not open text label file "<<src;
  Dtype v;
  std::vector<Dtype> pts;
  float src_height,src_width,src_channels,src_num;

  //temp set value for lane detect
  src_num = 1;
  src_channels = topinfo.channels();
  src_height = topinfo.height();
  src_width = topinfo.width();
  
/*
  inputfile>>src_num;
  inputfile>>src_channels;
  inputfile>>src_height;
  inputfile>>src_width;
 */
  while(inputfile >> v) 
    pts.push_back(v);
  CHECK(pts.size()%perpixel_labelnum ==0);
  
  const int dim = top->count() / top->num();
  const int spdim = top->height() * top->width();
  const int width = top->width();
  const int height = top->height();

  cv::Mat P;
  cv::Size sz(width, height);
  
    cv::Point_<float> srcCenter(src_width/2,src_height/2);
    cv::Point_<float> dstCenter(sz.width/2+distrib_x_*sz.width, sz.height/2+distrib_y_*sz.height);    
    float scale_local =  float(sz.width) / float(src_width)*distrib_scale_;
    float alpha = distrib_angle_*CV_PI/180;      
    P = Get_Affine_matrix(dstCenter,srcCenter,alpha,scale_local);
 

  Dtype * topdata = top->mutable_cpu_data() + item_id * dim;
  caffe_set(dim,Dtype(0.0),topdata); 
  
  cv::Mat_<float> Plus = cv::Mat::eye(3,3,CV_32FC1);
  Plus(0,0) = P.at<float>(0,0);
  Plus(0,1) = P.at<float>(0,1);
  Plus(0,2) = P.at<float>(0,2);
  Plus(1,0) = P.at<float>(1,0);
  Plus(1,1) = P.at<float>(1,1);
  Plus(1,2) = P.at<float>(1,2);



  float x,y;
  cv::Point2f pp;
  float a,b,c,A,B,C,maskv,transA,transB,transC,transa,transb,transc;
  float dx,dy;
  float transx,transy;
  for(int i=0;i<pts.size();i+=perpixel_labelnum)
  {   
    if(!P.data)
    {
      LOG(FATAL)<<"get trans M before distrib points";
    }
    y = pts[i];
    x = pts[i+1];

    pp.x = float(x);
    pp.y = float(y);
    Affine_Point(P,pp,pp);
    transx = pp.x;
    transy = pp.y;
    

    maskv = pts[i+2];
    a = pts[i+3];
    b = pts[i+4];
    c = pts[i+5];
    dx = x;
    dy = y;
    A = a;
    B = b - 2*A*dy;
    C = c-A*dy*dy-B*dy+dx;
    //std::cout<<"Plus:"<<Plus<<std::endl;
    cv::Mat_<float> M = Plus.inv();
    CHECK(M(0,1)==0 && M(1,0)==0 && M(0,0) > 0);
    transA = A*M(1,1)/M(0,0);
    transB = (2*A*M(1,1)*M(1,2) + B * M(1,1))/ M(0,0);
    transC = (A*M(1,2)*M(1,2) + B*M(1,2) + C - M(0,2))/M(0,0);

    transa = transA;
    transb = 2*transA*transy + transB;
    transc = transA * transy*transy + B*transy + transC - transx;

    x = int(pp.x);
    y = int(pp.y);
   
    if(!(y>=0 && y <= top->height()-1 && x>=0 && x <= top->width()-1))
    {
      //LOG(INFO)<<"transformed sparse point out of range. x="<<x<<",y="<<y;
      continue;
    }
    
    
    topdata[0*spdim + int(y) *width + int(x)] = maskv; 
    topdata[1*spdim + int(y) *width + int(x)] = transa; 
    topdata[2*spdim + int(y) *width + int(x)] = transb; 
    topdata[3*spdim + int(y) *width + int(x)] = transc;   

  }  

}

template <typename Dtype>
void SegmentDataLayer<Dtype>::GetDensePointTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo)
{ 
  CHECK(top->channels()==topinfo.channels());
  CHECK(top->width()==topinfo.width());
  CHECK(top->height()==topinfo.height());

  const int perpixel_labelnum = topinfo.channels() + 2; // x +y + channels
  std::ifstream inputfile(src.c_str());
  if(!inputfile.is_open())
    LOG(FATAL)<<"can not open text label file "<<src;
  Dtype v;
  std::vector<Dtype> pts;
  float src_height,src_width,src_channels,src_num;

  inputfile>>src_num;
  inputfile>>src_channels;
  inputfile>>src_height;
  inputfile>>src_width;
  //LOG(INFO)<<"num:"<<src_num<<",src c:"<<src_channels<<",srcheight:"<<src_height<<",width:"<<src_width;

  while(inputfile >> v) 
    pts.push_back(v);
  CHECK(pts.size()%perpixel_labelnum ==0);
  
  const int dim = top->count() / top->num();
  const int spdim = top->height() * top->width();
  const int width = top->width();
  const int height = top->height();

  cv::Mat M;
  cv::Size sz(width, height);
  
    cv::Point_<float> srcCenter(src_width/2,src_height/2);
    cv::Point_<float> dstCenter(sz.width/2+distrib_x_*sz.width, sz.height/2+distrib_y_*sz.height);    
    float scale_local =  float(sz.width) / float(src_width)*distrib_scale_;
    float alpha = distrib_angle_*CV_PI/180;      
    M = Get_Affine_matrix(dstCenter,srcCenter,alpha,scale_local);
 

  Dtype * topdata = top->mutable_cpu_data() + item_id * dim;
  caffe_set(dim,Dtype(0.0),topdata); 
  
  float x,y;
  cv::Point2f p;
  for(int i=0;i<pts.size();i+=perpixel_labelnum)
  {   
    y = pts[i];
    x = pts[i+1];
   
    if(!M.data)
    {
      LOG(FATAL)<<"get trans M before distrib points";
    }
    p.x = float(x);
    p.y = float(y);
    Affine_Point(M,p,p);
    x = int(p.x);
    y = int(p.y);
    if(use_flip_)
    {
      x = width - 1 -x;
    }
    if(!(y>=0 && y <= top->height()-1 && x>=0 && x <= top->width()-1))
    {
      LOG(INFO)<<"transformed sparse point out of range. x="<<x<<",y="<<y;
      continue;
    }
    
    for(int c=0;c<topinfo.channels();c++)
    {
      topdata[int(c)*spdim + int(y) *width + int(x)] = pts[i+2+c];
    }
  }  

}
template <typename Dtype>
void SegmentDataLayer<Dtype>::GetSparsePointTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo)
{
  CHECK(top->channels()==topinfo.channels());
  CHECK(top->width()==topinfo.width());
  CHECK(top->height()==topinfo.height());

  const int perpixel_labelnum = 4; // y,x,c,v
  std::ifstream inputfile(src.c_str());
  if(!inputfile.is_open())
    LOG(FATAL)<<"can not open text label file "<<src;
  Dtype v;
  std::vector<Dtype> pts;
  float src_height,src_width,src_channels,src_num;

  inputfile>>src_num;
  inputfile>>src_channels;
  inputfile>>src_height;
  inputfile>>src_width;
  //LOG(INFO)<<"num:"<<src_num<<",src c:"<<src_channels<<",srcheight:"<<src_height<<",width:"<<src_width;

  while(inputfile >> v) 
    pts.push_back(v);
  CHECK(pts.size()%perpixel_labelnum ==0);
  
  const int dim = top->count() / top->num();
  const int spdim = top->height() * top->width();
  const int width = top->width();
  const int height = top->height();

  cv::Mat M;
  cv::Size sz(width, height);

    cv::Point_<float> srcCenter(src_width/2,src_height/2);
    cv::Point_<float> dstCenter(sz.width/2+distrib_x_*sz.width, sz.height/2+distrib_y_*sz.height);    
    float scale_local =  float(sz.width) / float(src_width)*distrib_scale_;
    float alpha = distrib_angle_*CV_PI/180;      
    M = Get_Affine_matrix(dstCenter,srcCenter,alpha,scale_local);

  Dtype * topdata = top->mutable_cpu_data() + item_id * dim;
  caffe_set(dim,Dtype(0.0),topdata); 
  
  float x,y;
  cv::Point2f p;
  for(int i=0;i<pts.size();i+=perpixel_labelnum)
  {   
    y = pts[i];
    x = pts[i+1];
   
    
    float c = pts[i+2];
    CHECK(c>=0 && c <= top->channels()-1)<<"channel out of range "<<c;
   
      if(!M.data)
      {
        LOG(FATAL)<<"get trans M before distrib points";
      }
      p.x = float(x);
      p.y = float(y);
      Affine_Point(M,p,p);
      x = int(p.x);
      y = int(p.y);
      if(use_flip_)
      {
        x = width - 1 -x;
      }
      if(!(y>=0 && y <= top->height()-1 && x>=0 && x <= top->width()-1))
      {
        LOG(INFO)<<"transformed sparse point out of range. x="<<x<<",y="<<y;
        continue;
      }
    topdata[int(c)*spdim + int(y) *width + int(x)] = pts[i+3];
  }
 
}

template <typename Dtype>
void SegmentDataLayer<Dtype>::GetNUMTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo)
{
  float v = str2num(src);
  int spsize = top->channels() * top->width() * top->height();
  Dtype * topdata = top->mutable_cpu_data() + item_id * spsize;
  caffe_set(spsize,Dtype(v),topdata);
}

template <typename Dtype>
void SegmentDataLayer<Dtype>::GetImageTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo)
{
  int channels = topinfo.channels();
  CHECK(channels == 1 || channels ==3);
  int cv_read_flag = channels>1?CV_LOAD_IMAGE_COLOR:CV_LOAD_IMAGE_GRAYSCALE;
  int width = topinfo.width();
  int height = topinfo.height();
  TopInfo_ResizeType resize_type = topinfo.resize_type();
  int resizetype ;
  if(resize_type == TopInfo_ResizeType_LINEAR)
    resizetype = INTER_LINEAR;
  else
    resizetype=INTER_NEAREST;

  string imname = src;
  if(data_folder_.size()>0)
    imname = data_folder_+src;

  cv::Mat cv_img_origin = cv::imread(imname,cv_read_flag);
  if(!cv_img_origin.data)
  {
    LOG(FATAL)<<"can not open image: "<<imname;
  }

  cv::Mat cv_img;
  cv::Size sz(width, height);
  cv_img.create(sz,cv_img_origin.type());
        
  cv::Point_<float> srcCenter(cv_img_origin.cols/2,cv_img_origin.rows/2);
  cv::Point_<float> dstCenter(sz.width/2+distrib_x_*sz.width, sz.height/2+distrib_y_*sz.height); //transfer
        
  float scale_local =  float(sz.width) / float(cv_img_origin.cols)*distrib_scale_;//scale
        
  float alpha = distrib_angle_*CV_PI/180; //rotation in radian
       
  if(abs(alpha - 0) < FLT_EPSILON && abs(scale_local - 1) < FLT_EPSILON && abs(srcCenter.x - dstCenter.x) < FLT_EPSILON && abs(srcCenter.y - dstCenter.y) < FLT_EPSILON)
            cv_img = cv_img_origin;
  else
  {   cv::Mat_<float> M = Get_Affine_matrix(dstCenter,srcCenter,alpha,scale_local);
      Mat_<float> inv_M = inverseMatrix(M);
      mAffineWarp(inv_M, cv_img_origin, cv_img, resizetype);
  } 
  
 if(cv_img.rows!=height || cv_img.cols !=width)                                
  {                                                                             
    cv::resize(cv_img,cv_img,cv::Size(width,height),0,0,resizetype);            
  }       
  
  if(use_flip_)
  {
      cv::flip(cv_img,cv_img,1);         
  }


  //gamma distrib
  if(topinfo.gamma_distrib_alpha()>0)
  {
    cv::Mat tempimg = cv_img.clone();
    double alpha = 1 + ((caffe::caffe_rng_rand()%INT_MAX)*1.0f /(INT_MAX -1) -0.5) * topinfo.gamma_distrib_alpha();
    double gamma = ((caffe::caffe_rng_rand()%INT_MAX)*1.0f /(INT_MAX -1)) * topinfo.gamma_distrib_gamma();
    addWeighted(tempimg, alpha, tempimg, 0, gamma, cv_img);
  }

  vector<float> mean;
  mean.clear();
  CHECK(topinfo.mean_size()>=1);
  if(topinfo.mean_size() ==1 )
  {
    for(int i=0;i<channels;i++)
    {
      mean.push_back(topinfo.mean(0));
    }
  }
  else
  {
    for(int i=0;i<topinfo.mean_size();i++)
    {
      mean.push_back(topinfo.mean(i));
    }
  }
  //LOG(INFO)<<"mean:"<<mean[0];  
  float datascale = topinfo.data_scale();

  ImageDataToBlob(top, item_id,  cv_img,mean,datascale);
}

template <typename Dtype>
void SegmentDataLayer<Dtype>::load_batch(LiangjiBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  const int batch_size = batch->blobs_[0]->num();
  const int lines_size = lines_.size();
  
  string spflag = param_.source_splitflag();

  CHECK(batch_size == batch_size_);
 
  for(int item_id = 0 ; item_id < batch_size; item_id++) //loop for each sample
  {
    vector<string> toplines = str_split(lines_[lines_id_],spflag);
    CHECK(toplines.size()== top_size_);

    
    GetDistribParam();

    for(int top_id=0;top_id<top_size_;top_id++) //loop for each top blob
    {
      const TopInfo top_info = param_.topinfo(top_id);
      
      TopInfo_SrcType srctype = top_info.src_type();

      if(srctype == TopInfo_SrcType_IMAGE)
      {
        GetImageTop(toplines[top_id],batch->blobs_[top_id],item_id,top_info);
      }
      else if(srctype == TopInfo_SrcType_NUMBER)
      {
        GetNUMTop(toplines[top_id],batch->blobs_[top_id],item_id,top_info);
      }
      else if(srctype == TopInfo_SrcType_SPARSEPOINT)
      {
        GetSparsePointTop(toplines[top_id],batch->blobs_[top_id],item_id,top_info);
      }
      else if(srctype == TopInfo_SrcType_DENSEPOINT)
      {
        GetDensePointTop(toplines[top_id],batch->blobs_[top_id],item_id,top_info);
      }
      else if(srctype == TopInfo_SrcType_HEATMAP)
      {
        GetHeatmapTop(toplines[top_id],batch->blobs_[top_id],item_id,top_info);
      }
      else if(srctype == TopInfo_SrcType_LANE)
      {
        GetLaneTop(toplines[top_id],batch->blobs_[top_id],item_id,top_info);
      }
      else
        LOG(FATAL)<<"unknown src type: "<<srctype;
    }

    lines_id_++;
    if (lines_id_ >= lines_size) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (param_.shuffle()) {
        ShuffleImages();
      }
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(SegmentDataLayer);
REGISTER_LAYER_CLASS(SegmentData);

}  // namespace caffe
#endif  // USE_OPENCV
