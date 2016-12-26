#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/util_img.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV

//following add by liangji20040249@gmail.com
std::vector<std::string> str_split(std::string str,std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str+=pattern;
    int size=str.size();

    for(int i=0; i<size; i++)
    {
        pos=str.find(pattern,i);
        if(pos<size)
        {
            std::string s=str.substr(i,pos-i);
            result.push_back(s);
            i=pos+pattern.size()-1;
        }
    }
    return result;
}
bool Read_Images_ToDatum(const string& filenames,vector< int> channelNum,
            vector< int> height, vector < int> width, 
            vector< Datum > &datum,float angle,float scale,
            bool is_grid_perturb,float move_x,float move_y, 
            bool flipimg,vector<int> resizetype)
{
    std::vector< cv::Point2f  > points_;
    points_.clear();
    return Read_Images_ToDatum(filenames,channelNum,height,width,datum,
                        angle,scale,is_grid_perturb,move_x,move_y,
                        flipimg,resizetype,points_);
                        
}
            
bool Read_Images_ToDatum(const string& filenames,vector< int> channelNum,
            vector< int> height, vector < int> width, vector< Datum > &datum,float angle,float scale,
            bool is_grid_perturb,float move_x,float move_y, 
            bool flipimg,vector<int> resizetype,std::vector< cv::Point2f  > & points_)//add by liangji, 20150226
{
    vector<std::string> imagenames = str_split(filenames,"||");
    CHECK(imagenames.size() == channelNum.size())<<"input data size error! image name num "<<imagenames.size()<<", channelnum size "<<channelNum.size()<<", filenames name = "<<filenames;
    for(int data_id = 0;data_id<imagenames.size();data_id++)
    {
        cv::Mat cv_img;
        int cv_read_flag = (channelNum[data_id]>1 ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagenames[data_id], cv_read_flag);
        
        if (!cv_img_origin.data) {
            LOG(ERROR) << "Could not open or find file " << imagenames[data_id];
            return false;
        }

       
        
        cv::Size sz(width[data_id], height[data_id]);
        cv_img.create(sz,cv_img_origin.type());
        
        cv::Point_<float> srcCenter(cv_img_origin.cols/2,cv_img_origin.rows/2);
        cv::Point_<float> dstCenter(sz.width/2+move_x*sz.width, sz.height/2+move_y*sz.height); //transfer
        
        float scale_local =  float(sz.width) / float(cv_img_origin.cols)*scale;//scale
        
        float alpha = angle*CV_PI/180; //rotation in radian
        //LOG(INFO)<<"srcCenter.x "<<srcCenter.x<<",srcCenter.y "<<srcCenter.y
        //         <<",dstCenter.x "<<dstCenter.x<<",dstCenter.y"<<dstCenter.y
        //         <<",alpha "<<alpha<<",scale_local "<<scale_local;
        cv::Mat_<float> M = Get_Affine_Matrix(dstCenter,srcCenter,alpha,scale_local);
        if(abs(alpha - 0) < FLT_EPSILON && abs(scale_local - 1) < FLT_EPSILON && abs(srcCenter.x - dstCenter.x) < FLT_EPSILON && abs(srcCenter.y - dstCenter.y) < FLT_EPSILON)
            cv_img = cv_img_origin;
        else
        {   
            Mat_<float> inv_M = inverseMatrix(M);
            mAffineWarp(inv_M, cv_img_origin, cv_img, resizetype[data_id] ? INTER_NEAREST : INTER_LINEAR);
        } 
    
        //cv::imwrite("src.jpg",cv_img_origin);
        //cv::imwrite("dst.jpg",cv_img);
         
        if(points_.size()>0 && data_id == 0)
        {
            for (int i = 0; i < points_.size(); ++i) 
            {    
               //  LOG(INFO)<<"before affine points "<< points_[i].x<<","<< points_[i].y;
                 Affine_Point(M, points_[i], points_[i]); //support inplace operation
	       //  LOG(INFO)<<"after affine points "<< points_[i].x<<","<< points_[i].y;
            }
        }
       // CHECK(false);
        if(flipimg)
        {
            cv::flip(cv_img,cv_img,1);
            if(points_.size()>0 && data_id == 0)
            {
                for (int i = 0; i < points_.size(); ++i) 
                {
                    points_[i].x = width[data_id] -1 - points_[i].x;
                }
            }
        }
        if(is_grid_perturb)
        {
            ; //not implemented yet
        }

        //int num_channels = (is_color ? 3 : 1);
        datum[data_id].set_channels(channelNum[data_id]);
        datum[data_id].set_height(cv_img.rows);
        datum[data_id].set_width(cv_img.cols);
        // do not set label, liangji, 20141126
        //datum->set_label(label);

        datum[data_id].clear_data();
        datum[data_id].clear_float_data();
        string* datum_string = datum[data_id].mutable_data();
        if (channelNum[data_id] == 3) 
        {
            for (int c = 0; c < channelNum[data_id]; ++c) 
            {                            
                for (int h = 0; h < cv_img.rows; ++h) {
                    for (int w = 0; w < cv_img.cols; ++w) {
                        datum_string->push_back(
                                static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
                    }
                }
            }
        } 
        else if (channelNum[data_id]==1 )
        {  // Faster than repeatedly testing is_color for each pixel w/i loop
            for (int h = 0; h < cv_img.rows; ++h) {
                for (int w = 0; w < cv_img.cols; ++w) {
                    datum_string->push_back(static_cast<char>(cv_img.at<uchar>(h, w)));
                }
            }
        }
        else{
             LOG(FATAL) <<"illegal data channel number: "<<channelNum[data_id];
        }
    }
    return true;
}
template <typename Dtype>
void SaveBlobDataWithoutOverwrite(string savename, Blob<Dtype>& blob)
{ 
    std::ifstream fin(savename.c_str());
    if (!fin)
    {
       std::ofstream fout(savename.c_str());
       if(!fout.is_open())
       {
            LOG(INFO)<<"can not create file "<<savename;
       }
       else
       {
            const int count = blob.count();
            const Dtype * data = blob.cpu_data();
            for(int i=0;i< count;i++)
            {
                fout<<data[i]<<"\n";
            }
            fout.close();
       }
    }
    else
    {
        LOG(INFO)<<savename<<" already exists. Do not over write.";
    }
}

template void SaveBlobDataWithoutOverwrite<float>(string savename, Blob<float>& blob);
template void SaveBlobDataWithoutOverwrite<double>(string savename, Blob<double>& blob);


template <typename Dtype>
void SaveBlobDiffWithoutOverwrite(string savename, Blob<Dtype>& blob)
{ 
    std::ifstream fin(savename.c_str());
    if (!fin)
    {
       std::ofstream fout(savename.c_str());
       if(!fout.is_open())
       {
            LOG(INFO)<<"can not create file "<<savename;
       }
       else
       {
            const int count = blob.count();
            const Dtype * diff = blob.cpu_diff();
            for(int i=0;i< count;i++)
            {
                fout<<diff[i]<<"\n";
            }
            fout.close();
       }
    }
    else
    {
        LOG(INFO)<<savename<<" already exists. Do not over write.";
    }
}
template void SaveBlobDiffWithoutOverwrite<float>(string savename, Blob<float>& blob);
template void SaveBlobDiffWithoutOverwrite<double>(string savename, Blob<double>& blob);

template <typename Dtype>
std::string num2str(Dtype n)
{ 
    std::stringstream ss;
    std::string str;
    ss<<n;
    ss>>str;
    return str;
}
template std::string num2str<int>(int n);
template std::string num2str<float>(float n);
template std::string num2str<double>(double n);

//template <typename Dtype>
float str2num(std::string str )
{ 
    std::stringstream ss;
    float num;
    ss<<str;
    ss>>num;
    return num;
}
/*template int str2num<int>(std::string str);
template float str2num<float>(std::string str);
template double str2num<double>(std::string str);*/


template <typename Dtype>
void SaveBlobData(string savename, Blob<Dtype>& blob)
{ 
    
       std::ofstream fout(savename.c_str());
       if(!fout.is_open())
       {
            LOG(INFO)<<"can not create file "<<savename;
       }
       else
       {
            const int count = blob.count();
            const Dtype * data = blob.cpu_data();
            for(int i=0;i< count;i++)
            {
                fout<<data[i]<<"\n";
            }
            fout.close();
       }
    
}

template void SaveBlobData<float>(string savename, Blob<float>& blob);
template void SaveBlobData<double>(string savename, Blob<double>& blob);


template <typename Dtype>
void SaveBlobDiff(string savename, Blob<Dtype>& blob)
{ 
    
       std::ofstream fout(savename.c_str());
       if(!fout.is_open())
       {
            LOG(INFO)<<"can not create file "<<savename;
       }
       else
       {
            const int count = blob.count();
            const Dtype * diff = blob.cpu_diff();
            for(int i=0;i< count;i++)
            {
                fout<<diff[i]<<"\n";
            }
            fout.close();
       }
}
template void SaveBlobDiff<float>(string savename, Blob<float>& blob);
template void SaveBlobDiff<double>(string savename, Blob<double>& blob);

template <typename Dtype>
void SaveArray(string savename, const Dtype* data,const int count)
{ 
    
       std::ofstream fout(savename.c_str());
       if(!fout.is_open())
       {
            LOG(INFO)<<"can not create file "<<savename;
       }
       else
       {
            
           
            for(int i=0;i< count;i++)
            {
                fout<<data[i]<<"\n";
            }
            fout.close();
       }
    
}

template void SaveArray<float>(string savename, const float* data,const int count);
template void SaveArray<double>(string savename, const double* data,const int count);

}  // namespace caffe
