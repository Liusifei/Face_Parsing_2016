#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include "time.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(testfile, "",
    "The test text file..");
DEFINE_string(meanfile, "",
    "The mean data file..");
DEFINE_string(outputlayername, "",
    "The outputlayername ..");
DEFINE_string(savefolder, "",
    "The savefolder ..");
DEFINE_int32(height, -1,
    "Run in GPU mode on given device ID.");
DEFINE_int32(width, -1,
    "Run in GPU mode on given device ID.");
DEFINE_bool(autoscale, false,
    "Run in GPU mode on given device ID.");
DEFINE_bool(autoresize, false,
    "Run in GPU mode on given device ID.");
DEFINE_string(testimagename, "",
    "The test image file..");
DEFINE_double(datascale, 0.00390625,
    "The data scale..");
DEFINE_int32(datachannel, 3,
    "The data channel..");
DEFINE_string(saveonefile, "",
    "save feature ..");



// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}
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

void get_caffe_inputdata(cv::Mat img,float * inputdata,float * meandata, float data_scale, bool color=true)
{
  int topindex=0;
  if (color) {
    for (int c = 0; c < img.channels(); ++c) {
      for (int h = 0; h < img.rows; ++h) {
        for (int w = 0; w < img.cols; ++w) {
          float datum_element = static_cast<float>(static_cast<unsigned char>(img.at<cv::Vec3b>(h, w)[c]));
            
           inputdata[topindex]=(datum_element-meandata[topindex])*data_scale;
           topindex++;
            
        }
      }
    }
  } else { 
    for (int h = 0; h < img.rows; ++h) {
      for (int w = 0; w < img.cols; ++w) {
          float datum_element = static_cast<float>(static_cast<unsigned char>(img.at<uchar>(h, w)));
          
           inputdata[topindex]=(datum_element-meandata[topindex])*data_scale;
           topindex++;
        }
      }
  }
}
cv::Mat get_outputmap(const vector<vector<Blob<float>*> >& top_vecs,int outidx,bool auto_scale=false)
{
    
  vector<Blob<float>*> outblobs = top_vecs[outidx];
  const float * outdata=outblobs[0]->cpu_data();
  int count = outblobs[0]->count(); 
  int outheight= outblobs[0]->height();
  int outwidth= outblobs[0]->width();
  int channels = outblobs[0]->channels();
  int spacedim = outheight*outwidth;
  cv::Mat result = cv::Mat(outheight,outwidth,CV_8UC1);

  float maxv=-FLT_MAX;
  int maxid=0;
  float v=0;
  
  int scale_rate=1;
  if(auto_scale)
  {
    scale_rate = 255/(channels-1);
  }
  
  for(int h=0;h<outheight;h++)
  {
    //unsigned char * pdata = result.ptr<unsigned char>(h);
    for(int w=0;w<outwidth;w++)
    {
         
        for(int c=0;c<channels;c++)
        {
          v=outdata[c*spacedim + h* outwidth + w];
          if (v > maxv)
          {
            maxv = v;
            maxid = c;
          }
        }
        if(auto_scale)
        {
            maxid = maxid * scale_rate;
        }
        result.at<unsigned char>(h, w)=(unsigned char)(maxid);
        maxv=-FLT_MAX;
        maxid=0;
    }
  }
  return result;
}
cv::Mat forwardNet(Net<float> &caffe_net,std::string outputlayername, cv::Mat inputimg,int height,int width,float * datamean,float datascale,bool auto_scale=false,bool auto_resize=false)
{ 

  int outidx=-1;
  cv::Mat dummyresult;
  int input_height = inputimg.rows;
  int input_width = inputimg.cols;
  cv::Mat cv_img_origin = inputimg;

  cv::Mat cv_img;
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " ;
    return dummyresult;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width,height));
  } else {
    cv_img = cv_img_origin;
  }
    const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
    //const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();
    const vector<Blob<float>* >& input_blobs = caffe_net.input_blobs();
  const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();

  Blob<float>* inputblob = input_blobs[0];
  float * inputdata = inputblob->mutable_cpu_data();
  //Blob<float>* inputblob = bottom_vecs[0][0];
  //float * inputdata = inputblob->mutable_cpu_data();
  get_caffe_inputdata(cv_img,inputdata,datamean,datascale,cv_img.channels()>1?true:false);
  /////-------------------  fp --------------------------///
  for (int i = 0; i < layers.size(); ++i) {
      const caffe::string& layername = layers[i]->layer_param().name();
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      if(outputlayername == layername)
      {
            outidx = i;
            break;
      }
  }
  
  /////-------------------  fp  --------------------------///

  cv::Mat result =  get_outputmap( top_vecs, outidx ,auto_scale);
  if(auto_resize)
  {
    cv::resize(result, result, cv::Size(input_width,input_height),0,0,cv::INTER_NEAREST);
  }
  return result;
}

cv::Mat forwardNet(Net<float> &caffe_net,std::string outputlayername, std::vector< cv::Mat > inputimg,int height,int width,float * datamean,float datascale,bool auto_scale=false,bool auto_resize=false)
{
  
  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<Blob<float>* >& input_blobs = caffe_net.input_blobs();
  const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();

  Blob<float>* inputblob = input_blobs[0];
  float * inputdata = inputblob->mutable_cpu_data();

  //Blob<float>* inputblob = bottom_vecs[0][0];
  //float * inputdata = inputblob->mutable_cpu_data();
  
  int outidx=-1;
  cv::Mat dummyresult;
  int input_height = inputimg[0].rows;
  int input_width = inputimg[0].cols;

  for(int i=0;i<inputimg.size();i++)
  {
    if (!inputimg[i].data) {
        LOG(ERROR) << "Could not open or find file " ;
        return dummyresult;
    }
    cv::Mat cv_img;
    if (height > 0 && width > 0) {
        cv::resize(inputimg[i], cv_img, cv::Size(width,height));
    } else {
        cv_img = inputimg[i];
    }
    get_caffe_inputdata(cv_img,inputdata+ i* cv_img.channels() *cv_img.rows * cv_img.cols,datamean,datascale);
  }
  
  /////-------------------  fp --------------------------///
  for (int i = 0; i < layers.size(); ++i) {
      
      const caffe::string& layername = layers[i]->layer_param().name();
      LOG(INFO)<<"forward layer:"<<layername;
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      if(outputlayername == layername)
      {
            outidx = i;
            break;
      }
  }
  
  /////-------------------  fp  --------------------------///

  cv::Mat result =  get_outputmap( top_vecs, outidx ,auto_scale);
  if(auto_resize)
  {
    cv::resize(result, result, cv::Size(input_width,input_height),0,0,cv::INTER_NEAREST);
  }
  return result;
}


cv::Mat forwardNet_forvoc_multi_input(Net<float> &caffe_net,std::string outputlayername, cv::Mat edge_img,std::vector<float>  feature_data,int height,int width,float * datamean,float datascale,bool auto_scale=false,bool auto_resize=false)
{
  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();
  Blob<float>* inputblob = bottom_vecs[0][0];
  float * inputdata = inputblob->mutable_cpu_data();
  
  int outidx=-1;
  cv::Mat dummyresult;
  int input_height = edge_img.rows;
  int input_width = edge_img.cols;

   
    if (!edge_img.data) {
        LOG(ERROR) << "Could not open or find file " ;
        return dummyresult;
    }
    cv::Mat cv_img;
    if (height > 0 && width > 0) {
        cv::resize(edge_img, cv_img, cv::Size(width,height));
    } else {
        cv_img = edge_img;
    }
    
    CHECK(inputblob->count()== cv_img.rows * cv_img.cols*cv_img.channels() + cv_img.rows * cv_img.cols*21);   
    
    //LOG(INFO)<<"before get png data"; 
    
    get_caffe_inputdata(cv_img,inputdata,datamean,datascale,false);
  
  
    float * feature_input = inputdata+cv_img.rows * cv_img.cols*cv_img.channels();
    CHECK(feature_data.size() == cv_img.rows * cv_img.cols*21);
    
    //LOG(INFO)<<"before get feature data"; 
    
    for(int i=0;i<cv_img.rows * cv_img.cols*21;i++)
    {
        feature_input[i] = feature_data[i];
    }
            
  //LOG(INFO)<<"before froward"; 
  /*
  FILE * fp = fopen("inputblob.txt","w");
  inputdata = inputblob->mutable_cpu_data();
  for(int i=0;i<inputblob->count();i++)
  {
    fprintf(fp,"%f\n",inputdata[i]);
  }
  fclose(fp);
  */
  

  
  /////-------------------  fp --------------------------///
  for (int i = 0; i < layers.size(); ++i) {
      const caffe::string& layername = layers[i]->layer_param().name();
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      //LOG(INFO)<<"after froward layer "<<layername; 
      if(outputlayername == layername)
      {
            outidx = i;
            break;
      }
  }
  
  /*fp = fopen("outputblob.txt","w");
  float * outputdata = top_vecs[outidx][0]->mutable_cpu_data();
  for(int i=0;i<top_vecs[outidx][0]->count();i++)
  {
    fprintf(fp,"%f\n",outputdata[i]);
  }
  fclose(fp);
  */
  
  /////-------------------  fp  --------------------------///
  //LOG(INFO)<<"before get_outputmap"; 
  cv::Mat result =  get_outputmap( top_vecs, outidx ,auto_scale);
  if(auto_resize)
  {
    cv::resize(result, result, cv::Size(input_width,input_height),0,0,cv::INTER_NEAREST);
  }
  
  //cv::imwrite("result.png",result);
  //CHECK(false);
  return result;
}

vector<Blob<float>*> extractNetfeature(Net<float> &caffe_net,std::string outputlayername, cv::Mat inputimg,int height,int width,float * datamean,float datascale)
{
  int outidx=-1;

  int input_height = inputimg.rows;
  int input_width = inputimg.cols;
  cv::Mat cv_img_origin = inputimg;

  vector<Blob<float>*> dummy;
  cv::Mat cv_img;
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " ;
    return dummy;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width,height));
  } else {
    cv_img = cv_img_origin;
  }
    const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
    const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();
    const vector<Blob<float>* >& input_blobs = caffe_net.input_blobs();  
  
  Blob<float>* inputblob =input_blobs[0];
  float * inputdata = inputblob->mutable_cpu_data();
  get_caffe_inputdata(cv_img,inputdata,datamean,datascale);
  if(outputlayername == "data")
  {
    return bottom_vecs[0];
  }
  /////-------------------  fp --------------------------///
  for (int i = 0; i < layers.size(); ++i) {
      const caffe::string& layername = layers[i]->layer_param().name();
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      //LOG(INFO)<<"forward layer name: "<<layername;
      if(outputlayername == layername)
      {
            outidx = i;
            break;
      }
  }
  if(outidx<0)
  {
     LOG(INFO)<<"do not find layer: "<<outputlayername;
     return dummy;
  }
  
  /////-------------------  fp  --------------------------///
  vector<Blob<float>*> outblobs =  top_vecs[outidx];
  return outblobs;
}

int test_saveimg() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  LOG(INFO)<<"after init net";
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO)<<"after copy trained layers";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
    int datasize = height * width *FLAGS_datachannel;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 128.0;
        }
    }
    else
    {
        LOG(INFO)<<"load data mean from "<<FLAGS_meanfile.c_str();
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
   LOG(INFO)<<"after load mean data"; 
    vector<std::pair<std::string, std::string> > pair_lines_;
    const std::string source = FLAGS_testfile;
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string filename;
    std::string label;
    while (infile >> filename >> label) {
        pair_lines_.push_back(std::make_pair(filename, label));
    }
    int length = pair_lines_.size();
    LOG(INFO)<<"found total "<<length<<" lines";
    for (int i = 0; i < length; ++i) {
    
        std::vector<std::string> imagenames = str_split(pair_lines_[i].first,"||");
        int cv_read_flag = (FLAGS_datachannel>1 ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagenames[0], cv_read_flag);
        
        double timecost = (double)cv::getTickCount();
        LOG(INFO)<<" forward image "<<imagenames[0]; 
        cv::Mat result = forwardNet(caffe_net, outputlayername, cv_img_origin, height, width,datamean,FLAGS_datascale,FLAGS_autoscale,FLAGS_autoresize);
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(!result.data)
		{
			LOG(INFO)<<"can not process "<<imagenames[0];
			continue;
		}
        std::vector<std::string> tempname = str_split(imagenames[0],"/");
        std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        
        //LOG(INFO) << i <<"process img " << imagenames[0];
        std::string savename = tempname2[0];
        if(tempname2.size()>2)
        {
           for(int temid=1;temid<tempname2.size()-1;temid++)
           {      savename +='.';
                 savename += tempname2[temid];
           }
        }
        savename=savefolder + savename +".png";
        LOG(INFO) << i << " save img " << savename<<", time="<<timecost<<" ms";
        cv::imwrite(savename, result);
  }
 
    delete [] datamean;
  return 0;
}
RegisterBrewFunction(test_saveimg);

int test_saveimg_new() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  LOG(INFO)<<"after init net";
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO)<<"after copy trained layers";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
    int datasize = height * width *FLAGS_datachannel;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 0;
        }
    }
    else
    {
        LOG(INFO)<<"load data mean from "<<FLAGS_meanfile.c_str();
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
   LOG(INFO)<<"after load mean data"; 
    //vector<std::pair<std::string, std::string> > pair_lines_;
    std::vector<std::string> lines;
    const std::string source = FLAGS_testfile;
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string filename;
    //std::string label;
    while (infile >> filename ) {
        lines.push_back(filename);
    }
    int length = lines.size();
    LOG(INFO)<<"found total "<<length<<" lines";
    for (int i = 0; i < length; ++i) {
    
        std::vector<std::string> imagenames = str_split(lines[i],"||");
        int cv_read_flag = (FLAGS_datachannel>1 ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagenames[0], cv_read_flag);
        
        double timecost = (double)cv::getTickCount();
       
        
        cv::Mat result = forwardNet(caffe_net, outputlayername, cv_img_origin, height, width,datamean,FLAGS_datascale,FLAGS_autoscale,FLAGS_autoresize);
        //LOG(INFO)<<"after forward";
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(!result.data)
		{
			LOG(INFO)<<"can not process "<<imagenames[0];
			continue;
		}
        std::vector<std::string> tempname = str_split(imagenames[0],"/");
        std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        
        //LOG(INFO) << i <<"process img " << imagenames[0];
        std::string savename = tempname2[0];
        if(tempname2.size()>2)
        {
           for(int temid=1;temid<tempname2.size()-1;temid++)
           {      savename +='.';
                 savename += tempname2[temid];
           }
        }
        savename=savefolder + savename +".png";
        LOG(INFO) << i << " save img " << savename<<", time="<<timecost<<" ms";
        cv::imwrite(savename, result);
  }
 
    delete [] datamean;
  return 0;
}
RegisterBrewFunction(test_saveimg_new);



int testpair_saveimg() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
    int datasize = height * width *FLAGS_datachannel;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 128.0;
        }
    }
    else
    {
        printf("load data mean from %s\n",FLAGS_meanfile.c_str());
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
    
    vector<std::pair<std::string, int> > pair_lines_;
    const std::string source = FLAGS_testfile;
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string filename;
    int label;
    while (infile >> filename >> label) {
        pair_lines_.push_back(std::make_pair(filename, label));
    }
    int length = pair_lines_.size();
    for (int i = 0; i < length; ++i) {
    
        std::vector<std::string> imagenames = str_split(pair_lines_[i].first,"||");
        int cv_read_flag = (FLAGS_datachannel>1 ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        std::vector< cv::Mat > inputimages;
        for(int k=0;k<imagenames.size()-1;k++)
        {
            cv::Mat cv_img_origin = cv::imread(imagenames[k], cv_read_flag);
            inputimages.push_back(cv_img_origin);
        }
        
        
        double timecost = (double)cv::getTickCount();
        
        cv::Mat result = forwardNet(caffe_net, outputlayername, inputimages, height, width,datamean,FLAGS_datascale,FLAGS_autoscale,FLAGS_autoresize);
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(!result.data)
		{
			LOG(INFO)<<"can not process "<<imagenames[0];
			continue;
		}
        std::vector<std::string> tempname = str_split(pair_lines_[i].first,".");
        //std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        std::string savename = str_split(tempname[1],"i")[1]+"_"+str_split(tempname[3],"i")[1];
        savename = savefolder + savename + ".png";
        //LOG(INFO) << i <<"process img " << imagenames[0];
        LOG(INFO) << i << " save img " << savename<<", time="<<timecost<<" ms";
        cv::imwrite(savename, result);
  }
 
    delete [] datamean;
  return 0;
}
RegisterBrewFunction(testpair_saveimg);

int test_voc_multi_input() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO)<<"after copy trained models!!"; 

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
    int datasize = height * width *FLAGS_datachannel;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 0.0;
        }
    }
    else
    {
        printf("load data mean from %s\n",FLAGS_meanfile.c_str());
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
    
    vector<std::pair<std::string, std::string> > pair_lines_;
    const std::string source = FLAGS_testfile;
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string filename;
    std::string label;
    while (infile >> filename >> label) {
        pair_lines_.push_back(std::make_pair(filename, label));
    }
    int length = pair_lines_.size();
    for (int i = 0; i < length; ++i) {
    
        std::string edge_imgname = pair_lines_[i].first;
        std::string feature_name = pair_lines_[i].second;
        int cv_read_flag = CV_LOAD_IMAGE_GRAYSCALE;//(FLAGS_datachannel>1 ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        
        
        cv::Mat cv_edgeimg_origin = cv::imread(edge_imgname, cv_read_flag);
        
        
        std::vector<float> featureinput;
        featureinput.resize(height * height * 21,0);
        
        std::ifstream inputfile(feature_name.c_str());
        if(!inputfile.is_open())
            LOG(FATAL)<<"can not open text label file "<<feature_name;
        float v;
        
        int featurecountnum=0;
        while(inputfile >> v) 
        {    
            featureinput[featurecountnum]=v;
            featurecountnum+=1;
        }
        assert(featurecountnum == featureinput.size());
        /*FILE * fp=fopen("inputfeature.txt","w");
        for(int kk=0;kk<featureinput.size();kk++)
        {
            fprintf(fp,"%f\n",featureinput[kk]);
        }
        fclose(fp);*/
        
        double timecost = (double)cv::getTickCount();
        
        cv::Mat result = forwardNet_forvoc_multi_input(caffe_net, outputlayername, cv_edgeimg_origin,featureinput, height, width,datamean,FLAGS_datascale,FLAGS_autoscale,FLAGS_autoresize);
        
        //LOG(INFO)<<"after fp net!!"; 
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
        //LOG(INFO)<<"after time cost cal!!"; 
        //cv::imwrite("result.png",result);
    	if(!result.data)
		{
			LOG(INFO)<<"can not process "<<edge_imgname;
			continue;
		}
        //LOG(INFO)<<"before get savename"<<pair_lines_[i].first; 
        std::vector<std::string> tempname = str_split(pair_lines_[i].second,"/");
        //LOG(INFO)<<"1"<<tempname[0];
        std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        std::string savename = tempname2[0];
        //LOG(INFO)<<"2"<<savename;
        savename = savefolder + savename + ".png";
        //LOG(INFO) << i <<"process img " << imagenames[0];
        LOG(INFO) << i << " save img " << savename<<", time="<<timecost<<" ms";
        cv::imwrite(savename, result);
  }
 
    delete [] datamean;
  return 0;
}
RegisterBrewFunction(test_voc_multi_input);

int test_extractfeature() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    bool inputcolorimg=true;
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  
    int datasize = height * width *3;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 128.0;
        }
    }
    else
    {
        printf("load data mean from %s\n",FLAGS_meanfile.c_str());
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
    
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
    vector<std::pair<std::string, std::string> > pair_lines_;
    const std::string source = FLAGS_testfile;
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string filename;
    std::string label;
    while (infile >> filename >> label) {
        pair_lines_.push_back(std::make_pair(filename, label));
    }
    int length = pair_lines_.size();

    
	bool savefeature = false;
	std::ofstream fout;
	if(FLAGS_saveonefile.size() >= 0)
	{
		savefeature = true;
		fout.open(FLAGS_saveonefile.c_str());
	}

    for (int i = 0; i < length; ++i) {
    
        std::vector<std::string> imagenames = str_split(pair_lines_[i].first,"||");
        int cv_read_flag = (inputcolorimg ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagenames[0], cv_read_flag);
        
        double timecost = (double)cv::getTickCount();
        
        vector<Blob<float>*> outblobs = extractNetfeature(caffe_net, outputlayername, cv_img_origin, height, width,datamean,FLAGS_datascale);
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(outblobs.size()<1||!outblobs[0]->count())
		{
			LOG(INFO)<<"can not process "<<imagenames[0];
			continue;
		}
        std::vector<std::string> tempname = str_split(imagenames[0],"/");
        std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        std::string savename = tempname2[0];
        if(tempname2.size()>2)
        {
           for(int temid=1;temid<tempname2.size()-1;temid++)
           {      savename +='.';
                 savename += tempname2[temid];
           }
        }
     
        savename = savefolder + savename + ".txt";
        //LOG(INFO) << i <<"process img " << imagenames[0];
        LOG(INFO) << i << " save data " << savename<<", time="<<timecost<<" ms";
        
        const float * outdata=outblobs[0]->cpu_data();

		if(savefeature)
		{
			fout<<imagenames[0];
			for(int num = 0;num < outblobs[0]->count();num++)
			{
				fout<<" ";
				fout<<outdata[num];
			}
			fout<<"\n";
		}

        FILE *fp =fopen(savename.c_str(),"w");
        for(int num=0;num<outblobs[0]->count();num++)
        {
            fprintf(fp,"%f\n",outdata[num]);
        }
        fclose(fp);
        //cv::imwrite(savename, result);
  }

  	if(savefeature)
	{
		fout.close();
	}

    delete [] datamean;
  return 0;
}
RegisterBrewFunction(test_extractfeature);


int test_extractfeatureforfacerec() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    bool inputcolorimg=true;
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  
    int datasize = height * width *3;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 0.0;
        }
    }
    else
    {
        printf("load data mean from %s\n",FLAGS_meanfile.c_str());
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
    
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
    vector<std::pair<std::string, std::string> > pair_lines_;
    const std::string source = FLAGS_testfile;
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string filename;
    std::string label;
    while (infile >> filename >> label) {
        pair_lines_.push_back(std::make_pair(filename, label));
    }
    int length = pair_lines_.size();


	bool savefeature = false;
	std::ofstream fout;
	if(FLAGS_saveonefile.size() >= 0)
	{
		savefeature = true;
		fout.open(FLAGS_saveonefile.c_str());
	}


    for (int i = 0; i < length; ++i) {
    
        std::vector<std::string> imagenames = str_split(pair_lines_[i].first,"||");
        int cv_read_flag = (inputcolorimg ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagenames[0], cv_read_flag);
        
        double timecost = (double)cv::getTickCount();
        
        vector<Blob<float>*> outblobs = extractNetfeature(caffe_net, outputlayername, cv_img_origin, height, width,datamean,FLAGS_datascale);
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(outblobs.size()<1||!outblobs[0]->count())
		{
			LOG(INFO)<<"can not process "<<imagenames[0];
			continue;
		}

        std::vector<std::string> tempname = str_split(imagenames[0],".");
        std::string savename = tempname[0]+"_fea.txt";

        LOG(INFO) << i << " save data " << savename<<", time="<<timecost<<" ms";
        

	const float * outdata=outblobs[0]->cpu_data();

		if(savefeature)
		{
			fout<<imagenames[0];
			for(int num = 0;num < outblobs[0]->count();num++)
			{
				fout<<" ";
				fout<<outdata[num];
			}
			fout<<"\n";
		}

        
        FILE *fp =fopen(savename.c_str(),"w");
        for(int num=0;num<outblobs[0]->count();num++)
        {
            fprintf(fp,"%f\n",outdata[num]);
        }
        fclose(fp);
        //cv::imwrite(savename, result);
  }

	if(savefeature)
	{
		fout.close();
	}

    delete [] datamean;
  return 0;
}
RegisterBrewFunction(test_extractfeatureforfacerec);


int test_extract_one_feature() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    bool inputcolorimg=true;
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  
    int datasize = height * width *3;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 128.0;
        }
    }
    else
    {
        printf("load data mean from %s\n",FLAGS_meanfile.c_str());
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
    
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
 
    const std::string imagename = FLAGS_testimagename;
    LOG(INFO) << "Opening file " << imagename;
    
    
       int cv_read_flag = (inputcolorimg ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagename, cv_read_flag);
        
        double timecost = (double)cv::getTickCount();
        
        vector<Blob<float>*> outblobs = extractNetfeature(caffe_net, outputlayername, cv_img_origin, height, width,datamean,FLAGS_datascale);
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(outblobs.size()<1||!outblobs[0]->count())
		{
			LOG(INFO)<<"can not process "<<imagename;
			return 0;
		}
        std::vector<std::string> tempname = str_split(imagename,"/");
        std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        std::string savename = tempname2[0];
        savename = savefolder + savename + ".txt";
        //LOG(INFO) << i <<"process img " << imagenames[0];
        LOG(INFO)<< " save data " << savename<<", time="<<timecost<<" ms";
        
        const float * outdata=outblobs[0]->cpu_data();
        FILE *fp =fopen(savename.c_str(),"w");
        for(int num=0;num<outblobs[0]->count();num++)
        {
            fprintf(fp,"%f\n",outdata[num]);
        }
        fclose(fp);
        //cv::imwrite(savename, result);
  
    delete [] datamean;
  return 0;
}
RegisterBrewFunction(test_extract_one_feature);

int test_extractparam() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    bool inputcolorimg=true;
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  
  if(outputlayername == "ALL")
  {
     vector<std::string> layernames =  caffe_net.layer_names();
     for(int k=0;k<layernames.size();k++)
     {
        shared_ptr<Layer<float> > layer =  caffe_net.layer_by_name(layernames[k]);
        vector<shared_ptr<Blob<float> > > paramblobs =  layer->blobs();
        LOG(INFO)<<layernames[k]<<" have "<<paramblobs.size()<<" params.";
        for(int i=0;i<paramblobs.size();i++)
        {
            std::stringstream ss;
            std::string str;
            ss<<i;
            ss>>str;
            std::string savename = layernames[k]+"_param_"+str+".txt";
            const float * paramdata=paramblobs[i]->cpu_data();
            FILE *fp =fopen(savename.c_str(),"w");
            for(int num=0;num<paramblobs[i]->count();num++)
            {
                fprintf(fp,"%f\n",paramdata[num]);
            }
            fclose(fp);
        }
     }
  }
  else
  {
    shared_ptr<Layer<float> > layer =  caffe_net.layer_by_name(outputlayername);
    vector<shared_ptr<Blob<float> > > paramblobs =  layer->blobs();
    LOG(INFO)<<outputlayername<<" have "<<paramblobs.size()<<" params.";
    for(int i=0;i<paramblobs.size();i++)
    {
        std::stringstream ss;
        std::string str;
        ss<<i;
        ss>>str;
        std::string savename = outputlayername+"_param_"+str+".txt";
        const float * paramdata=paramblobs[i]->cpu_data();
        FILE *fp =fopen(savename.c_str(),"w");
        for(int num=0;num<paramblobs[i]->count();num++)
        {
            fprintf(fp,"%f\n",paramdata[num]);
        }
        fclose(fp);
    }
  }


    
   
 
  return 0;
}
RegisterBrewFunction(test_extractparam);

cv::Mat drawParsing(cv::Mat img, cv::Mat mask,vector<cv::Scalar> color)
{
    

    float alpha = 0.65;
   for(int h=0;h<img.rows;h++)
   {
     for(int w=0;w<img.cols;w++)
      {
         unsigned char v = mask.at<unsigned char>(h, w);
         if(v==0)
            continue;
         for(int c=0;c<3;c++)
         {
         int v1 = img.at<cv::Vec3b>(h, w)[c];
         int v2 = color[v][c];
         v1 = (int)((float)v1* alpha + (float)v2*(1-alpha));
         img.at<cv::Vec3b>(h, w)[c] =  (unsigned char)v1;
         }
       }
    }
    return img;
}

cv::Rect get_crop_rect (cv::Mat img, cv::Rect face, int border)
{
  if(face.x-border<0)  border = face.x;
  if(face.x + face.width + border >= img.cols)  border = img.cols - face.x-face.width-1;
  if(face.y-border<0)  border = face.y;
  if(face.y+ face.height + border >= img.rows)  border = img.rows - face.y - face.height -1;
  cv::Rect bigface;
  bigface.x = face.x - border;
  bigface.y = face.y - border;
  bigface.width = face.width + border*2;
  bigface.height = face.height + border*2;
  return bigface;
}

std::vector< cv::Rect > detect_face(cv::Mat image,CvHaarClassifierCascade * detector)
{
    IplImage temp = IplImage(image);  
    IplImage *pSrcImage =&temp;

    IplImage *pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);
    cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY);
        CvMemStorage *pcvMStorage = cvCreateMemStorage(0);
        cvClearMemStorage(pcvMStorage);
        // Ê¶±ð
        
        CvSeq *pcvSeqFaces = cvHaarDetectObjects(pGrayImage, detector, pcvMStorage,1.3,2,CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_ROUGH_SEARCH,cvSize(100,100));
        //dwTimeEnd = GetTickCount();

        //printf("ÈËÁ³žöÊý: %d   Ê¶±ðÓÃÊ±: %d ms\n", pcvSeqFaces->total, dwTimeEnd - dwTimeBegin);

        std::vector< cv::Rect > result;
        // ±êŒÇ
        for(int i = 0; i <pcvSeqFaces->total; i++)
        {
            CvRect* r = (CvRect*)cvGetSeqElem(pcvSeqFaces, i);
            CvPoint center;
            int radius;
            cv::Rect rr ;
            rr.width = r->width;
            rr.height = r->height;
            rr.x = r->x;
            rr.y = r->y;
            result.push_back(rr);
            
        }
        cvReleaseMemStorage(&pcvMStorage);
    //cvReleaseImage(&pSrcImage);    
    cvReleaseImage(&pGrayImage);
    return result;
}

void demolfw(Net<float> &caffe_net,std::string outputlayername, cv::Mat origin,cv::Rect bigface,cv::Mat & showimg,int height,int width, vector<cv::Scalar> color,float * datamean,float datascale)
{
    int outidx=-1;
  
  cv::Mat cv_img_origin = origin(bigface);

  cv::Mat cv_img;
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " ;
    return ;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv_img_origin;
  }
    const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
    const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();
    
  
  Blob<float>* inputblob = bottom_vecs[0][0];
  float * inputdata = inputblob->mutable_cpu_data();
  get_caffe_inputdata(cv_img,inputdata,datamean,datascale);
  /////-------------------  fp --------------------------///
  for (int i = 0; i < layers.size(); ++i) {
      const caffe::string& layername = layers[i]->layer_param().name();
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      if(outputlayername == layername)
      {
            outidx = i;
            break;
      }
  }
  
  /////-------------------  fp  --------------------------///

  cv::Mat result = get_outputmap( top_vecs, outidx );
  cv::Mat map;
  cv::resize(result,map,cv::Size(cv_img_origin.cols,cv_img_origin.rows),cv::INTER_NEAREST);

    cv::Mat bigmap = cv::Mat::zeros(origin.rows,origin.cols,CV_8UC1);
    cv::Mat temp = bigmap(bigface);
    map.copyTo(temp);
    showimg = drawParsing(showimg,bigmap,color);
    
}

// demo for lfw,liangji,20141230
int face_demo() {
   
    bool showlfw=false;
    bool showhelen=true;
  
    clock_t start, finish;  
    // face detect model
    const char *pstrCascadeFileName = "./models/haarcascade_frontalface_alt.xml";
    CvHaarClassifierCascade *pHaarCascade = NULL;
    pHaarCascade = (CvHaarClassifierCascade*)cvLoad(pstrCascadeFileName);
    const std::string helen_model="./models/helen_deploy.prototxt";
    const std::string helen_weights="./models/helen_iter_100000.caffemodel";


   
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  
  
    
  // Instantiate the caffe net.
  //Caffe::set_phase(Caffe::TEST);
  Net<float> caffe_net(FLAGS_model,caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "load model over!";

  Net<float> helen_net(helen_model,caffe::TEST);
  helen_net.CopyTrainedLayersFrom(helen_weights);
  LOG(INFO) << "load helen model over!"; 
  
  //------------set param ----------------------//
  const caffe::string outputlayername="deconv8";
  int height = 100;
  int width = 100;

  
  const caffe::string helen_outputlayername="deconv8";
  int helen_h=192;
  int helen_w=192; 
  
  
    int datasize = height * width *3;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 128.0;
        }
    }
    else
    {
        printf("load data mean from %s\n",FLAGS_meanfile.c_str());
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
    
    int helen_datasize = helen_h * helen_w *3;
    float* helen_datamean=new float [helen_datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<helen_datasize;tempidx++)
        {
            helen_datamean[tempidx] = 128.0;
        }
    }
    else
    {
        printf("load data mean from %s\n",FLAGS_meanfile.c_str());
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<helen_datasize;k++) 
            {
                fscanf(fp,"%f\n",&helen_datamean[k]);
            }
        fclose(fp);
    }
    
    vector<cv::Scalar> color;
    color.push_back(CV_RGB(0,0,0));//the first class must be background

    color.push_back(CV_RGB(255,0,0));
    color.push_back(CV_RGB(0,255,0));
    color.push_back(CV_RGB(0,0,255));

    color.push_back(CV_RGB(255,255,0));
    color.push_back(CV_RGB(255,0,255));
    color.push_back(CV_RGB(0,255,255));

    color.push_back(CV_RGB(255,0,127));
    color.push_back(CV_RGB(0,127,255));
    color.push_back(CV_RGB(127,0,255));
    color.push_back(CV_RGB(0,127,0));
    color.push_back(CV_RGB(0,0,127));
    color.push_back(CV_RGB(127,0,0));

    

  cv::VideoCapture cap(0); 
  if(!cap.isOpened())  // check if we succeeded
  { 
    LOG(INFO) << "can not open video ";     
    return -1;
  }
  int count=0;
  while(1)
  {
  count ++;
  if(count==2)
  {
	count =0;
	continue;
  }
  /////--------------------- get input data --------------------------///
  cv::Mat origin;
  cap >> origin;
  if(origin.empty())
  {
    LOG(INFO) << "frame empty";     
      break;
  }
  
  start = clock();  
  std::vector <cv::Rect> facerects  = detect_face(origin,pHaarCascade);
  finish = clock();  
  printf("face detect time = %d\n",(finish - start)*1000/CLOCKS_PER_SEC );
  
  if(facerects.size()==0)
  {
      imshow("win",origin);
      cv::waitKey(2);
      continue;
  }
  cv::Rect face = facerects[0];
  cv::Mat showimg = origin.clone();
    
  if(showlfw)
  {
  
    start = clock();  
    cv::Rect bigface = get_crop_rect (origin, face, face.width/2);
    demolfw( caffe_net, outputlayername,origin,bigface,showimg, height, width,color,datamean,FLAGS_datascale);
    finish = clock();  
    printf("lfw forward time = %d\n",(finish - start)*1000/CLOCKS_PER_SEC );
  }
  if(showhelen)
  {
    start = clock();  
    cv::Rect bigface = get_crop_rect (origin, face, 0);
    demolfw( helen_net, helen_outputlayername,origin,bigface,showimg, helen_h, helen_w,color,helen_datamean,FLAGS_datascale);
    finish = clock();  
    printf("helen forward time = %d\n",(finish - start)*1000/CLOCKS_PER_SEC );
  }
 
    
   cv::imshow("win",showimg);
   int c=cv::waitKey(20);
   if(c=='q')
    break;
    
    
  }//end while
  cv::destroyAllWindows();

  delete [] datamean;
  delete [] helen_datamean;
  return 0;
}
RegisterBrewFunction(face_demo);


int multiface_demo() {
   
  
  
    clock_t start, finish;  
    // face detect model
    //const char *pstrCascadeFileName = "./models/haarcascade_frontalface_alt.xml";
    //CvHaarClassifierCascade *pHaarCascade = NULL;
    //pHaarCascade = (CvHaarClassifierCascade*)cvLoad(pstrCascadeFileName);
    //const std::string helen_model="./models/helen_deploy.prototxt";
    //const std::string helen_weights="./models/helen_iter_100000.caffemodel";


   
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  //Caffe::set_phase(Caffe::TEST);
  Net<float> caffe_net(FLAGS_model,caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "load model over!";

  /*Net<float> helen_net(helen_model,caffe::TEST);
  helen_net.CopyTrainedLayersFrom(helen_weights);
  LOG(INFO) << "load helen model over!"; 
  */
  //------------set param ----------------------//
  const caffe::string outputlayername="score8-6";
  int height = 512;
  int width = 512;

  int datasize = height * width *3;
    float* datamean=new float [datasize];
    if(FLAGS_meanfile.size()<1)
    {
        for(int tempidx=0;tempidx<datasize;tempidx++)
        {
            datamean[tempidx] = 128.0;
        }
    }
    else
    {
        printf("load data mean from %s\n",FLAGS_meanfile.c_str());
        FILE * fp =fopen(FLAGS_meanfile.c_str(),"r");
            int k=0;
            for( k=0;k<datasize;k++) 
            {
                fscanf(fp,"%f\n",&datamean[k]);
            }
        fclose(fp);
    }
    
  /*const caffe::string helen_outputlayername="deconv8";
  int helen_h=192;
  int helen_w=192; */
  
    vector<cv::Scalar> color;
    color.push_back(CV_RGB(0,0,0));//the first class must be background

    color.push_back(CV_RGB(255,0,0));
    color.push_back(CV_RGB(0,255,0));
    color.push_back(CV_RGB(0,0,255));

    color.push_back(CV_RGB(255,255,0));
    color.push_back(CV_RGB(255,0,255));
    color.push_back(CV_RGB(0,255,255));

    color.push_back(CV_RGB(255,0,127));
    color.push_back(CV_RGB(0,127,255));
    color.push_back(CV_RGB(127,0,255));
    color.push_back(CV_RGB(0,127,0));
    color.push_back(CV_RGB(0,0,127));
    color.push_back(CV_RGB(127,0,0));

    

  cv::VideoCapture cap(0); 
  if(!cap.isOpened())  // check if we succeeded
  { 
    LOG(INFO) << "can not open video ";     
    return -1;
  }
  int count=0;
  while(1)
  {
  /*count ++;
  if(count==2)
  {
	count =0;
	continue;
  }
  */
  
  /////--------------------- get input data --------------------------///
  cv::Mat origin;
  cap >> origin;
  if(origin.empty())
  {
    LOG(INFO) << "frame empty";     
      break;
  }
  
  /*start = clock();  
  std::vector <cv::Rect> facerects  = detect_face(origin,pHaarCascade);
  finish = clock();  
  printf("face detect time = %d\n",(finish - start)*1000/CLOCKS_PER_SEC );
  */
  /*if(facerects.size()==0)
  {
      imshow("win",origin);
      cv::waitKey(2);
      continue;
  }*/
  //cv::Rect face = facerects[0];
  cv::Mat showimg = origin.clone();
    
    int bound = (origin.cols - origin.rows)/2;
    cv::Rect bigface ;
    bigface.x=bound;bigface.y=0;
    bigface.width=origin.rows;bigface.height = origin.rows;
  
    start = clock();  
 
    demolfw( caffe_net, outputlayername,origin,bigface,showimg, height, width,color,datamean,FLAGS_datascale);
    finish = clock();  
    printf("parsing forward time = %d\n",(finish - start)*1000/CLOCKS_PER_SEC );
  
  
   cv::imshow("win",showimg);
   int c=cv::waitKey(20);
   if(c=='q')
    break;
    
    
  }//end while
  cv::destroyAllWindows();
  delete [] datamean;
  return 0;
}
RegisterBrewFunction(multiface_demo);

int test_cam()
{
 cv::Mat img; 
 // CvCapture * cap = cvCreateCameraCapture(-1); 
  cv::VideoCapture cap(0);
  LOG(INFO)<<"after create cap";
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  //IplImage * p=NULL;

 if(!cap.isOpened())
 {
     LOG(INFO)<<"can not open video";
    return -1;
  }
 
  while(1)
  {
   // p = cvQueryFrame(cap);
    // cv::Mat img(p);
      cap >> img;
     if(img.empty())
     {
        LOG(INFO)<<"frame empty";
        break;

     }
    cv::imshow("win",img);
    int c=cv::waitKey(50);
    if(c == 'q')
       break;

  }
  cv::destroyAllWindows();
  return 0;
}
RegisterBrewFunction(test_cam);

int main(int argc, char** argv) {
    // Print output to stderr (while still logging).
#ifdef USE_CAFFE_MPI_VERSION
    MPI_Init(&argc, &argv);
#endif
    FLAGS_alsologtostderr = 1;
    // Usage message.
    gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  test_saveimg    test and save result as image\n"
      "  test_extractfeature  extract and save feature data \n");
    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);
    if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
    } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
    }
}
