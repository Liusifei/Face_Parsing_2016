#include <cstdio>

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/util_img.hpp"
//#include <math.h>

namespace caffe {

int cmp(const void * a, const void * b)
{
     return((*(float*)a-*(float*)b>0)?1:-1);
}


template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    test_nets_.clear();
    //InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::SaveBlobDataDiff() {
    const vector<string>& blob_names = net_->blob_names();
    //LOG(INFO) << "------------ PrintDataDiff -------------";
    for(int i=0;i< blob_names.size();i++)
    {
        const shared_ptr<Blob<Dtype> > blob = net_->blob_by_name(blob_names[i]); 
        if(blob == NULL)
        {
            LOG(INFO) << "Blob "<<blob_names[i]<<" do not have data or diff";
            continue;
        }
        std::string savedataname = param_.save_blob_path()+"BlobData_iter_"+num2str(iter_)+"_"+blob_names[i]+".txt";
        SaveBlobDataWithoutOverwrite(savedataname,*(blob));
        LOG(INFO) << "save Blob data "<<blob_names[i]<<" in "<<savedataname;
        std::string savediffname = param_.save_blob_path()+"BlobDiff_iter_"+num2str(iter_)+"_"+blob_names[i]+".txt";
        SaveBlobDiffWithoutOverwrite(savediffname,*(blob));
        LOG(INFO) << "save Blob diff "<<blob_names[i]<<" in "<<savediffname;
        
    }
}

template <typename Dtype>
void Solver<Dtype>::PrintDataDiff() {
    const vector<string>& blob_names = net_->blob_names();
    //LOG(INFO) << "------------ PrintDataDiff -------------";
    for(int i=0;i< blob_names.size();i++)
    {
        const shared_ptr<Blob<Dtype> > blob = net_->blob_by_name(blob_names[i]); 
        if(blob == NULL)
        {
            LOG(INFO) << "Blob "<<blob_names[i]<<" do not have data or diff";
            continue;
        }
        blob->run_statistics();
        if(blob->count() == 1)
        {   
            Dtype data = blob->cpu_data()[0];
            Dtype diff = blob->cpu_diff()[0];
            LOG(INFO) << "Blob "<<blob_names[i] <<" data "
                    << ": min=" << data << " max=" << data<<" mean="<<data
                    <<" asum="<<data<<" std="<<data;
            LOG(INFO) << "Blob "<<blob_names[i] <<" diff "
                    << ": min=" << diff << " max=" << diff<<" mean="<<diff
                    <<" asum="<<diff<<" std="<<diff;
        }
        else
        {
            LOG(INFO) << "Blob "<<blob_names[i] <<" data "
                        << ": min=" << blob->min_data() << " max=" << blob->max_data()<<" mean="<<blob->mean_data()
                        <<" asum="<<blob->asum_data()<<" std="<<blob->std_data();
            LOG(INFO) << "Blob "<<blob_names[i] <<" diff "
                        << ": min=" << blob->min_diff() << " max=" << blob->max_diff()<<" mean="<<blob->mean_diff()
                        <<" asum="<<blob->asum_diff()<<" std="<<blob->std_diff();;   
        }
    }
}

template <typename Dtype>
void Solver<Dtype>::PrintParam() {
    const vector<shared_ptr<Layer<Dtype> > >& layers = net_->layers();
    const vector<string>& layer_names = net_->layer_names();
    //LOG(INFO) << "-------------- PrintParam ---------------";
    for(int i=0;i<layers.size();i++)
    {
        vector<shared_ptr<Blob<Dtype> > >& params = layers[i]->blobs();
        for(int j = 0; j < params.size(); ++j) {
        
            params[j]->run_statistics();
            if(params[j]->count() == 1)
            {
                Dtype data = params[j]->cpu_data()[0];
                Dtype diff = params[j]->cpu_diff()[0];
                LOG(INFO) << "Param "<< layer_names[i] << " param " << j <<" data "
                            << ": min=" << data << " max=" << data <<" mean="<<data
                            <<" asum="<<data<<" std="<<data;
                LOG(INFO) << "Param "<< layer_names[i] << " param " << j <<" diff "
                            << ": min=" <<diff << " max=" << diff<<" mean"<<diff
                            <<" asum="<<diff<<" std="<<diff;
            }
            else
            {
                LOG(INFO) << "Param "<< layer_names[i] << " param " << j <<" data "
                            << ": min=" << params[j]->min_data() << " max=" << params[j]->max_data()<<" mean="<<params[j]->mean_data()
                            <<" asum="<<params[j]->asum_data()<<" std="<<params[j]->std_data();
                            
                LOG(INFO) << "Param "<< layer_names[i] << " param " << j <<" diff "
                            << ": min=" << params[j]->min_diff() << " max=" << params[j]->max_diff()
                            <<" mean="<<params[j]->mean_diff()
                            <<" asum="<<params[j]->asum_diff()<<" std="<<params[j]->std_diff();
            }
        }
    }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;

  while (iter_ < stop_iter) {
    // zero-init the params
    net_->ClearParamDiffs();
    

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_.iter_size();
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", loss = " << smoothed_loss_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }

    if (param_.print_param() > 0 && (iter_ ) % param_.print_param() == 0 && Caffe::root_solver()) 
    {
      LOG(INFO)<< " ----------------  print param data diff before update ---------------- ";
      PrintParam();
    }

    ApplyUpdate();

    if (param_.print_datadiff() > 0&& (iter_ ) % param_.print_datadiff() == 0 && Caffe::root_solver()) 
    {
      LOG(INFO)<< " ----------------  print blob data diff ---------------- ";
      PrintDataDiff();
    }
    if (param_.print_param() >0 && (iter_ ) % param_.print_param() == 0 && Caffe::root_solver()) 
    {
      LOG(INFO)<< " ----------------  print param data diff after update ---------------- ";
      PrintParam();
    }
    if (param_.save_blob() >0 && (iter_ ) % param_.save_blob() == 0 && Caffe::root_solver()) 
    {
      SaveBlobDataDiff();
    }

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }

    ++iter_;
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}
template <typename Dtype>
vector<Blob<float>*> Solver<Dtype>::Get_net_output_blob(Net<float> &caffe_net,std::vector< cv::Mat> inputimages,TestProto testproto)
{
	const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
	const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
	const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
	const vector<Blob<float>* >& input_blobs = caffe_net.input_blobs();
	

	CHECK(input_blobs.size() == inputimages.size());

	
	CHECK(testproto.has_output_layername());
	std::string outputlayername = testproto.output_layername();	

	for(int i=0;i<inputimages.size();i++)
	{
	   
		cv::Mat image = inputimages[i];
		
   
		Blob<float>* inputblob = input_blobs[i];
  
    		float * inputdata = inputblob->mutable_cpu_data();
  
   		int idcount=0;
		const int channel = image.channels();
		const float datascale = testproto.datascale();
		const float meanvalue = testproto.meanvalue();

		CHECK(channel == 1 || channel ==3);
		CHECK(image.rows == inputblob->height());
		CHECK(image.cols == inputblob->width());
		CHECK(image.channels() == inputblob->channels());
		CHECK(inputblob->num()==1);

		if(channel ==3)
		{
			for (int c = 0; c < channel; ++c) 
			{                            
				for (int h = 0; h < image.rows; ++h) {
				    for (int w = 0; w < image.cols; ++w) {

					float v =static_cast<Dtype>(static_cast<unsigned char>(image.at<cv::Vec3b>(h, w)[c]));
					inputdata[idcount] = (v-meanvalue)*datascale;
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
				inputdata[idcount] =  (v-meanvalue)*datascale;
				idcount++;
		
			    }
			}
		}
	}//end for
 
	int outidx = -1;
	for (int i = 0; i < layers.size(); ++i)
	{
		const caffe::string& layername = layers[i]->layer_param().name();
		layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
		layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
		if(outputlayername == layername)
		{
		    outidx = i;
		    break;
		}
	}
   
	vector<Blob<float>*> outblobs;
	if(outidx<0)
	{
		LOG(INFO)<<"do not find layer: "<<outputlayername;
		return outblobs;
	}
	outblobs =  top_vecs[outidx];
	
	

	return outblobs;
	

}
template <typename Dtype>
std::vector<float> Solver<Dtype>::Get_net_output(Net<float> &caffe_net,std::vector< cv::Mat> inputimages,TestProto testproto)
{
	std::vector<float> result;

	vector<Blob<float>*> outblobs =  Get_net_output_blob(caffe_net, inputimages, testproto);
	if(outblobs.size()<1)
	{
		LOG(INFO)<<"do not get outblobs.";
		return result;
	}
	
	result.clear();

	const float * outdata=outblobs[0]->cpu_data();
	for(int num=0;num<outblobs[0]->count();num++)
        {
           result.push_back(outdata[num]);
        }


	return result;
	
	
}

template <typename Dtype>
void Solver<Dtype>::Test_Segment(TestProto testproto) {
	std::string testfile = testproto.test_file();
	std::string datafolder = testproto.data_folder();
	std::string deploynet = testproto.deploy_net();
	FLAGS_minloglevel=1;
	NetParameter net_param;
	this->net_->ToProto(&net_param);
	Net<float> caffe_net(deploynet, caffe::TEST);
  	caffe_net.CopyTrainedLayersFrom(net_param);
	FLAGS_minloglevel=0;

	LOG(INFO)<<"open test file:"<<testfile;
	LOG(INFO)<<"Iteration:" << iter_ <<"  TestingResult:Test_Segment"; 

	std::ifstream infile(testfile.c_str());
	string line;
	string splitflag = testproto.source_splitflag();
	string imname;
	string labelname;
	int datachannels = testproto.data_channels();
	CHECK(datachannels == 1 || datachannels == 3);
	float maxv,maxid;
	float totalcount=0;
	float errorcount=0;
	int width = testproto.width();
	int height = testproto.height();

	vector<float> total_insec, total_un;
	total_insec.clear();
	total_un.clear();
	float total_acc=0;
	float sample_count=0;
	int outwidth = testproto.out_width();
	int outheight = testproto.out_height();
	int outchannels = testproto.out_channels();
	while(infile >> line)
	{
		vector<string> strs = str_split(line,splitflag);

		CHECK(strs.size()==2);
		imname = datafolder + strs[0];
		labelname = datafolder + strs[1];
		cv::Mat img = cv::imread(imname,datachannels>1?CV_LOAD_IMAGE_COLOR:CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat label = cv::imread(labelname,CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat resizeimg,resizelabel;
		cv::resize(img,resizeimg,cv::Size(width,height),cv::INTER_LINEAR);
		cv::resize(label,resizelabel,cv::Size(outwidth,outheight),cv::INTER_NEAREST);
		vector<Mat> inputs;
		inputs.push_back(resizeimg);
		//std::vector<float> out = Get_net_output(caffe_net,inputs,testproto);
		vector<Blob<float>*> outblobs = Get_net_output_blob(caffe_net,inputs,testproto);
		float maxv;
  		int maxid=0;
  		float v=0;
		CHECK(outblobs[0]->height() == resizelabel.rows)<<outblobs[0]->height() << " vs "<<resizelabel.rows;
		CHECK(outblobs[0]->width() == resizelabel.cols);
		CHECK(outblobs[0]->channels() == outchannels);
		const float * outdata=outblobs[0]->cpu_data();
		int channels = outblobs[0]->channels();	
		int spacedim = outwidth * outheight;
                vector<float> cls_insec,cls_un;
		cls_insec.resize(channels,0);
		cls_un.resize(channels,0);
		float acc=0;
		unsigned char pred,lb;
		for(int h=0;h<resizelabel.rows;h++)
		{
			for(int w=0;w<resizelabel.cols;w++)
			{
				maxv = outdata[h* outwidth + w] -1.;
				maxid = 0;
				for(int c=0;c<outblobs[0]->channels();c++)
				{
					v=outdata[c*spacedim + h* outwidth + w];
					if (v > maxv)
					{
						maxv = v;
						maxid = c;
					}
				}
				pred = static_cast<unsigned char>(maxid);
				lb = resizelabel.at<unsigned char>(h,w);
				CHECK(lb<=channels);
				if(pred == lb)	
				{
					acc +=1.;
					cls_insec[lb]+=1.;
					cls_un[lb]+=1.;
				}
				else
				{
					cls_un[pred]+=1.;
					cls_un[lb]+=1.;
				}				
					
			}		
		}
                
		acc = acc / static_cast<float>(outwidth * outheight);
		
		total_acc += acc;
		if(total_insec.size()==0)
		{
			total_insec = cls_insec;
			total_un = cls_un;
		}
		else
		{
			for(int k=0;k<total_insec.size();k++)
			{
				total_insec[k]+= cls_insec[k];
				total_un[k] += cls_un[k];
			}
		}

		sample_count+=1.;
	}
	infile.close();

	LOG(INFO)<<"total test image num :"<<sample_count;     	
	if(sample_count==0) total_acc = 0;
	else total_acc /= sample_count;


	
	LOG(INFO)<<"Pixel Accuracy:"<<total_acc;
	for(int k=0;k<total_insec.size();k++)
	{
		float iou=0;
		if(int(total_un[k])!=0)
			iou = total_insec[k]/total_un[k];
		LOG(INFO)<<"class "<<k<<" IOU: "<<iou;	
	}
	
}
template <typename Dtype>
void Solver<Dtype>::Test_Classify(TestProto testproto) {
	std::string testfile = testproto.test_file();
	std::string datafolder = testproto.data_folder();
	std::string deploynet = testproto.deploy_net();

	FLAGS_minloglevel=1;
	NetParameter net_param;
	this->net_->ToProto(&net_param);
	Net<float> caffe_net(deploynet, caffe::TEST);
  	caffe_net.CopyTrainedLayersFrom(net_param);
	FLAGS_minloglevel=0;

	LOG(INFO)<<"open test file:"<<testfile;
	LOG(INFO)<<"Iteration:" << iter_ <<"  TestingResult:Test_Classify"; 
	std::ifstream infile(testfile.c_str());
	string line;
	string splitflag = testproto.source_splitflag();
	string imname;
	float label;
	int datachannels = testproto.data_channels();
	CHECK(datachannels == 1 || datachannels == 3);
	float maxv,maxid;
	float totalcount=0;
	float errorcount=0;
	int width = testproto.width();
	int height = testproto.height();
	while(infile >> line)
	{
		vector<string> strs = str_split(line,splitflag);
		CHECK(strs.size()==2);
		imname = datafolder + strs[0];
		label = str2num(strs[1]);
		cv::Mat img = cv::imread(imname,datachannels>1?CV_LOAD_IMAGE_COLOR:CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat resizeimg;
		cv::resize(img,resizeimg,cv::Size(width,height),cv::INTER_LINEAR);

		vector<Mat> inputs;
		inputs.push_back(resizeimg);
		std::vector<float> out = Get_net_output(caffe_net,inputs,testproto);
		maxv = FLT_MIN;
		maxid = 0;
		for(int k=0;k<out.size();k++)
		{
			if(out[k]>maxv)	
			{
				maxv = out[k];
				maxid = k;
			}
		}
		if(static_cast<int>(maxid) != static_cast<int>(label))
		{
			errorcount+=1;
		}
		totalcount+=1;
		
		
	}
	infile.close();

	float  acc = 0;
	if(totalcount >0)
		acc = 1-errorcount / totalcount;
	LOG(INFO)<<"Classify accuracy: "<<acc;
}
template <typename Dtype>
void Solver<Dtype>::Test_FaceVER(TestProto testproto) {
	std::string testfile = testproto.test_file();
	std::string datafolder = testproto.data_folder();
	std::string deploynet = testproto.deploy_net();
	
	int lmk_num = testproto.key_points_count();
	//AffineImageParameter face_affineparam 	

	FLAGS_minloglevel=1;
	NetParameter net_param;
	this->net_->ToProto(&net_param);
	Net<float> caffe_net(deploynet, caffe::TEST);
  	caffe_net.CopyTrainedLayersFrom(net_param);
	FLAGS_minloglevel=0;
	LOG(INFO)<<"open test file:"<<testfile;
	LOG(INFO)<<"Iteration:" << iter_ <<"  TestingResult:Test_FaceVER"; 
	std::ifstream infile(testfile.c_str());
	bool label;
	string im1,im2;
	std::vector<float> lmk1,lmk2;
	float v;

	vector<pair<float, bool> > pred_results;
	pred_results.clear();

	bool savefeature = false;	
	std::ofstream fout;
	if(testproto.save_feature_filename().size()>0)
	{
		savefeature = true;
		fout.open(testproto.save_feature_filename().c_str());
	}

	int testcount=0;
	while (infile >> label) 
	{	
		lmk1.clear();
		lmk2.clear();
		infile>>im1;
		for(int i=0;i<lmk_num*2;i++)
		{
			infile >> v;
			lmk1.push_back(v);
		}
		infile>>im2;
		for(int i=0;i<lmk_num*2;i++)
		{
			infile >> v;
			lmk2.push_back(v);
		}
		
		string imname1 = datafolder + im1;
		string imname2 = datafolder + im2;
		
		cv::Mat img1 = cv::imread(imname1);
		cv::Mat img2 = cv::imread(imname2);
		
		std::vector<cv::Mat> affineimgs1;
		for(int i=0;i<testproto.affine_image_param_size();i++)
		{
			AffineImageParameter affine_image_param = testproto.affine_image_param(i);
			int h = affine_image_param.image_info().height();
			int w = affine_image_param.image_info().width();
			cv::Mat affineimg(h,w,CV_8UC3);
			GetAffineImage(img1,affineimg,lmk1,affine_image_param);	
			cv::Mat tempimg;
			if(affine_image_param.image_info().is_color())
				tempimg = affineimg;
			else
				cvtColor(affineimg, tempimg, CV_BGR2GRAY);
			affineimgs1.push_back(tempimg);
		}	
		std::vector<float> feature1 = Get_net_output(caffe_net,affineimgs1,testproto);

		std::vector<cv::Mat> affineimgs2;
		for(int i=0;i<testproto.affine_image_param_size();i++)
		{
			AffineImageParameter affine_image_param = testproto.affine_image_param(i);
			int h = affine_image_param.image_info().height();
			int w = affine_image_param.image_info().width();
			cv::Mat affineimg(h,w,CV_8UC3);
			GetAffineImage(img2,affineimg,lmk2,affine_image_param);	
			cv::Mat tempimg;
                        if(affine_image_param.image_info().is_color())
                                tempimg = affineimg;
                        else
                                cvtColor(affineimg, tempimg, CV_BGR2GRAY);

			affineimgs2.push_back(tempimg);
		}	
		std::vector<float> feature2 = Get_net_output(caffe_net,affineimgs2,testproto);

		float dist=0;
		for(int i=0;i< feature1.size();i++)
		{
			dist += (feature1[i] - feature2[i])*(feature1[i] - feature2[i]);
		}
		pair<float, bool> pred;
		pred.first = dist;
		pred.second = label;
		pred_results.push_back(pred);
		testcount +=1;
		
		if(savefeature)
		{
			fout<<label;
			for(int idx = 0;idx < feature1.size();idx++)
			{
				fout<<" ";
				fout<<feature1[idx];
			}
			for(int idx = 0;idx < feature2.size();idx++)
			{
				fout<<" ";
				fout<<feature2[idx];
			}
			fout<<"\n";
		}
		//if(testcount%100==0)
		//	LOG(INFO)<<"test count:"<<testcount;
		//if(testcount>3000)
		//	break;
	
	}
	if(savefeature)
	{
		fout.close();
	}
	infile.close();
	LOG(INFO)<<"Total test pair:"<<testcount;

	/*for(int i=0;i<pred_results.size();i++)
		LOG(INFO)<<" dist:"<<pred_results[i].first<<" label:"<<pred_results[i].second;*/

	pair<float, float> best_acc = GetBestAccuracy(pred_results);
	LOG(INFO)<< "Best accuracy: " << best_acc.first<< ", threshold: " << best_acc.second;

	vector<float> FAR_vecs;
	FAR_vecs.push_back(1);
	FAR_vecs.push_back(0.1);
	FAR_vecs.push_back(0.01);
	FAR_vecs.push_back(0.001);

	vector<vector<float> > tp_fp_rates;
  	LOG(INFO) << "AUC: " << GetROCData(pred_results, tp_fp_rates);
	// 得出每个negative pair的index
	  vector<int> neg_inds;
	  for (int i = 0; i < tp_fp_rates.size(); ++i) {
	    if (!pred_results[i].second) {
	      neg_inds.push_back(i);
	    }
	  }

	  for (int far_i = 0; far_i < FAR_vecs.size(); ++far_i) {
	    const int neg_ind = FAR_vecs[far_i] * 1e-2 * neg_inds.size();
	    if (neg_ind + 1 < neg_inds.size()) {
	      const int ind = neg_inds[neg_ind];
	      LOG(INFO) << "FAR " << FAR_vecs[far_i] << "%("
		  << (tp_fp_rates[ind][3] * 100) << "%), "
		  << "GAR " << (tp_fp_rates[ind][1] * 100) << "%" << std::endl;
	    }
	  }

	return;
}



template <typename Dtype>
void Solver<Dtype>:: Test_FaceRET(TestProto testproto) {
	std::string testfile = testproto.test_file();
	std::string datafolder = testproto.data_folder();
	std::string deploynet = testproto.deploy_net();
	
	int lmk_num = testproto.key_points_count();
	std::vector<int>topn;
	for(int i=0;i<testproto.topn_size();i++)
	{
		int n=testproto.topn(i);
		topn.push_back(n);	
	}
	
	FLAGS_minloglevel=1;
	NetParameter net_param;
	this->net_->ToProto(&net_param);
	Net<float> caffe_net(deploynet, caffe::TEST);
  	caffe_net.CopyTrainedLayersFrom(net_param);
	FLAGS_minloglevel=0;
	LOG(INFO)<<"open test file:"<<testfile;
	LOG(INFO)<<"Iteration:" << iter_ <<"  TestingResult:Test_FaceRET"; 
	std::ifstream infile(testfile.c_str());
	
	string imname;
	std::vector<float>lmk;
	vector<pair<string, vector<float> > > gallery_infos;
	float v=0;
	int testcount=0;

	bool savefeature = false;	
	std::ofstream fout;
	if(testproto.save_feature_filename().size()>0)
	{
		savefeature = true;
		fout.open(testproto.save_feature_filename().c_str());
	}

	while (infile >> imname) 
	{
		lmk.clear();
		for(int i=0;i<lmk_num*2;i++)
		{
			infile >> v;
			lmk.push_back(v);
		}

		string imagename = datafolder + imname;
		
		
		cv::Mat image = cv::imread(imagename);
		if(image.empty())
		{
			LOG(INFO)<<"can not open image:"<<imagename;
			continue;
		}
		
		
		
		std::vector<cv::Mat> affineimgs;
		for(int i=0;i<testproto.affine_image_param_size();i++)
		{
			AffineImageParameter affine_image_param = testproto.affine_image_param(i);
			int h = affine_image_param.image_info().height();
			int w = affine_image_param.image_info().width();
			cv::Mat affineimg(h,w,CV_8UC3);
			GetAffineImage(image,affineimg,lmk,affine_image_param);
			cv::Mat tempimg;
                        if(affine_image_param.image_info().is_color())
                                tempimg = affineimg;
                        else
                                cvtColor(affineimg, tempimg, CV_BGR2GRAY);
	
			affineimgs.push_back(tempimg);
		}	
		std::vector<float> feature = Get_net_output(caffe_net,affineimgs,testproto);

		gallery_infos.push_back(std::make_pair(imname,feature));
		testcount+=1;

		if(savefeature)
		{
			if(savefeature)
			{
				fout<<imname;
				for(int idx = 0;idx < feature.size();idx++)
				{
					fout<<" ";
					fout<<feature[idx];
				}
				fout<<"\n";
			}
		}
		
	}// end while read image

	infile.close();
	if(savefeature)
		fout.close();
	
	LOG(INFO)<<"Total Test Sample:"<<gallery_infos.size();

	float totalsmaplecount = gallery_infos.size();
	float top1acc=0;
	vector<float> topnacc(topn.size(),0);
	vector<float> topnknn(topn.size(),0);
	vector<float> topnhit(topn.size(),0);
	vector<float> topnrecall(topn.size(),0);
	
	
	//mean dist, (total pos neg)
	vector<float> diststatics(3,0);
	vector<float> diststatics_count(3,0);

	float *scoremap = NULL, *SameScore = NULL, *DiffScore = NULL;
	SameScore = (float *)calloc(totalsmaplecount * totalsmaplecount, sizeof(float));
	DiffScore = (float *)calloc(totalsmaplecount * totalsmaplecount, sizeof(float));
	scoremap = (float *)calloc(totalsmaplecount*totalsmaplecount, sizeof(float));
	int Same_Num = 0, Diff_Num = 0;


	for(int i=0;i<gallery_infos.size();i++)
	{
		string curname = gallery_infos[i].first;
		const int split_ind = curname.find_last_of('/');
		CHECK_NE(split_ind, string::npos)<<curname << " has no class information";
		const string curclassname = curname.substr(0, split_ind);

		vector<float> curfea = gallery_infos[i].second;

		//vector<pair<float, bool> > pred_results;
		vector<pair<float, string> > pred_results;

		float total_cur_sample_count = 0;

		for(int j=0;j<gallery_infos.size();j++)
		{
			
			

			string comparename = gallery_infos[j].first;
			const int split_ind2 = comparename.find_last_of('/');
			CHECK_NE(split_ind2, string::npos)<<comparename << " has no class information";
			const string compareclassname = comparename.substr(0, split_ind2); 
			
			

			
			float dist = GetL2Distance(curfea,gallery_infos[j].second);
			pred_results.push_back(std::make_pair(dist,compareclassname));
			if(compareclassname == curclassname)
				total_cur_sample_count +=1;


			if (i!=j)
			{
				if (curclassname != compareclassname) DiffScore[Diff_Num++] = dist;
				if (curclassname == compareclassname) SameScore[Same_Num++] = dist;
			}

			if(i==0&&j==0)
			{
				for(int did=0;did<diststatics.size();did++)
					diststatics[did]=dist;
			}
			else if(i!=j)
			{
				//mean
				diststatics[0]+=dist;
				diststatics_count[0]+=1;
				if(compareclassname == curclassname)
				{
					diststatics[1]+=dist;
					diststatics_count[1]+=1;
				}
				else
				{
					diststatics[2]+=dist;
					diststatics_count[2]+=1;
				}

			}
				
		}

		total_cur_sample_count -=1;

	


		std::stable_sort(pred_results.begin(), pred_results.end());
		
	

				
		if(pred_results[1].second == curclassname)
			top1acc+=1;

		if(totalsmaplecount<topn.size())
			continue;

		for(int k=0;k<topn.size();k++)
		{
			float n = topn[k];
			if(totalsmaplecount <= n)
				continue;
			float correctcount = 0;
			int start_id=1;
			vector<pair<string, float> > statics;
			statics.clear();

			for(int id=start_id;id<start_id+n;id++)
			{
				if(pred_results[id].second == curclassname)
					correctcount+=1;

				bool found = false;
				for(int subid=0;subid<statics.size();subid++)
				{
					if(statics[subid].first == pred_results[id].second)
					{
						found=true;
						statics[subid].second +=1;
						break;
					}
				}
				if(!found)
				{
					statics.push_back(std::make_pair(pred_results[id].second,1));
				}


			}
			
			if(total_cur_sample_count > 0)
				topnrecall[k] += (correctcount/total_cur_sample_count);

			topnacc[k] += (correctcount/n);
			
			if(correctcount>0)
				topnhit[k]+=1;

			
			int maxv=-1;
			string maxname="";
			for(int id = 0;id<statics.size();id++)
			{
				if(statics[id].second>maxv)
				{
					maxv = statics[id].second;
					maxname = statics[id].first;
				}
			}
			if(maxname == curclassname)
				topnknn[k]+=1;
		}
		
		
	}//end all samples

	std::qsort((float*)DiffScore, Diff_Num, sizeof(DiffScore[0]), cmp);
	float FAR[4] = {0.01, 0.001, 0.0001, 0.00001};
	for (int k = 0; k < 4; k++){
		int cnt = 0;
		int num = FAR[k]*Diff_Num;
		float thr = DiffScore[num];
		for (int i = 0; i < Same_Num; i++){
			if (SameScore[i]<thr) cnt++;
		}
		float GAR = float(cnt)/Same_Num;
	    //printf("thr: %f FAR:%f GAR:%f\n", thr, FAR[k], GAR);
		LOG(INFO)<<"thr:"<<thr<<" FAR:"<<FAR[k]<<" GAR:"<<GAR;
	}

	
	top1acc = top1acc / totalsmaplecount;
	for(int i=0;i<topn.size();i++)
	{
		topnacc[i] = topnacc[i] / totalsmaplecount;
		topnknn[i] = topnknn[i] / totalsmaplecount;
		topnhit[i] = topnhit[i] / totalsmaplecount;
		topnrecall[i] = topnrecall[i] / totalsmaplecount;
	}
	
	LOG(INFO)<<"Top 1 Acc:"<<top1acc;
	for(int i=0;i<topn.size();i++)
	{
		LOG(INFO)<<"Top "<<topn[i]<<" Acc:"<<topnacc[i]<<" Knn:"<<topnknn[i]<<" HitRate:"<<topnhit[i]<<" Recall:"<<topnrecall[i];
		
	}
	
	LOG(INFO)<<"Total mean distance:"<<diststatics[0]/diststatics_count[0]<<" Pos mean distance:"<<diststatics[1]/diststatics_count[1]<<" Neg mean distance:"<<diststatics[2]/diststatics_count[2];
	
	if (SameScore) free(SameScore);
	if (DiffScore) free(DiffScore);
	if (scoremap) free(scoremap);


	return;	

}


template <typename Dtype>
void Solver<Dtype>::TestAll() {

  if(this->param_.has_test_protos())
  {
	LOG(INFO)<<"start test...";
  
	TestProtos testprotos = param_.test_protos();
	//LOG(INFO)<<"test size = "<<testprotos.test_proto_size();
	for(int i=0;i< testprotos.test_proto_size();i++)
	{
		TestProto testproto = testprotos.test_proto(i);
		switch(testproto.test_type())
		{
			case TestProto_TestType_FACEVER:
				Test_FaceVER(testproto);
				break;
			case TestProto_TestType_FACERET:
				Test_FaceRET(testproto);
				break;
			case TestProto_TestType_CLASSIFY:
				Test_Classify(testproto);
				break;
			case TestProto_TestType_SEGMENT:
				Test_Segment(testproto);
				break;
			default:
				LOG(FATAL)<<"unknown test type";
				break;

		}
	}
  }

  /*for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }*/
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
