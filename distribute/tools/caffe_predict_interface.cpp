#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include "time.h"
#include <fstream>
#include <iostream>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

DEFINE_int32( gpu, -1,
	      "Run in GPU mode on given device ID." );
DEFINE_string( solver, "",
	       "The solver definition protocol buffer text file." );
DEFINE_string( model, "",
	       "The model definition protocol buffer text file.." );
DEFINE_string( snapshot, "",
	       "Optional; the snapshot solver state to resume training." );
DEFINE_string( weights, "",
	       "Optional; the pretrained weights to initialize finetuning. "
	       "Cannot be set simultaneously with snapshot." );
DEFINE_int32( iterations, 50,
	      "The number of iterations to run." );
DEFINE_string( testfile, "",
	       "The test text file.." );
DEFINE_string( meanfile, "",
	       "The mean data file.." );
DEFINE_string( outputlayername, "",
	       "The outputlayername .." );
DEFINE_string( savefolder, "",
	       "The savefolder .." );
DEFINE_int32( height, -1,
	      "Run in GPU mode on given device ID." );
DEFINE_int32( width, -1,
	      "Run in GPU mode on given device ID." );
DEFINE_bool( autoscale, false,
	     "Run in GPU mode on given device ID." );
DEFINE_bool( autoresize, false,
	     "Run in GPU mode on given device ID." );
DEFINE_string( testimagename, "",
	       "The test image file.." );
DEFINE_double( datascale, 0.00390625,
	       "The data scale.." );
DEFINE_int32( datachannel, 3,
	      "The data channel.." );
DEFINE_string( saveonefile, "",
	       "save feature .." );


/* A simple registry for caffe commands. */
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction( func ) \
	namespace { \
		class __Registerer_ ## func { \
public:                 /* NOLINT */ \
			__Registerer_ ## func() { \
				g_brew_map[# func] = &func; \
			} \
		}; \
		__Registerer_ ## func g_registerer_ ## func; \
	}

static BrewFunction GetBrewFunction( const caffe::string & name )
{
	if ( g_brew_map.count( name ) )
	{
		return(g_brew_map[name]);
	} else {
		LOG( ERROR ) << "Available caffe actions:";
		for ( BrewMap::iterator it = g_brew_map.begin();
		      it != g_brew_map.end(); ++it )
		{
			LOG( ERROR ) << "\t" << it->first;
		}
		LOG( FATAL ) << "Unknown action: " << name;
		return(NULL); /* not reachable, just to suppress old compiler warnings. */
	}
}


/*
 * caffe commands to call by
 *     caffe <command> <args>
 *
 * To add a command, define a function "int command()" and register it with
 * RegisterBrewFunction(action);
 */

/* Device Query: show diagnostic information for a GPU device. */
int device_query()
{
	CHECK_GT( FLAGS_gpu, -1 ) << "Need a device ID to query.";
	LOG( INFO ) << "Querying device ID = " << FLAGS_gpu;
	caffe::Caffe::SetDevice( FLAGS_gpu );
	caffe::Caffe::DeviceQuery();
	return(0);
}


RegisterBrewFunction( device_query );


/*
 * Load the weights from the specified caffemodel(s) into the train and
 * test nets.
 */
void CopyLayers( caffe::Solver<float>* solver, const std::string & model_list )
{
	std::vector<std::string> model_names;
	boost::split( model_names, model_list, boost::is_any_of( "," ) );
	for ( int i = 0; i < model_names.size(); ++i )
	{
		LOG( INFO ) << "Finetuning from " << model_names[i];
		solver->net()->CopyTrainedLayersFrom( model_names[i] );
		for ( int j = 0; j < solver->test_nets().size(); ++j )
		{
			solver->test_nets()[j]->CopyTrainedLayersFrom( model_names[i] );
		}
	}
}


std::vector<std::string> str_split( std::string str, std::string pattern )
{
	std::string::size_type		pos;
	std::vector<std::string>	result;
	str += pattern;
	int size = str.size();

	for ( int i = 0; i < size; i++ )
	{
		pos = str.find( pattern, i );
		if ( pos < size )
		{
			std::string s = str.substr( i, pos - i );
			result.push_back( s );
			i = pos + pattern.size() - 1;
		}
	}
	return(result);
}


cv::Mat get_outputmap( const vector<vector<Blob<float>*> > & top_vecs, int outidx, bool auto_scale = false )
{
	vector<Blob<float>*>	outblobs	= top_vecs[outidx];
	const float		* outdata	= outblobs[0]->cpu_data();
	int			count		= outblobs[0]->count();
	int			outheight	= outblobs[0]->height();
	int			outwidth	= outblobs[0]->width();
	int			channels	= outblobs[0]->channels();
	int			spacedim	= outheight * outwidth;
	cv::Mat			result		= cv::Mat( outheight, outwidth, CV_8UC1 );

	float	maxv	= -FLT_MAX;
	int	maxid	= 0;
	float	v	= 0;

	int scale_rate = 1;
	if ( auto_scale )
	{
		scale_rate = 255 / (channels - 1);
	}

	for ( int h = 0; h < outheight; h++ )
	{
		/* unsigned char * pdata = result.ptr<unsigned char>(h); */
		for ( int w = 0; w < outwidth; w++ )
		{
			for ( int c = 0; c < channels; c++ )
			{
				v = outdata[c * spacedim + h * outwidth + w];
				if ( v > maxv )
				{
					maxv	= v;
					maxid	= c;
				}
			}
			if ( auto_scale )
			{
				maxid = maxid * scale_rate;
			}
			result.at<unsigned char>( h, w )	= (unsigned char) (maxid);
			maxv					= -FLT_MAX;
			maxid					= 0;
		}
	}
	return(result);
}


void get_caffe_inputdata( cv::Mat img, float * inputdata, std::vector< float > meandata, float data_scale, bool color = true )
{
	int topindex = 0;
	if ( color )
	{
		for ( int c = 0; c < img.channels(); ++c )
		{
			for ( int h = 0; h < img.rows; ++h )
			{
				for ( int w = 0; w < img.cols; ++w )
				{
					float datum_element = static_cast<float>(static_cast<unsigned char>(img.at<cv::Vec3b>( h, w )[c]) );

					inputdata[topindex] = (datum_element - meandata[topindex]) * data_scale;
					topindex++;
				}
			}
		}
	} else {
		for ( int h = 0; h < img.rows; ++h )
		{
			for ( int w = 0; w < img.cols; ++w )
			{
				float datum_element = static_cast<float>(static_cast<unsigned char>(img.at<uchar>( h, w ) ) );

				inputdata[topindex] = (datum_element - meandata[topindex]) * data_scale;
				topindex++;
			}
		}
	}
}


cv::Mat forwardNet( Net<float> * caffe_net, std::string outputlayername, std::vector< cv::Mat > inputimg, int height, int width, std::vector<float> datamean, float datascale, bool auto_scale, bool auto_resize )
{
	const vector<shared_ptr<Layer<float> > > & layers = caffe_net->layers();

	const vector<vector<Blob<float>*> > &	bottom_vecs		= caffe_net->bottom_vecs();
	const vector<vector<Blob<float>*> > &	top_vecs		= caffe_net->top_vecs();
	const vector<Blob<float>* > &		input_blobs		= caffe_net->input_blobs();
	const vector<vector<bool> > &		bottom_need_backward	= caffe_net->bottom_need_backward();

	Blob<float>	* inputblob	= input_blobs[0];
	float		* inputdata	= inputblob->mutable_cpu_data();

	int	outidx = -1;
	cv::Mat dummyresult;
	int	input_height	= inputimg[0].rows;
	int	input_width	= inputimg[0].cols;

	for ( int i = 0; i < inputimg.size(); i++ )
	{
		if ( !inputimg[i].data )
		{
			LOG( ERROR ) << "Could not open or find file ";
			return(dummyresult);
		}
		cv::Mat cv_img;
		if ( height > 0 && width > 0 )
		{
			cv::resize( inputimg[i], cv_img, cv::Size( width, height ) );
		} else {
			cv_img = inputimg[i];
		}
		get_caffe_inputdata( cv_img, inputdata + i * cv_img.channels() * cv_img.rows * cv_img.cols, datamean, datascale );
	}

	bool findlayer = false;
	for ( int i = 0; i < layers.size(); ++i )
	{
		const caffe::string & layername = layers[i]->layer_param().name();

		layers[i]->Reshape( bottom_vecs[i], top_vecs[i] );
		layers[i]->Forward( bottom_vecs[i], top_vecs[i] );
		if ( outputlayername == layername )
		{
			outidx		= i;
			findlayer	= true;
			break;
		}
	}

	if ( !findlayer )
		return(dummyresult);


	cv::Mat result = get_outputmap( top_vecs, outidx, auto_scale );
	if ( auto_resize )
	{
		cv::resize( result, result, cv::Size( input_width, input_height ), 0, 0, cv::INTER_NEAREST );
	}
	return(result);
}


/* ////////////////////////////////////////// */
class Predict_interface
{
public:

	std::string		outputlayername_;
	shared_ptr<Net<float> > net_;
	int			gpuid_;
	std::string		modelfile_;
	std::string		deployfile_;
	int			inputheight_;
	int			inputwidth_;
	std::string		meanfile_;
	float			datascale_;
	int			inputchannel_;
	std::vector<float >	datamean_;
	bool			autoscale_;
	bool			autoresize_;


	Predict_interface()
	{
	};
	~Predict_interface()
	{
	};
	bool init( std::string modelfile,
		   std::string deployfile,
		   std::string outputlayername,
		   std::string meanfile,
		   int gpuid,
		   int intputheight,
		   int inputwidth,
		   float datascale,
		   int inputchannel,
		   bool autoscale,
		   bool autoresize )
	{
		modelfile_		= modelfile;
		gpuid_			= gpuid;
		deployfile_		= deployfile;
		outputlayername_	= outputlayername;
		meanfile_		= meanfile;
		inputheight_		= intputheight;
		inputwidth_		= inputwidth;
		datascale_		= datascale;
		inputchannel_		= inputchannel;
		autoscale_		= autoscale;
		autoresize_		= autoresize;
		net_.reset( new Net<float>( deployfile, caffe::TEST ) );
		net_->CopyTrainedLayersFrom( modelfile_ );
		if ( gpuid_ >= 0 )
		{
			LOG( INFO ) << "Use GPU with device ID " << gpuid_;
			Caffe::SetDevice( gpuid_ );
			Caffe::set_mode( Caffe::GPU );
		} else {
			LOG( INFO ) << "Use CPU.";
			Caffe::set_mode( Caffe::CPU );
		}

		int datasize = inputheight_ * inputwidth_ * inputchannel_;

		datamean_.clear();

		if ( meanfile_.size() > 1 )
		{
			LOG( INFO ) << "load data mean from " << meanfile_.c_str();

			std::ifstream	infile( meanfile_.c_str() );
			float		v;
			int		count = 0;

			while ( infile >> v && count < datasize )
			{
				datamean_.push_back( v );
				count += 1;
			}
			infile.close();
		}else {
			datamean_.resize( datasize, 0 );
		}
		CHECK( datamean_.size() == datasize ) << datamean_.size() << " vs " << datasize;
		LOG( INFO ) << "net init over!";
	}


	cv::Mat predict( std::vector<cv::Mat> imgs )
	{
		return(forwardNet( net_.get(), outputlayername_, imgs, inputheight_, inputwidth_, datamean_, datascale_, autoscale_, autoresize_ ) );
	}
};


int main( int argc, char** argv )
{
	std::string root = "./";

	cv::Mat			img = cv::imread( "1.jpg", CV_LOAD_IMAGE_COLOR );
	std::vector<cv::Mat>	inputimgs;
	inputimgs.push_back( img );


	Predict_interface net;


	net.init( root + "models/roadparsing_iter_160000.caffemodel",
		  root + "roadparsing_deploy.prototxt",
		  "deconv8",
		  root + "datamean",
		  0,
		  100,
		  100,
		  0.00390625,
		  3,
		  1,
		  1 );


	cv::Mat result = net.predict( inputimgs );
	if ( result.data )
	{
		cv::imwrite( "result.png", result );
	}
}



