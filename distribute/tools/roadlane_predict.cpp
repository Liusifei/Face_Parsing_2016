#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include "time.h"
#include "caffe/util/io.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>


using namespace std;
using namespace cv;

using	caffe::Blob;
using	caffe::Caffe;
using	caffe::Net;
using	caffe::Layer;
using	caffe::shared_ptr;
using	caffe::Timer;
using	caffe::vector;


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
DEFINE_bool( useipm, false,
	     "useipm or not." );
DEFINE_bool( draworg, false,
	     "draw org image or not." );


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


void get_caffe_inputdata( cv::Mat img, float * inputdata, float * meandata, float data_scale, bool color = true )
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
					float datum_element = static_cast<float>( static_cast<unsigned char>( img.at<cv::Vec3b>( h, w )[c] ) );

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
				float datum_element = static_cast<float>( static_cast<char>( img.at<uchar>( h, w ) ) );
				inputdata[topindex] = (datum_element - meandata[topindex]) * data_scale;
				topindex++;
			}
		}
	}
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


cv::Mat forwardNet( Net<float> &caffe_net, std::string outputlayername, cv::Mat inputimg, int height, int width, float * datamean, float datascale, bool auto_scale = false, bool auto_resize = false )
{
	int	outidx = -1;
	cv::Mat dummyresult;
	int	input_height	= inputimg.rows;
	int	input_width	= inputimg.cols;
	cv::Mat cv_img_origin	= inputimg;

	cv::Mat cv_img;
	if ( !cv_img_origin.data )
	{
		LOG( ERROR ) << "Could not open or find file ";
		return(dummyresult);
	}
	if ( height > 0 && width > 0 )
	{
		cv::resize( cv_img_origin, cv_img, cv::Size( width, height ) );
	} else {
		cv_img = cv_img_origin;
	}
	const vector<shared_ptr<Layer<float> > > &	layers			= caffe_net.layers();
	const vector<vector<Blob<float>*> > &		bottom_vecs		= caffe_net.bottom_vecs();
	const vector<vector<Blob<float>*> > &		top_vecs		= caffe_net.top_vecs();
	const vector<vector<bool> > &			bottom_need_backward	= caffe_net.bottom_need_backward();


	Blob<float>	* inputblob	= bottom_vecs[0][0];
	float		* inputdata	= inputblob->mutable_cpu_data();
	get_caffe_inputdata( cv_img, inputdata, datamean, datascale );
	/* ///-------------------  fp --------------------------/// */
	for ( int i = 0; i < layers.size(); ++i )
	{
		const caffe::string & layername = layers[i]->layer_param().name();
		layers[i]->Reshape( bottom_vecs[i], top_vecs[i] );
		layers[i]->Forward( bottom_vecs[i], top_vecs[i] );
		if ( outputlayername == layername )
		{
			outidx = i;
			break;
		}
	}

	/* ///-------------------  fp  --------------------------/// */

	cv::Mat result = get_outputmap( top_vecs, outidx, auto_scale );
	if ( auto_resize )
	{
		cv::resize( result, result, cv::Size( input_width, input_height ), 0, 0, cv::INTER_NEAREST );
	}
	return(result);
}


cv::Mat forwardNet( Net<float> &caffe_net, std::string outputlayername, std::vector< cv::Mat > inputimg, int height, int width, float * datamean, float datascale, bool auto_scale = false, bool auto_resize = false )
{
	const vector<shared_ptr<Layer<float> > > &	layers			= caffe_net.layers();
	const vector<vector<Blob<float>*> > &		bottom_vecs		= caffe_net.bottom_vecs();
	const vector<vector<Blob<float>*> > &		top_vecs		= caffe_net.top_vecs();
	const vector<vector<bool> > &			bottom_need_backward	= caffe_net.bottom_need_backward();
	Blob<float>					* inputblob		= bottom_vecs[0][0];
	float						* inputdata		= inputblob->mutable_cpu_data();

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

	/* ///-------------------  fp --------------------------/// */
	for ( int i = 0; i < layers.size(); ++i )
	{
		const caffe::string & layername = layers[i]->layer_param().name();
		layers[i]->Reshape( bottom_vecs[i], top_vecs[i] );
		layers[i]->Forward( bottom_vecs[i], top_vecs[i] );
		if ( outputlayername == layername )
		{
			outidx = i;
			break;
		}
	}

	/* ///-------------------  fp  --------------------------/// */

	cv::Mat result = get_outputmap( top_vecs, outidx, auto_scale );
	if ( auto_resize )
	{
		cv::resize( result, result, cv::Size( input_width, input_height ), 0, 0, cv::INTER_NEAREST );
	}
	return(result);
}


vector<Blob<float>*> extractNetfeature( Net<float> &caffe_net, std::string outputlayername, cv::Mat inputimg, int height, int width, float * datamean, float datascale )
{
	int outidx = -1;

	int	input_height	= inputimg.rows;
	int	input_width	= inputimg.cols;
	cv::Mat cv_img_origin	= inputimg;

	vector<Blob<float>*>	dummy;
	cv::Mat			cv_img;
	if ( !cv_img_origin.data )
	{
		LOG( ERROR ) << "Could not open or find file ";
		return(dummy);
	}

	if ( height > 0 && width > 0 )
	{
		cv::resize( cv_img_origin, cv_img, cv::Size( width, height ) );
	} else {
		cv_img = cv_img_origin;
	}

	const vector<shared_ptr<Layer<float> > > &	layers			= caffe_net.layers();
	const vector<vector<Blob<float>*> > &		bottom_vecs		= caffe_net.bottom_vecs();
	const vector<vector<Blob<float>*> > &		top_vecs		= caffe_net.top_vecs();
	const vector<vector<bool> > &			bottom_need_backward	= caffe_net.bottom_need_backward();
	const vector<Blob<float>* > &			input_blobs		= caffe_net.input_blobs();


	Blob<float>	* inputblob	= input_blobs[0];
	float		* inputdata	= inputblob->mutable_cpu_data();

	get_caffe_inputdata( cv_img, inputdata, datamean, datascale );
	if ( outputlayername == "data" )
	{
		return(bottom_vecs[0]);
	}
	/* ///-------------------  fp --------------------------/// */
	for ( int i = 0; i < layers.size(); ++i )
	{
		
		const caffe::string & layername = layers[i]->layer_param().name();
		layers[i]->Reshape( bottom_vecs[i], top_vecs[i] );
		layers[i]->Forward( bottom_vecs[i], top_vecs[i] );
		if ( outputlayername == layername )
		{
			outidx = i;
			break;
		}
	}
	if ( outidx < 0 )
	{
		LOG( INFO ) << "do not find layer: " << outputlayername;
		return(dummy);
	}

	/* ///-------------------  fp  --------------------------/// */
	vector<Blob<float>*> outblobs = top_vecs[outidx];
	return(outblobs);
}


/* /////////////////////////////////////////////// */


void loadPredictedResult( string filename, float* data )
{
	fstream fin;
	fin.open( filename.c_str(), ios::in );
	int index = 0;
	if ( fin )
	{
		while ( fin >> data[index] )
			index++;
	}else
		cout << "Result file load failed!!!!!!!!!!" << endl;

	cout << "Result data size is " << index << endl;
}


void computePixelProb( float* pBack, float* pFront, float *pA, float* pB, float* pC, uint* pPixelProb, int width, int height )
{
	for ( int h = 0; h < height; ++h )
	{
		for ( int w = 0; w < width; ++w )
		{
			int	index	= h * width + w;
			float	A	= pA[index];
			float	B	= pB[index];
			float	C	= pC[index];
			float	a	= A;
			float	b	= B - 2 * A * h;
			float	c	= -1 * A * h * h - B * h + C + w;
			float	back	= pBack[index];
			float	front	= pFront[index];
			if ( front > back )
			{
				for ( int y = 0; y < height; ++y )
				{
					int x = static_cast<int>( a * y * y + b * y + c + 0.5f );
					if ( x > -1 && x < width )
					{
						int coord = y * width + x;
						pPixelProb[coord]++;
					}
				}
			}
		}
	}
}


uint getMaxPixelProb( uint *pProb, int size )
{
	uint max_value = 0;
	for ( int i = 0; i < size; ++i )
	{
		max_value = pProb[i] > max_value ? pProb[i] : max_value;
	}
	return(max_value);
}


void computePixelProbImage( uint* pProb, IplImage* img, int image_width, int image_height )
{
	uchar* pImg = (uchar *) img->imageData;
	cvZero( img );

	/* get max value in pProb */
	int	len		= image_width * image_height;
	uint	max_value	= getMaxPixelProb( pProb, len );
	/* cout << "max value: " << max_value << endl; */
	for ( int h = 0; h < image_height; ++h )
	{
		for ( int w = 0; w < image_width; ++w )
		{
			int	index	= h * image_width + w;
			float	prob	= (float) (pProb[index]) / (float) (max_value);
			uchar	pixel	= static_cast<uchar>( prob * 255 );
			pImg[index] = pixel;
		}
	}
}


int otsuThreshold( IplImage *frame )
{
	int	width		= frame->width;
	int	height		= frame->height;
	int	GrayScale	= 256;
	int	pixelCount[GrayScale];
	float	pixelPro[GrayScale];
	int	i, j, pixelSum = width * height, threshold = 0;
	uchar	* data = (uchar *) frame->imageData;
	for ( i = 0; i < GrayScale; i++ )
	{
		pixelCount[i]	= 0;
		pixelPro[i]	= 0;
	}
	for ( i = 0; i < height; i++ )
	{
		for ( j = 0; j < width; j++ )
		{
			pixelCount[(int) data[i * width + j]]++;
		}
	}
	for ( i = 0; i < GrayScale; i++ )
	{
		pixelPro[i] = (float) pixelCount[i] / pixelSum;
	}
	float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
	for ( i = 0; i < GrayScale; i++ )
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for ( j = 0; j < GrayScale; j++ )
		{
			if ( j <= i )
			{
				w0	+= pixelPro[j];
				u0tmp	+= j * pixelPro[j];
			}else  {
				w1	+= pixelPro[j];
				u1tmp	+= j * pixelPro[j];
			}
		}
		u0		= u0tmp / w0;
		u1		= u1tmp / w1;
		u		= u0tmp + u1tmp;
		deltaTmp	= w0 * pow( (u0 - u), 2 ) + w1 * pow( (u1 - u), 2 );


		if ( deltaTmp > deltaMax )
		{
			deltaMax	= deltaTmp;
			threshold	= i;
		}
	}
	return(threshold);
}


void  getContour( IplImage* img_src, IplImage* img_dst, double minarea, vector<IplImage*> & vContours )
{
/* LOG(INFO)<<"in getcontour"<<endl; */
	CvMemStorage	* storage		= cvCreateMemStorage();
	CvSeq		* contour		= 0;
	int		contour_num		= cvFindContours( img_src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint( 0, 0 ) );
	CvSeq		* contour_head		= contour;
	CvSeq		* current_contour	= contour;
	int		count			= 0;
	while ( current_contour != 0 )
	{
		CvSeq	* next	= current_contour->h_next;
		double	area	= fabs( cvContourArea( current_contour ) );
		if ( area < minarea )
		{
			if ( current_contour != contour_head )
			{
				CvSeq* pre = current_contour->h_prev;
				pre->h_next = next;
				cvClearSeq( current_contour );
			}else  {
				contour_head = next;
				cvClearSeq( current_contour );
			}
		}else  {
			CvSeq* tmpseq = cvCloneSeq( current_contour );
			tmpseq->h_prev	= 0;
			tmpseq->h_next	= 0;
			IplImage* tmpimg = cvCreateImage( cvGetSize( img_src ), 8, 1 );
			/* LOG(INFO)<<"before drawcontours"<<endl; */
			cvDrawContours( tmpimg, tmpseq, CV_RGB( 255, 255, 255 ), CV_RGB( 255, 255, 255 ), 2, CV_FILLED, 8, cvPoint( 0, 0 ) );
			/* LOG(INFO)<<"after drawcontours"<<endl; */
			vContours.push_back( tmpimg );
		}
		current_contour = next;
	}
	cvZero( img_dst );
/*
 * LOG(INFO)<<"before drawcontours2"<<endl;
 *   cvDrawContours(img_dst, contour_head,  CV_RGB(255,255,255), CV_RGB(255, 255, 255),  2, CV_FILLED, 8, cvPoint(0,0));
 * LOG(INFO)<<"after drawcontours2"<<endl;
 */
}


vector<float> getMaxScoreLines( vector<IplImage*> vImgContours, uint* pPixelProb, float* pBack, float* pFront, float* pA, float* pB, float* pC, int width, int height )
{
	int						contour_num = vImgContours.size();
	vector<pair<vector<uint>, vector<CvPoint> > >	all;
	all.resize( contour_num );
	for ( int h = 0; h < height; ++h )
	{
		for ( int w = 0; w < width; ++w )
		{
			/* calculate params */
			int	index		= h * width + w;
			float	A		= pA[index];
			float	B		= pB[index];
			float	C		= pC[index];
			float	a		= A;
			float	b		= B - 2 * A * h;
			float	c		= -1 * A * h * h - B * h + C + w;
			float	back		= pBack[index];
			float	front		= pFront[index];
			CvPoint point_index	= cvPoint( w, h );
			if ( front > back )
			{
				vector<uint> line_in_each_contour_score( contour_num, 0 );


				vector<uint> line_in_each_contour_count( contour_num, 0 );


				for ( int y = 0; y < height; ++y )
				{
					/* calculate x */
					int x = static_cast<int>( a * y * y + b * y + c + 0.5f );
					if ( x > -1 && x < width )
					{
						int cur_index = y * width + x;
						for ( int i = 0; i < contour_num; ++i )
						{
							uchar	* pImgContourData	= (uchar *) vImgContours[i]->imageData;
							uint	tmp			= (uint) (pImgContourData[cur_index] / 255);
							/* line_in_each_contour_score[i] += ((uint)(pImgContourData[cur_index] / 255) * pPixelProb[cur_index]); */
							line_in_each_contour_score[i]	+= (pPixelProb[cur_index]);
							line_in_each_contour_count[i]	+= ( (uint) (pImgContourData[cur_index]) );
						}
					}
				}
				vector<uint>::iterator	it			= max_element( line_in_each_contour_count.begin(), line_in_each_contour_count.end() );
				int			in_which_contour	= distance( line_in_each_contour_count.begin(), it );
				all[in_which_contour].first.push_back( line_in_each_contour_score[in_which_contour] );
				all[in_which_contour].second.push_back( point_index );
			}       /* end if front > back */
		}               /* end for w */
	}                       /* end for h */
	  /* cout << "size:" << all[0].first.size() << endl; */
	vector<CvPoint> max_lines;
	max_lines.resize( 0 );
	for ( int i = 0; i < contour_num; ++i )
	{
		if ( all[i].first.size() > 0 )
		{
			vector<uint>::iterator	it	= max_element( all[i].first.begin(), all[i].first.end() );
			int			index	= distance( all[i].first.begin(), it );
			max_lines.push_back( all[i].second[index] );
		}
	}

	vector<float> result;
	for ( int i = 0; i < max_lines.size(); ++i )
	{
		int	w	= max_lines[i].x;
		int	h	= max_lines[i].y;
		int	index	= h * width + w;
		float	A	= pA[index];
		float	B	= pB[index];
		float	C	= pC[index];
		float	a	= A;
		float	b	= B - 2 * A * h;
		float	c	= -1 * A * h * h - B * h + C + w;
		result.push_back( a );
		result.push_back( b );
		result.push_back( c );
	}
	return(result);
}


void DrawResult( string src_file, string img_file, vector<float> & pos, int image_width, int image_height )
{
	string	rst_file	= img_file + ".rst.jpg";
	IplImage* img		= cvLoadImage( src_file.c_str(), CV_LOAD_IMAGE_COLOR );
	for ( int i = 0; i < pos.size() / 3; ++i )
	{
		float	a	= pos[i * 3 + 0];
		float	b	= pos[i * 3 + 1];
		float	c	= pos[i * 3 + 2];
		for ( int y = 0; y < image_height; ++y )
		{
			int x = static_cast<int>( a * y * y + b * y + c + 0.5f );
			if ( x > -1 && x < image_width )
			{
				cvCircle( img, cvPoint( x, y ), 1, CV_RGB( 255, 0, 0 ), 1 );
			}
		}
	}
	cvSaveImage( rst_file.c_str(), img );
}


vector<float> getPredictedLanes( float* pRstData, int image_width, int image_height )
{
	int	offset		= image_width * image_height;
	float	* pBackConf	= pRstData + 0 * offset;
	float	* pFrontConf	= pRstData + 1 * offset;
	float	* pA		= pRstData + 2 * offset;
	float	* pB		= pRstData + 3 * offset;
	float	* pC		= pRstData + 4 * offset;

	/* 2 */
	uint	* pPixelProb	= new uint[image_width * image_height];
	IplImage* img_prob	= cvCreateImage( cvSize( image_width, image_height ), 8, 1 );
	memset( pPixelProb, 0, image_width * image_height * sizeof(uint) );
	computePixelProb( pBackConf, pFrontConf, pA, pB, pC, pPixelProb, image_width, image_height );
	computePixelProbImage( pPixelProb, img_prob, image_width, image_height );

	/* 3 threshold */
	IplImage* img_thre	= cvCreateImage( cvGetSize( img_prob ), 8, 1 );
	int	thre		= otsuThreshold( img_prob );
	cvThreshold( img_prob, img_thre, thre, 255, CV_THRESH_BINARY );

	/* 4 dilate */
	IplImage* img_dilate = cvCreateImage( cvGetSize( img_thre ), 8, 1 );
	/* IplConvKernel* dilateMode = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_ELLIPSE); */
	IplConvKernel* dilateMode = cvCreateStructuringElementEx( 5, 5, 3, 3, CV_SHAPE_RECT );
	cvDilate( img_thre, img_dilate, dilateMode, 1 );
	/* cvDilate(img_thre, img_dilate, 0, 1); */


	/*
	 * cv::Mat temp1(img_thre);
	 * cv::Mat temp2(img_dilate);
	 * cv::imshow("temp1",temp1);
	 * cv::imshow("temp2",temp2);
	 */


	/* 5 find contour */
	IplImage		* img_contour = cvCreateImage( cvGetSize( img_dilate ), 8, 1 );
	vector<IplImage*>	vEachContour;
	getContour( img_dilate, img_contour, 200, vEachContour );
/*
 * cv::Mat temp3(img_contour);
 * cv::imshow("temp3",temp3);
 */

	vector<float> result = getMaxScoreLines( vEachContour, pPixelProb, pBackConf, pFrontConf, pA, pB, pC, image_width, image_height );
	/*
	 * cv::waitKey(-1);
	 * cv::destroyAllWindows();
	 * release resource
	 */
	delete[] pPixelProb;
	cvReleaseImage( &img_prob );
	cvReleaseImage( &img_thre );
	cvReleaseImage( &img_dilate );
	cvReleaseImage( &img_contour );
	for ( int i = 0; i < vEachContour.size(); ++i )
	{
		cvReleaseImage( &vEachContour[i] );
	}
	return(result);
}

bool laneconfilt(std::vector<float> lane1,std::vector<float> lane2,int imh, int imw)
{
    float a1 ,b1,c1,a2,b2,c2;
	
	a1=lane1[2];
	b1=lane1[3];
	c1=lane1[4];
	a2=lane2[2];
	b2=lane2[3];
	c2=lane2[4];
	float x1,x2;
	for(int y=0;y<imh;y++)
	{
		x1=a1*y*y + b1*y +c1;
		x2=a2*y*y + b2*y +c2;
		if(x1>=0 && x1<imw && x2>=0 && x2<imw)
		{
	//LOG(INFO)<<"x1:"<<x1<<" x2:"<<x2;
			if(std::fabs(x1-x2)<2)
				return true;
		}
	}

    
    return  false;
}
int my_ascending_cmp(std::vector<float>  p1,std::vector<float>   p2)
{
	return p1[5] < p2[5];
}
int my_descending_cmp(std::vector<float>  p1,std::vector<float>   p2)
{
	return p1[5] > p2[5];
}
std::vector< std::vector<float>  > getPredictedLanes_new( float* outdata, int outheight,int outwidth )
{
	std::vector< std::vector<float>  > result;

	int	offset		= outwidth * outheight;
	float	* pBackConf	= outdata + 0 * offset;
	float	* pFrontConf	= outdata + 1 * offset;
	float	* pA		= outdata + 2 * offset;
	float	* pB		= outdata + 3 * offset;
	float	* pC		= outdata + 4 * offset;

	cv::Mat segmap=cv::Mat::zeros(outheight,outwidth,CV_8UC1 );
	cv::Mat confmap=cv::Mat::zeros(outheight,outwidth,CV_32FC1 );
	std::vector<cv::Mat > lane_region;

	std::vector< std::vector<float>  > lanes;

	float a,b,c,A,B,C;
	float dx,dy;
	int x,y;

	for(int rr=0;rr<outheight;rr++)
	{
		for(int cc=0;cc<outwidth;cc++)
		{
			if(pFrontConf[rr*outwidth + cc] > pBackConf[rr*outwidth + cc])
			{
				segmap.at<unsigned char>(rr,cc)=255;

				a = pA[rr*outwidth + cc];
				b = pB[rr*outwidth + cc];
				c = pC[rr*outwidth + cc];
				dx=float(cc);
				dy=float(rr);
				A = a;
				B = b - 2 * a * dy;
				C = c + dx + a * dy * dy - b * dy;
/*
				for(int row=0;row<outheight;row++)
				{
					float tt = A*row*row + B*row +C;
					int col=int(tt);
					if(col>=0 && col<outwidth)
					{
						confmap.at<float>(row,col) = confmap.at<float>(row,col) + 1.0 + 1 * row / outheight;
					}
				}
*/
				std::vector<float> l;
				l.push_back(float(cc));
				l.push_back(float(rr));

				
				l.push_back(A);
				l.push_back(B);
				l.push_back(C);
				lanes.push_back(l);
			}

		}
	}

	vector<vector<Point> > contours; 
     	cv::findContours(segmap,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); 


	for(int i=0;i<contours.size();i++)
	{
		if(cv::contourArea(contours[i])<100)
			continue;
		cv::Mat temp=cv::Mat::zeros(outheight,outwidth,CV_8UC1 );
		cv::fillConvexPoly(temp,&contours[i][0],contours[i].size(),Scalar(255));
		lane_region.push_back(temp);
		
	}
	if(lane_region.size()==0)
		return result;

	std::vector< std::vector< std::vector<float>  > > grouplanes;
	//map<std::vector< std::vector<float>  >,   int>   grouplanes; 
	grouplanes.resize(lane_region.size());

	for(int i=0;i<lanes.size();i++)
	{
		A=lanes[i][2]; B=lanes[i][3]; C=lanes[i][4];
		for(int row=0;row<outheight;row++)
		{
			float tt = A*row*row + B*row +C;
			int col=int(tt);

			if(col>=0 && col<outwidth)
			{

				confmap.at<float>(row,col) = confmap.at<float>(row,col) + 1.0 + 1 * row / outheight;
			}
		}
		x = lanes[i][0];
		y = lanes[i][1];
		int lane_id=-1;
		for(int k=0;k<lane_region.size();k++)
		{
			if(lane_region[k].at<unsigned char>(y,x)>0)
			{
				lane_id=k;
				break;
			}
		}
		if(lane_id>=0)
		{
			std::vector<float> l;
			l.push_back(float(x));
			l.push_back(float(y));

				
			l.push_back(A);
			l.push_back(B);
			l.push_back(C);
			grouplanes[lane_id].push_back(l);
			
		}

	}

//LOG(INFO)<<"gp size"<<grouplanes.size();

	double minv,maxv;
	minMaxLoc(confmap,&minv,&maxv);


	if(maxv<1e-5)
		maxv=1e-5;
	confmap = confmap * (255/maxv);

	//cv::imwrite("conf.jpg",confmap);
	

	//for(int i=0;i<lane_region.size();i++)
	//	cv::imwrite(caffe::num2str(i)+".jpg",lane_region[i]);

	Mat resultp(segmap.size(),CV_8U,Scalar(0)); 
     	drawContours(resultp,contours,-1,Scalar(255),2); 
	//cv::imwrite("cont.jpg",resultp);


	std::vector< std::vector<float>  > maxlanes;
	
	float maxscore=-1;
	float curscore;
	for(int i=0;i<grouplanes.size();i++)
	{
		int n=grouplanes[i].size();
		maxscore=-1;
		
		std::vector<float> maxlane;
		for(int j=0;j<n;j++)
		{
			std::vector<float> l = grouplanes[i][j];
			A=l[2]; B=l[3]; C=l[4];
			//LOG(INFO)<<"i "<<i<<" A"<<A<<" B"<<B<<" C"<<C;

			curscore = 0;
			for(int row=0;row<outheight;row++)
			{
				float tt = A*row*row + B*row +C;
				int col=int(tt);
				if(col>=0 && col<outwidth)
				{
					curscore += confmap.at<float>(row,col);
				}
			}
			if(curscore > maxscore)
			{
				maxscore = curscore;
				maxlane = l;

			}
			
		}
		if(maxscore > 0)
		{
			maxlane.push_back(maxscore);
			maxlanes.push_back(maxlane);
		}
		
	}

	/*for(int i=0;i<maxlanes.size();i++)
	{
		std::vector<float> l = maxlanes[i];
		A=l[2]; B=l[3]; C=l[4];
		LOG(INFO)<<"after "<<i<<" A:"<<A<<" B:"<<B<<" C:"<<C<<" score:"<<l[5];

	}*/

	sort(maxlanes.begin(),maxlanes.end(),my_descending_cmp);
	/*for(int i=0;i<maxlanes.size();i++)
	{
		std::vector<float> l = maxlanes[i];
		A=l[2]; B=l[3]; C=l[4];
		LOG(INFO)<<"after "<<i<<" A:"<<A<<" B:"<<B<<" C:"<<C<<" score:"<<l[5];

	}*/


	
	std::vector<bool>   conflitflag;
	conflitflag.resize(maxlanes.size(),false);
	
	
	for(int i=0;i<maxlanes.size();i++)
	{
		for(int j=i+1;j<maxlanes.size();j++)
		{
			if(laneconfilt(maxlanes[i],maxlanes[j],outheight,outwidth))
				conflitflag[j]=true;
		}
	}

	

	std::vector< std::vector<float>  > finallanes;
	for(int i=0;i<conflitflag.size();i++)
	{
		if(!conflitflag[i])
			finallanes.push_back(maxlanes[i]);
	}

	result=finallanes;

	/*for(int i=0;i<finallanes.size();i++)
	{
		std::vector<float> l = finallanes[i];
		A=l[2]; B=l[3]; C=l[4];
		//LOG(INFO)<<"i "<<i<<" A"<<A<<" B"<<B<<" C"<<C;
		result.push_back(A);
		result.push_back(B);
		result.push_back(C);
	
	}*/


	
}


cv::Mat draw_lanes( std::vector< std::vector<float> > lanes, cv::Mat image, int lanew=0 ,float top_rate=0)
{
	
	cv::Mat resultimg	= image;
	int	height		= image.rows;
	int	width		= image.cols;
	int	lane_num	= lanes.size();
	float	a, b, c;
	float	row, col;
	int starth= float(height) * top_rate;
	for ( int i = 0; i < lane_num; i++ )
	{
		
		a	= lanes[i][2];
		b	= lanes[i][3];
		c	= lanes[i][4];

		for ( row = starth; row < height; row += 1 )
		{
			col = (row) * (row) * a + (row) * b + c + 0.5f;
			if ( col >= lanew && col < width -lanew)
			{
				
				for(int k=col-lanew;k<col+lanew+1;k++)
				{
					resultimg.at<Vec3b>( int(row), int(k) )[0] = 0;
					resultimg.at<Vec3b>( int(row), int(k) )[1] = 0;
					resultimg.at<Vec3b>( int(row), int(k) )[2] = 255;
				}

			}
		}
	}
	return(resultimg);
}

void Polyfit2( vector<float> x, vector<float> y, vector<float> & param )
	{       /* cout<<"in poly"<<endl; */
		assert( x.size() == y.size() );
		int n = x.size();
		if ( n < 3 )
		{
			cout << "too little pst num:" << n << endl;
			return;
		}
		Mat	X	= Mat( n, 3, CV_32FC1 );
		Mat	Y	= Mat( n, 1, CV_32FC1 );
		for ( int i = 0; i < n; i++ )
		{
			X.at<float>( i, 0 )	= x[i] * x[i];
			X.at<float>( i, 1 )	= x[i];
			X.at<float>( i, 2 )	= 1.;
			Y.at<float>( i, 0 )	= y[i];
		}
		Mat	Xt	= X.t();
		Mat	XtX	= Xt * X;
		Mat	A	= XtX.inv() * Xt * Y;
		param.clear();
		for ( int i = 0; i < 3; i++ )
		{
			param.push_back( A.at<float>( i, 0 ) );
		}
		if ( fabs( param[0] - 0 ) < 1e-9 && fabs( param[1] - 0 ) < 1e-9 && fabs( param[2] - 0 ) < 1e-9 )
			param.clear();
		return;
	}

void translanes(std::vector< std::vector<float> > src , std::vector< std::vector<float> > & dst, cv::Mat M,int height,int width)
{
	
	dst.clear();
	vector<cv::Point2f> points, points_trans; 

	float stage=10;
	float step= float(height) / stage;
	float A,B,C;
	for(int i=0;i<src.size();i++)
	{
		points.clear();
		points_trans.clear();
		A = src[i][2];
		B = src[i][3];
		C = src[i][4];

		for(float row=0;row<height;row+=step)
		{
			float col = A*row*row + B*row + C;
			if(col>0 && col <width)
			{
				points.push_back(cv::Point2f(col,row));
			}
		}
		cv::perspectiveTransform( points, points_trans, M);
		if(points.size()!=points_trans.size())
		{
			LOG(INFO)<<"missing points in trans lane points";
			return;
		}
		/*for(int i=0;i<points.size();i++)
		{
			LOG(INFO)<<" "<<points[i].x<<" "<<points[i].y<<" "<<points_trans[i].x<<" "<<points_trans[i].y;
		}*/
		
		vector<float> x ,y,param;
		for(int i=0;i<points_trans.size();i++)
		{
			x.push_back(points_trans[i].x);
			y.push_back(points_trans[i].y);
		}
		
		Polyfit2( y, x, param );
		if(param.size()!=3)
		{
			LOG(INFO)<<"err in fiting  points in trans lane points";
			return;
		}
		
		std::vector<float> lane;
		lane.push_back(src[i][0]);
		lane.push_back(src[i][1]);
		lane.push_back(param[0]);
		lane.push_back(param[1]);
		lane.push_back(param[2]);
		lane.push_back(src[i][5]);
		dst.push_back(lane);

	}
	return;
}

int test_roadlane()
{
	CHECK_GT( FLAGS_model.size(), 0 ) << "Need a model definition to score.";
	CHECK_GT( FLAGS_weights.size(), 0 ) << "Need model weights to score.";
	CHECK_GT( FLAGS_outputlayername.size(), 0 ) << "Need output layer name.";
	CHECK_GT( FLAGS_savefolder.size(), 0 ) << "Need save folder.";
	CHECK( FLAGS_datachannel == 3 || FLAGS_datachannel == 1 ) << "should be 3 or 1 , get " << FLAGS_datachannel;
	bool		inputcolorimg	= FLAGS_datachannel == 3 ? true : false;
	std::string	outputlayername = FLAGS_outputlayername;
	std::string	savefolder	= FLAGS_savefolder;
	const int	height		= FLAGS_height;
	const int	width		= FLAGS_width;


	/* Set device id and mode */
	if ( FLAGS_gpu >= 0 )
	{
		LOG( INFO ) << "Use GPU with device ID " << FLAGS_gpu;
		Caffe::SetDevice( FLAGS_gpu );
		Caffe::set_mode( Caffe::GPU );
	} else {
		LOG( INFO ) << "Use CPU.";
		Caffe::set_mode( Caffe::CPU );
	}

	int	datasize	= height * width * 3;
	float	* datamean	= new float [datasize];
	if ( FLAGS_meanfile.size() < 1 )
	{
		for ( int tempidx = 0; tempidx < datasize; tempidx++ )
		{
			datamean[tempidx] = 0.0;
		}
	}else  {
		printf( "load data mean from %s\n", FLAGS_meanfile.c_str() );
		FILE	* fp	= fopen( FLAGS_meanfile.c_str(), "r" );
		int	k	= 0;
		for ( k = 0; k < datasize; k++ )
		{
			fscanf( fp, "%f\n", &datamean[k] );
		}
		fclose( fp );
	}

	/* Instantiate the caffe net. */
	Net<float> caffe_net( FLAGS_model, caffe::TEST );


	caffe_net.CopyTrainedLayersFrom( FLAGS_weights );


	vector<Blob<float>* >	bottom_vec;
	vector<int>		test_score_output_id;
	vector<float>		test_score;
	float			loss = 0;

	clock_t start, finish;
	time_t	t_start, t_end;

	vector<std::pair<std::string, std::string> >	pair_lines_;
	const std::string				source = FLAGS_testfile;
	LOG( INFO ) << "Opening file " << source;
	std::ifstream infile( source.c_str() );


	std::string	filename;
	std::string	label = "0";
	while ( infile >> filename )
	{
		pair_lines_.push_back( std::make_pair( filename, label ) );
	}
	int length = pair_lines_.size();
	LOG( INFO ) << "test list length = " << length;

	cv::Mat transM = cv::Mat::eye( 3, 3, CV_32FC1 );
	if ( FLAGS_useipm )
	{
		transM.at<float>( 0, 0 )	= -2.55920258e-02;
		transM.at<float>( 0, 1 )	= -3.62431395e-01;
		transM.at<float>( 0, 2 )	= 1.47505542e+02;
		transM.at<float>( 1, 0 )	= -3.05311332e-15;
		transM.at<float>( 1, 1 )	= -9.08104142e-01;
		transM.at<float>( 1, 2 )	= 3.56059378e+02;
		transM.at<float>( 2, 0 )	= -1.00017650e-17;
		transM.at<float>( 2, 1 )	= -2.76711868e-03;
		transM.at<float>( 2, 2 )	= 1.00000000e+00;
	}
	cv::Mat transM_inv = transM.inv();


	/* main loop */
	for ( int i = 0; i < length; ++i )
	{
		std::vector<std::string>	imagenames	= str_split( pair_lines_[i].first, "||" );
		int				cv_read_flag	= (inputcolorimg ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat				cv_img_origin	= cv::imread( imagenames[0], cv_read_flag );

		cv::Mat inputimg = cv_img_origin;
		cv::Mat ipm;
		if(!cv_img_origin.data)
		{
			LOG(INFO)<<"file error :"<<imagenames[0];
			continue;
		}
		if ( FLAGS_useipm )
		{
			warpPerspective( cv_img_origin, ipm, transM, cv::Size( width, height ), cv::INTER_LINEAR );
			inputimg = ipm;
		}
		cv::resize( inputimg, inputimg, cv::Size( width, height ), cv::INTER_LINEAR );

		double timecost = (double) cv::getTickCount();

		vector<Blob<float>*> outblobs = extractNetfeature( caffe_net, outputlayername, inputimg, height, width, datamean, FLAGS_datascale );

		timecost = ( (double) cv::getTickCount() - timecost) * 1000 / cv::getTickFrequency();
		if ( outblobs.size() < 1 || !outblobs[0]->count() )
		{
			LOG( INFO ) << "can not process " << imagenames[0];
			continue;
		}
		float * outdata = outblobs[0]->mutable_cpu_data();

		std::vector<std::string>	tempname	= str_split( imagenames[0], "/" );
		std::vector<std::string>	tempname2	= str_split( tempname[tempname.size() - 1], "." );
		std::string			savename	= tempname2[0];
		if ( tempname2.size() > 2 )
		{
			for ( int temid = 1; temid < tempname2.size() - 1; temid++ )
			{
				savename	+= '.';
				savename	+= tempname2[temid];
			}
		}


		double timecost_nms = (double) cv::getTickCount();


		std::vector< std::vector<float> > ipmlanes = getPredictedLanes_new( outdata, height,width );
	
		std::vector< std::vector<float> > orglanes;

		translanes(ipmlanes , orglanes,  transM_inv, height, width);
		
		timecost_nms = ( (double) cv::getTickCount() - timecost_nms) * 1000 / cv::getTickFrequency();

		savename = savefolder + savename ;


		cv::Mat cv_img;
		if ( height > 0 && width > 0 && (height != inputimg.rows || width != inputimg.cols) )
		{
			cv::resize( inputimg, cv_img, cv::Size( width, height ) );
		} else {
			cv_img = inputimg;
		}

		cv::Mat ipmresult = draw_lanes( ipmlanes, cv_img );
		cv::imwrite( savename+"_ipm.jpg", ipmresult );

	
		string savetxtname = savename+"_ipm.txt";
		FILE *fp =fopen(savetxtname.c_str(),"w");
		for(int i=0;i<ipmlanes.size();i++)
		{
			fprintf(fp,"%f %f %f %f\n",ipmlanes[i][2],ipmlanes[i][3],ipmlanes[i][4],ipmlanes[i][5]);
		}
		fclose(fp);
	
		if(FLAGS_draworg)
		{

			cv::Mat orgresult = draw_lanes( orglanes, cv_img_origin,5,0.54 );
			cv::imwrite( savename+"_org.jpg", orgresult );
		
		savetxtname = savename+"_org.txt";
		fp =fopen(savetxtname.c_str(),"w");
		for(int i=0;i<orglanes.size();i++)
		{
			fprintf(fp,"%f %f %f %f\n",orglanes[i][2],orglanes[i][3],orglanes[i][4],orglanes[i][5]);
		}
		fclose(fp);

		}

		LOG( INFO )	<< i << " save data " << savename << ", caffe time=" << timecost << " ms, nms time=" << timecost_nms
				<< " ms, total time =" << timecost + timecost_nms<<" ms";
	}

	LOG( INFO ) << "test list length = " << length;
	delete[] datamean;
	return(0);
}


RegisterBrewFunction( test_roadlane );


int main( int argc, char** argv )
{
	/* Print output to stderr (while still logging). */
#ifdef USE_CAFFE_MPI_VERSION
	MPI_Init( &argc, &argv );
#endif
	FLAGS_alsologtostderr = 1;
	/* Usage message. */
	gflags::SetUsageMessage( "command line brew\n"
				 "usage: caffe <command> <args>\n\n"
				 "commands:\n"
				 "  \n"
				 "  test_roadlane  extract and save feature data \n" );
	/* Run tool or show usage. */
	caffe::GlobalInit( &argc, &argv );
	if ( argc == 2 )
	{
		return(GetBrewFunction( caffe::string( argv[1] ) ) () );
	} else {
		gflags::ShowUsageWithFlagsRestrict( argv[0], "tools/caffe" );
	}
}
