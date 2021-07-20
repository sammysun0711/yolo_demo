/*
 * http://ffmpeg.org/doxygen/trunk/index.html
 *
 * Main components
 *
 * Format (Container) - a wrapper, providing sync, metadata and muxing for the streams.
 * Stream - a continuous stream (audio or video) of data over time.
 * Codec - defines how data are enCOded (from Frame to Packet)
 *        and DECoded (from Packet to Frame).
 * Packet - are the data (kind of slices of the stream data) to be decoded as raw frames.
 * Frame - a decoded raw frame (to be encoded or filtered).
 */

#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <ocv_common.hpp>
#include <slog.hpp>
#include <yolo.hpp>

#ifdef __cplusplus
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
#endif

#ifdef av_err2str
#undef av_err2str

av_always_inline std::string av_err2string(int errnum) {
    char str[AV_ERROR_MAX_STRING_SIZE];
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#define av_err2str(err) av_err2string(err).c_str()
#endif  // av_err2str

using namespace cv;

// (TODO) Move global variable
int how_many_packets_to_process = 100;
int frame_number = 0;
int frame_base = 6;

// print out the steps and errors
static void logging(const char *fmt, ...);
// decode packets into frames
static int decode_packet(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, cv::Mat &image);
static cv::Mat avframe_to_cvmat(const AVFrame * frame);

int main(int argc, const char *argv[])
{
  double total_latency_time = 0.0;
  logging("Initialize YOLO Model.\n");
  std::string model_path = std::string(argv[1]);
  //YOLOv3 model = YOLOv3();
  YOLOv5 model = YOLOv5();
  model.initialize_model(model_path);
  
  if (argc < 3) {
    printf("You need to specify a input model path and media file.\n");
    printf("Example ./main [input_model_path] [input_media_path].\n");
    return -1;
  }
  
  logging("Initializing all the containers, codecs and protocols.");
  
  // AVFormatContext holds the header information from the format (Container)
  // Allocating memory for this component
  // http://ffmpeg.org/doxgen/trunk/structAVFormatContext.html
  AVFormatContext *pFormatContext = avformat_alloc_context();
  if (!pFormatContext) {
    logging("ERROR could not allocate memory for Format Context");
    return -1;
  }

  logging("Opening the input file (%s) and loading format (container) header", argv[2]);
  // Open the file and read its header. The codecs are not opened.
  // The function arguments are:
  // AVFormatContext (the component we allocated memory for),
  // url (filename),
  // AVInputFormat (if you pass NULL it'll do the auto detect)
  // and AVDictionary (which are options to the demuxer)
  // http://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga31d601155e9035d5b0e7efedc894ee49
  if (avformat_open_input(&pFormatContext, argv[2], NULL, NULL) != 0) {
    logging("ERROR could not open the file");
    return -1;
  }

  // now we have access to some information about our file
  // since we read its header we can say what format (container) it's
  // and some other information related to the format itself.
  logging("Format %s, duration %lld us, bit_rate %lld", pFormatContext->iformat->name, pFormatContext->duration, pFormatContext->bit_rate);

  logging("Finding stream info from format");
  // read Packets from the Format to get stream information
  // this function populates pFormatContext->streams
  // (of size equals to pFormatContext->nb_streams)
  // the arguments are:
  // the AVFormatContext
  // and options contains options for codec corresponding to i-th stream.
  // On return each dictionary will be filled with options that were not found.
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#gad42172e27cddafb81096939783b157bb
  if (avformat_find_stream_info(pFormatContext,  NULL) < 0) {
    logging("ERROR could not get the stream info");
    return -1;
  }

  // the component that knows how to enCOde and DECode the stream
  // it's the codec (audio or video)
  // http://ffmpeg.org/doxygen/trunk/structAVCodec.html
  AVCodec *pCodec = NULL;
  // this component describes the properties of a codec used by the stream i
  // https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html
  AVCodecParameters *pCodecParameters =  NULL;
  int video_stream_index = -1;
  int width, height;
  // loop though all the streams and print its main information
  for (size_t i = 0; i < pFormatContext->nb_streams; i++)
  {
    AVCodecParameters *pLocalCodecParameters =  NULL;
    pLocalCodecParameters = pFormatContext->streams[i]->codecpar;
    logging("AVStream->time_base before open coded %d/%d", pFormatContext->streams[i]->time_base.num, pFormatContext->streams[i]->time_base.den);
    logging("AVStream->r_frame_rate before open coded %d/%d", pFormatContext->streams[i]->r_frame_rate.num, pFormatContext->streams[i]->r_frame_rate.den);
    logging("AVStream->start_time %" PRId64, pFormatContext->streams[i]->start_time);
    logging("AVStream->duration %" PRId64, pFormatContext->streams[i]->duration);

    logging("Finding the proper decoder (CODEC)");

    // AVCodec *pLocalCodec = NULL;

    // finds the registered decoder for a codec ID
    // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga19a0ca553277f019dd5b0fec6e1f9dca
    auto pLocalCodec = avcodec_find_decoder(pLocalCodecParameters->codec_id);

    if (pLocalCodec==NULL) {
      logging("ERROR unsupported codec!");
      // In this example if the codec is not found we just skip it
      continue;
    }

    // when the stream is a video we store its index, codec parameters and codec
    if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO) {
      if (video_stream_index == -1) {
        video_stream_index = i;
        pCodec = pLocalCodec;
        pCodecParameters = pLocalCodecParameters;
      }

      logging("Video Codec: resolution %d x %d", pLocalCodecParameters->width, pLocalCodecParameters->height);
      width = pLocalCodecParameters->width;
      height = pLocalCodecParameters->height;
    } else if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_AUDIO) {
      logging("Audio Codec: %d channels, sample rate %d", pLocalCodecParameters->channels, pLocalCodecParameters->sample_rate);
    }

    // print its name, id and bitrate
    logging("\tCodec %s ID %d bit_rate %lld", pLocalCodec->name, pLocalCodec->id, pLocalCodecParameters->bit_rate);
  }

  if (video_stream_index == -1) {
    logging("File %s does not contain a video stream!", argv[2]);
    return -1;
  }

  // https://ffmpeg.org/doxygen/trunk/structAVCodecContext.html
  AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
  if (!pCodecContext)
  {
    logging("Failed to allocated memory for AVCodecContext");
    return -1;
  }

  // Fill the codec context based on the values from the supplied codec parameters
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#gac7b282f51540ca7a99416a3ba6ee0d16
  if (avcodec_parameters_to_context(pCodecContext, pCodecParameters) < 0)
  {
    logging("Failed to copy codec params to codec context");
    return -1;
  }

  // Initialize the AVCodecContext to use the given AVCodec.
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
  if (avcodec_open2(pCodecContext, pCodec, NULL) < 0)
  {
    logging("Failed to open codec through avcodec_open2");
    return -1;
  }

  // https://ffmpeg.org/doxygen/trunk/structAVFrame.html
  AVFrame *pFrame = av_frame_alloc();
  if (!pFrame)
  {
    logging("Failed to allocated memory for AVFrame");
    return -1;
  }
  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *pPacket = av_packet_alloc();
  if (!pPacket)
  {
    logging("Failed to allocated memory for AVPacket");
    return -1;
  }
  cv::Mat image = cv::Mat(width, height, CV_8UC3);
  int response = 0;
  // fill the Packet with data from the Stream
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga4fdb3084415a82e3810de6ee60e46a61

  while (av_read_frame(pFormatContext, pPacket) >= 0)
  {
    // if it's the video stream
    if (pPacket->stream_index == video_stream_index) {
	// Decoding
        auto t0 = std::chrono::high_resolution_clock::now();
        frame_number++;
	slog::info << " ************************************************* Frame Number " << frame_number << " *************************************************" << slog::endl;
	response = decode_packet(pPacket, pCodecContext, pFrame, image);
	// Skip frame according to frame base
	--how_many_packets_to_process;
	if (frame_number % frame_base != 0){
          auto t1 = std::chrono::high_resolution_clock::now();
          double decode_inferenece_time = std::chrono::duration_cast<ms>(t1 - t0).count();
          slog::info << "[Frame " << frame_number << "] Single Packet Decode + Image Conversion + Inference Time: " << decode_inferenece_time << " ms" << slog::endl;
          total_latency_time += decode_inferenece_time;
          continue;
	}
	if (response>=0){
	  // Inference 
	  auto inference_t0 = std::chrono::high_resolution_clock::now();
          model.inference(image, frame_number);
          auto inference_t1 = std::chrono::high_resolution_clock::now();
          double inference_time = std::chrono::duration_cast<ms>(inference_t1 - inference_t0).count();
          slog::info << "[Frame " << frame_number << "] Inference Time:  " << inference_time << "ms " << slog::endl;
	  cv::imshow("frame", image);
	  cv::waitKey(0);
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	double decode_inferenece_time = std::chrono::duration_cast<ms>(t1 - t0).count();
	slog::info << "[Frame " << frame_number << "] Single Packet Decode + Image Conversion + Inference Time: " << decode_inferenece_time << " ms" << slog::endl;
	total_latency_time += decode_inferenece_time;
	if (response < 0) break;
        // stop it when maximal packet processed
        if (how_many_packets_to_process <= 0) break;
    }
    // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
    av_packet_unref(pPacket);
  }
  
  double mean_latency = total_latency_time / frame_number;
  double mean_throughput = 1.0 / mean_latency * 1000;
  slog::info << "Mean Latency: " << mean_latency << " ms" << slog::endl;
  slog::info << "Mean Throughput:  " << mean_throughput << " FPS" << slog::endl;
  logging("releasing all the resources");
 
  avformat_close_input(&pFormatContext);
  av_packet_free(&pPacket);
  av_frame_free(&pFrame);
  avcodec_free_context(&pCodecContext);
  return 0;
}

static void logging(const char *fmt, ...)
{
    va_list args;
    fprintf( stderr, "LOG: " );
    va_start( args, fmt );
    vfprintf( stderr, fmt, args );
    va_end( args );
    fprintf( stderr, "\n" );
}

static cv::Mat avframe_to_cvmat(const AVFrame * frame)
{
    int width = frame->width;
    int height = frame->height;
    cv::Mat image(height, width, CV_8UC3);
    int cvLinesizes[1];
    cvLinesizes[0] = image.step1();
    SwsContext* conversion = sws_getContext(width, height, (AVPixelFormat) frame->format, width, height, AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cvLinesizes);
    sws_freeContext(conversion);
    return image;
}

static int decode_packet(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, cv::Mat &image)
{
  // Supply raw packet data as input to a decoder
  // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga58bc4bf1e0ac59e27362597e467efff3
  int response = avcodec_send_packet(pCodecContext, pPacket);

  if (response < 0) {
    logging("Error while sending a packet to the decoder: %s", av_err2str(response));
    return response;
  }

  while (response >= 0)
  {
    // Return decoded output data (into a frame) from a decoder
    // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga11e6542c4e66d3028668788a1a74217c
    auto decoding_t0 = std::chrono::high_resolution_clock::now();
    response = avcodec_receive_frame(pCodecContext, pFrame);
    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
      break;
    } else if (response < 0) {
      logging("Error while receiving a frame from the decoder: %s", av_err2str(response));
      return response;
    }
    auto decoding_t1 = std::chrono::high_resolution_clock::now();
    double decoding_time = std::chrono::duration_cast<ms>(decoding_t1 - decoding_t0).count();
    slog::info << "[Frame " << frame_number << "] Decoding Time:  " << decoding_time << " ms" << slog::endl;

    if (response >= 0) {
      if (frame_number % frame_base != 0){
          return 0;
      }

      // Convert avframe (AV_PIX_FMT_YUV420P) to cv::Mat (AV_PIX_FMT_BGR24)
      auto conversion_t0 = std::chrono::high_resolution_clock::now();
      image = avframe_to_cvmat(pFrame);
      auto conversion_t1 = std::chrono::high_resolution_clock::now();
      double conversion_time = std::chrono::duration_cast<ms>(conversion_t1 - conversion_t0).count();
      slog::info << "[Frame " << frame_number << "] Image Conversion Time:  " << conversion_time << " ms " << slog::endl;
    }
      
  }
  return 0;
}
