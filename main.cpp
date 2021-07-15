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

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <ocv_common.hpp>
#include <slog.hpp>

#ifdef __cplusplus
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#endif
//#include <libavcodec/avcodec.h>
//#include <libavformat/avformat.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

#ifdef av_err2str
#undef av_err2str
#include <string>
av_always_inline std::string av_err2string(int errnum) {
    char str[AV_ERROR_MAX_STRING_SIZE];
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#define av_err2str(err) av_err2string(err).c_str()
#endif  // av_err2str

// (TODO) Move global variable 
std::string FLAGS_m = "./models/yolov3/INT8/yolo-v3-tf.xml";
std::string FLAGS_labels = "./coco.names";
std::string FLAGS_d = "CPU";
bool FLAGS_auto_resize = true;
bool FLAGS_r = true;
double FLAGS_t = 0.5;
double FLAGS_iou_t = 0.4;
bool FLAGS_no_show = false;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
double ocv_render_time = 0;
int how_many_packets_to_process = 100;
int frame_number = 0;

using namespace InferenceEngine;

void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

class YoloParams {
    template <typename T>
    void computeAnchors(const std::vector<T> & mask) {
        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i) {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};

    YoloParams() {}

    YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
        coords = regionYolo->get_num_coords();
        classes = regionYolo->get_num_classes();
        anchors = regionYolo->get_anchors();
        auto mask = regionYolo->get_mask();
        num = mask.size();

        computeAnchors(mask);
    }
};

void ParseYOLOV3Output(const YoloParams &params, const std::string & output_name,
                       const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + output_name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));

    auto side = out_blob_h;
    auto side_square = side * side;
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < params.num; ++n) {
            int obj_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
            int box_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * params.anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * params.anchors[2 * n];
            for (int j = 0; j < params.classes; ++j) {
                int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

class YOLOv3 {
	public:
		void initialize_model();
		void inference(cv::Mat& frame);

	private:
		ExecutableNetwork network;
		InputsDataMap inputInfo;
		OutputsDataMap outputInfo;
		std::map<std::string, YoloParams> yoloParams;
		std::vector<std::string> labels;
};

void YOLOv3::initialize_model(){
	std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating the input arguments ---------------------------------

	std::cout << "Reading input" << std::endl;

  	// --------------------------- 1. Load inference engine -------------------------------------
	std::cout << "Loading Inference Engine" << std::endl;
        Core ie;

        std::cout << "Device info: " << std::endl;
        slog::info << ie.GetVersions(FLAGS_d);
        
        // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
	std::cout << "Loading network files" << std::endl;
        /** Reading network model **/
        auto cnnNetwork = ie.ReadNetwork(FLAGS_m);
        
        /** Reading labels (if specified) **/
        if (!FLAGS_labels.empty()) {
            std::ifstream inputFile(FLAGS_labels);
            std::string label; 
            while (std::getline(inputFile, label)) {
                labels.push_back(label);
            }
            if (labels.empty())
                throw std::logic_error("File empty or not found: " + FLAGS_labels);
        }
        
        /** YOLOV3-based network should have one input and three output **/
        // --------------------------- 3. Configuring input and output -----------------------------------------
        // --------------------------------- Preparing input blobs ---------------------------------------------
	std::cout << "Checking that the inputs are as the demo expects" << std::endl;
	inputInfo = InputsDataMap(cnnNetwork.getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks that have only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;

        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }

        ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
        SizeVector& inSizeVector = inputShapes.begin()->second;
        inSizeVector[0] = 1;  // set batch to 1
        cnnNetwork.reshape(inputShapes); 
	// --------------------------------- Preparing output blobs -------------------------------------------
	std::cout << "Checking that the outputs are as the demo expects" << std::endl;
	outputInfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }

        if (auto ngraphFunction = cnnNetwork.getFunction()) {
            for (const auto op : ngraphFunction->get_ops()) {
                auto outputLayer = outputInfo.find(op->get_friendly_name());
                if (outputLayer != outputInfo.end()) {
                    auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                    if (!regionYolo) {
                        throw std::runtime_error("Invalid output type: " +
                            std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                    }
                    yoloParams[outputLayer->first] = YoloParams(regionYolo);
                }
            }
        }
        else {
            throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
        }
	
        if (!labels.empty() && static_cast<int>(labels.size()) != yoloParams.begin()->second.classes) {
            throw std::runtime_error("The number of labels is different from numbers of model classes");
        }

        // -----------------------------------------------------------------------------------------------------
	// --------------------------- 4. Loading model to the device ------------------------------------------
        std::cout << "Loading model to the device" << std::endl;
	network = ie.LoadNetwork(cnnNetwork, FLAGS_d);
}

void YOLOv3::inference(cv::Mat& frame){
        auto preprocessing_t0 = std::chrono::high_resolution_clock::now();
	// -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Creating infer request -----------------------------------------------
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Doing inference ------------------------------------------------------
	std::cout << "Start inference " << std::endl;

	auto inputName = inputInfo.begin()->first;        	
	auto outputName = outputInfo.begin()->first;
	if (!frame.empty()) {
		FrameToBlob(frame, async_infer_request_curr, inputName);
        } else {
		throw std::logic_error("Failed to get frame from cv::VideoCapture");
	}
	auto preprocessing_t1 = std::chrono::high_resolution_clock::now();
	double preprocessing_time = std::chrono::duration_cast<ms>(preprocessing_t1 - preprocessing_t0).count();
        slog::info << "[Frame " << frame_number << " ] Image Conversion Time:  " << preprocessing_time << " ms " << slog::endl;
        
	auto ir_inference_t0 = std::chrono::high_resolution_clock::now();
	async_infer_request_curr->StartAsync();
        if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
            auto ir_inference_t1 = std::chrono::high_resolution_clock::now();
	    double ir_inference_time = std::chrono::duration_cast<ms>(ir_inference_t1 - ir_inference_t0).count();
            slog::info << "[Frame " << frame_number << " ] IR Inference Time:  " << ir_inference_time << " ms " << slog::endl;
            // ---------------------------Processing output blobs--------------------------------------------------
            // Processing results of the CURRENT request 
	    
	    auto post_processing_t0 = std::chrono::high_resolution_clock::now();

	    const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
            unsigned long resized_im_h = getTensorHeight(inputDesc);
            unsigned long resized_im_w = getTensorWidth(inputDesc);
            std::vector<DetectionObject> objects;
            // Parsing outputs
	    size_t width = frame.cols;
	    size_t height = frame.rows;
            for (auto &output : outputInfo) {
                auto output_name = output.first;
                Blob::Ptr blob = async_infer_request_curr->GetBlob(output_name);
                ParseYOLOV3Output(yoloParams[output_name], output_name, blob, resized_im_h, resized_im_w, height, width, FLAGS_t, objects);
            }
            // Filtering overlapping boxes
            std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
            for (size_t i = 0; i < objects.size(); ++i) {
		//std::cout << "objects #" << i <<  " , confidence: " << objects[i].confidence << std::endl;
                if (objects[i].confidence == 0)
                    continue;
                for (size_t j = i + 1; j < objects.size(); ++j)
                    if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t)
                        objects[j].confidence = 0;
            }
            // Drawing boxes
	    for (auto &object : objects) {
                    if (object.confidence < FLAGS_t)
                        continue;
                    auto label = object.class_id;
                    float confidence = object.confidence;
                    if (FLAGS_r) {
                        std::cout << "[" << label << "] element, prob = " << confidence <<
                                  "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                                  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                    }
                    if (confidence > FLAGS_t) {
                        /** Drawing only objects when >confidence_threshold probability **/
                        std::ostringstream conf;
                        conf << ":" << std::fixed << std::setprecision(3) << confidence;
                        cv::putText(frame,
                                    (!labels.empty() ? labels[label] :std::string("label #") + std::to_string(label)) + conf.str(),
                                    cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    cv::Scalar(0, 0, 255));
                        cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                                      cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));
                    }
             }
	     auto post_processing_t1 = std::chrono::high_resolution_clock::now();
	     double post_processing_time = std::chrono::duration_cast<ms>(post_processing_t1 - post_processing_t0).count();
             slog::info << "[Frame " << frame_number << " ] Post Processing Time:  " << post_processing_time << " ms " << slog::endl;

	}        
    
	std::cout << "Execution successful" << std::endl;
}

// print out the steps and errors
static void logging(const char *fmt, ...);
// decode packets into frames
static int decode_packet(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, YOLOv3 model);
// save a frame into a .pgm file
// static void save_gray_frame(unsigned char *buf, int wrap, int xsize, int ysize, char *filename);
static cv::Mat avframe_to_cvmat(const AVFrame * frame);

int main(int argc, const char *argv[])
{
  double total_inference_time = 0.0; 
  logging("initialize yolo model.\n");
  YOLOv3 model = YOLOv3();
  model.initialize_model();
  if (argc < 2) {
    printf("You need to specify a media file.\n");
    return -1;
  }
  
  logging("initializing all the containers, codecs and protocols.");
  
  // AVFormatContext holds the header information from the format (Container)
  // Allocating memory for this component
  // http://ffmpeg.org/doxgen/trunk/structAVFormatContext.html
  AVFormatContext *pFormatContext = avformat_alloc_context();
  if (!pFormatContext) {
    logging("ERROR could not allocate memory for Format Context");
    return -1;
  }

  logging("opening the input file (%s) and loading format (container) header", argv[1]);
  // Open the file and read its header. The codecs are not opened.
  // The function arguments are:
  // AVFormatContext (the component we allocated memory for),
  // url (filename),
  // AVInputFormat (if you pass NULL it'll do the auto detect)
  // and AVDictionary (which are options to the demuxer)
  // http://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga31d601155e9035d5b0e7efedc894ee49
  if (avformat_open_input(&pFormatContext, argv[1], NULL, NULL) != 0) {
    logging("ERROR could not open the file");
    return -1;
  }

  // now we have access to some information about our file
  // since we read its header we can say what format (container) it's
  // and some other information related to the format itself.
  logging("format %s, duration %lld us, bit_rate %lld", pFormatContext->iformat->name, pFormatContext->duration, pFormatContext->bit_rate);

  logging("finding stream info from format");
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

  // loop though all the streams and print its main information
  for (size_t i = 0; i < pFormatContext->nb_streams; i++)
  {
    AVCodecParameters *pLocalCodecParameters =  NULL;
    pLocalCodecParameters = pFormatContext->streams[i]->codecpar;
    logging("AVStream->time_base before open coded %d/%d", pFormatContext->streams[i]->time_base.num, pFormatContext->streams[i]->time_base.den);
    logging("AVStream->r_frame_rate before open coded %d/%d", pFormatContext->streams[i]->r_frame_rate.num, pFormatContext->streams[i]->r_frame_rate.den);
    logging("AVStream->start_time %" PRId64, pFormatContext->streams[i]->start_time);
    logging("AVStream->duration %" PRId64, pFormatContext->streams[i]->duration);

    logging("finding the proper decoder (CODEC)");

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
    } else if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_AUDIO) {
      logging("Audio Codec: %d channels, sample rate %d", pLocalCodecParameters->channels, pLocalCodecParameters->sample_rate);
    }

    // print its name, id and bitrate
    logging("\tCodec %s ID %d bit_rate %lld", pLocalCodec->name, pLocalCodec->id, pLocalCodecParameters->bit_rate);
  }

  if (video_stream_index == -1) {
    logging("File %s does not contain a video stream!", argv[1]);
    return -1;
  }

  // https://ffmpeg.org/doxygen/trunk/structAVCodecContext.html
  AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
  if (!pCodecContext)
  {
    logging("failed to allocated memory for AVCodecContext");
    return -1;
  }

  // Fill the codec context based on the values from the supplied codec parameters
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#gac7b282f51540ca7a99416a3ba6ee0d16
  if (avcodec_parameters_to_context(pCodecContext, pCodecParameters) < 0)
  {
    logging("failed to copy codec params to codec context");
    return -1;
  }

  // Initialize the AVCodecContext to use the given AVCodec.
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
  if (avcodec_open2(pCodecContext, pCodec, NULL) < 0)
  {
    logging("failed to open codec through avcodec_open2");
    return -1;
  }

  // https://ffmpeg.org/doxygen/trunk/structAVFrame.html
  AVFrame *pFrame = av_frame_alloc();
  if (!pFrame)
  {
    logging("failed to allocated memory for AVFrame");
    return -1;
  }
  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *pPacket = av_packet_alloc();
  if (!pPacket)
  {
    logging("failed to allocated memory for AVPacket");
    return -1;
  }

  int response = 0;

  // fill the Packet with data from the Stream
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga4fdb3084415a82e3810de6ee60e46a61
  while (av_read_frame(pFormatContext, pPacket) >= 0)
  {
    // if it's the video stream
    if (pPacket->stream_index == video_stream_index) {
        //logging("AVPacket->pts %" PRId64, pPacket->pts);
        auto t0 = std::chrono::high_resolution_clock::now();
        
	//auto decoding_t0 = std::chrono::high_resolution_clock::now();
        frame_number++;
	slog::info << "****************************** Frame Number " << frame_number << "*************************************************" << slog::endl;
	response = decode_packet(pPacket, pCodecContext, pFrame, model);
        //auto decoding_t1 = std::chrono::high_resolution_clock::now();
	//double decoding_time = std::chrono::duration_cast<ms>(decoding_t1 - decoding_t0).count();
	/*
	if (response >= 0){
          auto t1 = std::chrono::high_resolution_clock::now();
      	  double decode_inferenece_time = std::chrono::duration_cast<ms>(t1 - t0).count();
	  frame_number++;
          slog::info << "****************************** Frame Number " << frame_number << "*************************************************" << slog::endl;
          logging("Frame %d (type=%c, size=%d bytes, format=%d) pts %d key_frame %d [DTS %d]",
			  pCodecContext->frame_number,
			  av_get_picture_type_char(pFrame->pict_type),
			  pFrame->pkt_size,
			  pFrame->format,
			  pFrame->pts,
			  pFrame->key_frame,
			  pFrame->coded_picture_number
          );
	  
          // Convert avframe (AV_PIX_FMT_YUV420P) to cv::Mat (AV_PIX_FMT_BGR24)
          auto conversion_t0 = std::chrono::high_resolution_clock::now();
          cv::Mat image = avframe_to_cvmat(pFrame);
          auto conversion_t1 = std::chrono::high_resolution_clock::now();
          double conversion_time = std::chrono::duration_cast<ms>(conversion_t1 - conversion_t0).count();
          slog::info << "[Frame " << frame_number << " ] Image Conversion Time:  " << conversion_time << " ms " << slog::endl;
    	  
	  // Inference 
          
	  auto inference_t0 = std::chrono::high_resolution_clock::now();
          model.inference(image);
          auto inference_t1 = std::chrono::high_resolution_clock::now();
          double inference_time = std::chrono::duration_cast<ms>(inference_t1 - inference_t0).count();
          slog::info << "[Frame " << frame_number << " ] Inference Time:  " << inference_time << "ms " << slog::endl;
          slog::info << "[Frame " << frame_number << "] Single Packet Decode + Image Conversion + Inference Time: " << decode_inferenece_time << " ms" << slog::endl;
          total_inference_time += decode_inferenece_time;
	}
	*/
	auto t1 = std::chrono::high_resolution_clock::now();
	double decode_inferenece_time = std::chrono::duration_cast<ms>(t1 - t0).count();
	slog::info << "[Frame " << frame_number << "] Single Packet Decode + Image Conversion + Inference Time: " << decode_inferenece_time << " ms" << slog::endl;
	total_inference_time += decode_inferenece_time;
	if (response < 0) break;
        // stop it, otherwise we'll be saving hundreds of frames
        if (--how_many_packets_to_process <= 0) break;
    }
    // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
    av_packet_unref(pPacket);
  }
  
  double mean_inference_time = total_inference_time / frame_number;
  double mean_throughput = 1.0 / mean_inference_time * 1000;
  slog::info << "Mean Inference Time: " << mean_inference_time << " ms" << slog::endl;
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

static int decode_packet(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, YOLOv3 model)
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
    auto decoding_t1 = std::chrono::high_resolution_clock::now();
    double decoding_time = std::chrono::duration_cast<ms>(decoding_t1 - decoding_t0).count();
    slog::info << "[Frame " << frame_number << " ] Decoding Time:  " << decoding_time << " ms" << slog::endl;
    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
      break;
    } else if (response < 0) {
      logging("Error while receiving a frame from the decoder: %s", av_err2str(response));
      //logging("Error while receiving a frame from the decoder: %s", a(response));
      return response;
    }
    
    if (response >= 0) {
      /*
      logging(
          "Frame %d (type=%c, size=%d bytes, format=%d) pts %d key_frame %d [DTS %d]",
          pCodecContext->frame_number,
          av_get_picture_type_char(pFrame->pict_type),
          pFrame->pkt_size,
          pFrame->format,
          pFrame->pts,
          pFrame->key_frame,
          pFrame->coded_picture_number
      );
      */

      //char frame_filename[1024];
      //snprintf(frame_filename, sizeof(frame_filename), "%s-%d.pgm", "frame", pCodecContext->frame_number);
      // Check if the frame is a planar YUV 4:2:0, 12bpp
      // That is the format of the provided .mp4 file
      // RGB formats will definitely not give a gray image
      // Other YUV image may do so, but untested, so give a warning
      // cv2::Mat mat(avctx->height, avctx->width, CV_8UC3, framergb->data[0], framergb->linesize[0]);
      // cv2::imshow("frame", mat);
      // cv2::waitKey(10);
      //logging("Frame height: %d, width: %d, linesize: %d", pFrame->width, pFrame->height, pFrame->linesize[0]);

      // Convert avframe (AV_PIX_FMT_YUV420P) to cv::Mat (AV_PIX_FMT_BGR24)
      auto conversion_t0 = std::chrono::high_resolution_clock::now();
      cv::Mat image = avframe_to_cvmat(pFrame);
      auto conversion_t1 = std::chrono::high_resolution_clock::now();
      double conversion_time = std::chrono::duration_cast<ms>(conversion_t1 - conversion_t0).count();
      slog::info << "[Frame " << frame_number << " ] Image Conversion Time:  " << conversion_time << " ms " << slog::endl;
      
      // Inference 
      auto inference_t0 = std::chrono::high_resolution_clock::now();
      model.inference(image);
      auto inference_t1 = std::chrono::high_resolution_clock::now();
      double inference_time = std::chrono::duration_cast<ms>(inference_t1 - inference_t0).count();
      slog::info << "[Frame " << frame_number << " ] Inference Time:  " << inference_time << "ms " << slog::endl;
      
      cv::imshow("frame", image);
      cv::waitKey(0);
      
      
      if (pFrame->format != AV_PIX_FMT_YUV420P)
      {
        logging("Warning: the generated file may not be a grayscale image, but could e.g. be just the R component if the video format is RGB");
      }
      
      // save a grayscale frame into a .pgm file
      // save_gray_frame(pFrame->data[0], pFrame->linesize[0], pFrame->width, pFrame->height, frame_filename);
      }
      
  }
  return 0;
}

/*
static void save_gray_frame(unsigned char *buf, int wrap, int xsize, int ysize, char *filename)
{
    FILE *f;
    int i;
    f = fopen(filename,"w");
    // writing the minimal required header for a pgm file format
    // portable graymap format -> https://en.wikipedia.org/wiki/Netpbm_format#PGM_example
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);

    // writing line by line
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}
*/
