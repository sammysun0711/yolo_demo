#ifndef YOLO_H // include guard
#define YOLO_H

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

std::string FLAGS_m_yolov3 = "./models/yolov3/INT8/yolo-v3-tf.xml";
std::string FLAGS_m_yolov5 = "./models/yolov5/INT8/yolov5s_v5.xml";
std::string FLAGS_labels = "./coco.names";
std::string FLAGS_d = "CPU";
bool FLAGS_auto_resize = true;
bool FLAGS_r = true;
double FLAGS_t = 0.5;
double FLAGS_iou_t = 0.4;
bool FLAGS_no_show = false;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

using namespace InferenceEngine;
/*
namespace ngraph {
    namespace op {
        namespace v0 {
            class RegionYolo;
        }
        using v0::RegionYolo;
    }
}
*/

double sigmoid(double x){
    return 1.0 / (1.0 + std::exp(-x));
}

cv::Mat letterbox(cv::Mat img, int width=640, int height=640, int color=114, bool automatic=true, bool scaleFill = false, bool scaleUp = true){
    // cv::imwrite("test_img.png", img);
    // Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    int rows = img.rows; // correspond to height
    int cols = img.cols; // correspond to width
    //slog::info << "image shape: ( " << rows << "," << cols << " )" << slog::endl;
    //slog::info << "size: ( " << width << "," << height << " )" << slog::endl;
    cv::Scalar value(color, color, color);
    // Scale ratio (new / old)
    double r = std::min(1.0 * height/rows, 1.0 * width/cols);
    slog::info << "r: " << r << slog::endl;
    if (!scaleUp) {
	    r = std::min(r, 1.0);
    }
    double r_w = r;
    double r_h = r;
    
    // Compute padding
    int new_unpad_cols = (int)(std::round(cols*r));
    int new_unpad_rows = (int)(std::round(rows*r));
    //slog::info << "new_unpad: (" << new_unpad_cols << "." << new_unpad_rows << " )" << slog::endl;
    // Paddding width and height
    int dw = width - new_unpad_cols;
    int dh = height - new_unpad_rows;
    //slog::info << "dw before auto resize: " << dw << slog::endl;
    //slog::info << "dh before auto resize: " << dh << slog::endl;
    if (automatic){
	    dw = dw % 64;
	    dh = dh % 64;
	    //slog::info << "dw after auto resize: " << dw << slog::endl;
            //slog::info << "dh after auto resize: " << dh << slog::endl;
    }
    // Stretch
    else if (scaleFill){
	    dw = 0.0;
	    dh = 0.0;
	    new_unpad_cols = width;
	    new_unpad_rows = height;
	    r_w = 1.0 * width / cols;
	    r_h = 1.0 * height / rows;
    }
    // Divide padding into 2 sides
    dw = dw / 2;
    dh = dh / 2;
    //slog::info << "dw after dividing padding into 2 sides: " << dw << slog::endl;
    //slog::info << "dh after dividing padding into 2 sides:: " << dh << slog::endl;

    // Resize
    cv::Size size(new_unpad_cols, new_unpad_rows);
    if (((img.rows!= new_unpad_rows) && img.cols!=new_unpad_cols)){
    	//slog::info << "img: (" << img.rows << "," << img.cols << ")" << "!= new_unpad: (" << new_unpad_rows << "," << new_unpad_cols << " )" << slog::endl;
	cv::resize(img, img, size, 0, 0, cv::INTER_LINEAR);
    }
    int top = (int)(round(dh - 0.1));
    int bottom = (int)(round(dh + 0.1));
    int left = (int)(round(dw - 0.1));
    int right = (int)(round(dw + 0.1));
    //slog::info << "top: " << top << ", bottom: " << bottom << ", left:" << left << ", right" << right << slog::endl;
    //cv::Mat img_resize;
    //cv::Mat img_resize_2;
    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, value);
    int top2 = 0;
    int bottom2 = 0;
    int left2 = 0;
    int right2 = 0;
    if (img.rows != height){
    	//slog::info << "img.rows: " << img.rows << "!= height: "<< height << slog::endl;
	top2 = (height - img.rows) / 2;
	bottom2 = top2;
	//slog::info << "top2: " << top2 << ", bottom2: " << bottom2 << ", left2: " << left2 << ", right2: " << right2 << slog::endl;
	cv::copyMakeBorder(img, img, top2, bottom2, left2, right2, cv::BORDER_CONSTANT, value);
    }
    else if (img.cols != width){
        //slog::info << "img.cols: " << img.cols << "!= width: "<< width << slog::endl;
    	left2 = (width - img.cols) / 2;
	right2 = left2;
	//slog::info << "top2: " << top2 << ", bottom2: " << bottom2 << ", left2:" << left2 << ", right2: " << right2 << slog::endl;
	cv::copyMakeBorder(img, img, top2, bottom2, left2, right2, cv::BORDER_CONSTANT, value);
    }
    return img;

}

void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
    cv::Mat frame_resize = letterbox(frame);
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        //inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame_resize));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        //matU8ToBlob<uint8_t>(frame, frameBlob);
        matU8ToBlob<uint8_t>(frame_resize, frameBlob);
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
	
    DetectionObject(){}
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


DetectionObject scale_box(DetectionObject *obj, double x, double y, double height, double width, int im_h, int im_w, int resized_im_h=640, int resized_im_w=640){
    DetectionObject scale_obj;
    
    double gain = std::min(1.0 * resized_im_w / im_w, 1.0 * resized_im_h / im_h); // gain = old / new
    //slog::info << "gain: " << gain << slog::endl; 
    double pad_w = (resized_im_w - im_w * gain) / 2;
    double pad_h = (resized_im_h - im_h * gain) / 2;
    
    //slog::info << "pad: (" << pad_w << " , " << pad_h << " )" << slog::endl;
    int x_new = (int)((x - pad_w) / gain);
    int y_new = (int)((y - pad_h) / gain);
    
    //slog::info << "x_new: " << x_new << ", y_new: " << y_new << slog::endl;

    int w = (int)(width / gain);
    int h = (int)(height / gain);
    
    //slog::info << "w: " << w << ", h: " << h << slog::endl; 

    int xmin = std::max(0, (int)(x_new - w / 2));
    int ymin = std::max(0, (int)(y_new - h / 2));
    int xmax = std::min(im_w, (int)(xmin + w));
    int ymax = std::min(im_h, (int)(ymin + h));
    
    //slog::info << "xmin: " << xmin << ", ymin: " << ymin << ", xmax: " << xmax << ", ymax: " << ymax << slog::endl;
    scale_obj.xmin = xmin;
    scale_obj.ymin = ymin;
    scale_obj.xmax = xmax;
    scale_obj.ymax = ymax;
    scale_obj.class_id = obj->class_id;
    scale_obj.confidence = obj->confidence;
    return scale_obj;
    /*
    DetectionObject obj(xmin, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
     this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    */
}

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
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};

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

class YoloParamsV5 {
    /*
    template <typename T>
    void computeAnchors(const std::vector<T> & mask) {
        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i) {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    }
    */

public:
    int num = 3;
    int classes = 80;
    int coords = 4;
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};

    YoloParamsV5() {}
    /*
    YoloParamsV5(const std::shared_ptr<ngraph::op::v0::RegionYolo> regionYolo) {
        coords = regionYolo->get_num_coords();
        classes = regionYolo->get_num_classes();
        anchors = regionYolo->get_anchors();
        auto mask = regionYolo->get_mask();
        num = mask.size();

        computeAnchors(mask);
    }
    */
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

void ParseYOLOV5Output(const YoloParamsV5 &params, const std::string & output_name,
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
            float scale = sigmoid(output_blob[obj_index]);

            if (scale < threshold)
                continue;
            double x_ = sigmoid(output_blob[box_index + 0 * side_square]);
            double y_ = sigmoid(output_blob[box_index + 1 * side_square]);
            double height_ = sigmoid(output_blob[box_index + 3 * side_square]);
            double width_ = sigmoid(output_blob[box_index + 2 * side_square]);
            
	    //slog::info << "original x: " << x_ << ", y: " << y_ << ", height_: " << height_ << ", width_" << width_ << slog::endl;
	    double x = (2*x_ - 0.5 + col)*(1.0 * resized_im_w / side);
	    double y = (2*y_ - 0.5 + row)*(1.0 * resized_im_h / side);
	    int idx;
	    if ((resized_im_w / side == 8) && (resized_im_h / side == 8))          // 80x80
	        idx = 0;
	    else if ((resized_im_w / side == 16) && (resized_im_h / side == 16))   // 40x40
		idx = 1;
	    else if ((resized_im_w / side == 32) && (resized_im_h / side == 32))   // 20x20
		idx = 2;
	    //slog::info << "idx: " << idx << slog::endl;
	    double width = std::pow(2*width_, 2) * params.anchors[idx * 6 + 2 * n];
	    double height = std::pow(2*height_, 2) * params.anchors[idx * 6 + 2 * n + 1];

	    //slog::info << "x: " << x << ", y: " << y << ", height: " << height << ", width: "  << width << slog::endl;
            for (int j = 0; j < params.classes; ++j) {
                int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
		float class_prob = sigmoid(output_blob[class_index]);
		//class_prob = 1.0 / (1.0 + std::exp(-class_prob));
		float prob = scale * class_prob;
		//float prob = class_prob;
		//slog::info << "detection probability: " << prob << slog::endl;
                //float prob = scale * output_blob[class_index];
		//slog::info << "prob before sigmoid: " << prob << slog::endl;
		//prob = 1.0 / (1.0 + std::exp(-prob));
		//slog::info << "prob after sigmoid: " << prob << slog::endl;
                if (prob < threshold)
                    continue;
		//slog::info << "detection probability: " << prob << slog::endl;
		//slog::info << "Before scale box, x: " << x << ", y: " << y << ", height: " << height << ", width: " << width << ", class id: " << class_index << ", prob: " << prob << ", original_im_h: " << original_im_h << ", original_im_w: " << original_im_w << ", resized_im_h: " << resized_im_h << ", resized_im_w: " << resized_im_w << slog::endl;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
		DetectionObject scaled_obj = scale_box(&obj, x, y, height, width,
		          static_cast<int>(original_im_h),
			  static_cast<int>(original_im_w),
			  static_cast<int>(resized_im_h),
			  static_cast<int>(resized_im_w));
		//slog::info << "After scale box, xmin: " << obj.xmin << ", ymin: " << obj.ymin << ", xmax: " << obj.xmax << ", ymax: " << obj.ymax << ", class_id: " << obj.class_id << "confidence: " << obj.confidence << slog::endl;
                //objects.push_back(obj);
                objects.push_back(scaled_obj);
            
            }
        }
    }

}


class YOLOv3 {
	public:
		void initialize_model();
		void inference(cv::Mat& frame, int frame_number);

	private:
		ExecutableNetwork network;
		InputsDataMap inputInfo;
		OutputsDataMap outputInfo;
		std::map<std::string, YoloParams> yoloParams;
		std::vector<std::string> labels;
		std::string name = "YOLOv3";
};

void YOLOv3::initialize_model(){
    try{
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
        auto cnnNetwork = ie.ReadNetwork(FLAGS_m_yolov3);
        
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
    catch(const std::runtime_error& re)
    {
	    // speciffic handling for runtime_error
           std::cerr << "Runtime error: " << re.what() << std::endl;
    }
}

void YOLOv3::inference(cv::Mat& frame, int frame_number){
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

class YOLOv5 : public YOLOv3{
	public:
		void initialize_model();
		void inference(cv::Mat& frame, int frame_number);

	private:
		ExecutableNetwork network;
		InputsDataMap inputInfo;
		OutputsDataMap outputInfo;
		std::map<std::string, YoloParamsV5> yoloParams;
		std::vector<std::string> labels;
		std::string name = "YOLOv5";
};

void YOLOv5::initialize_model() {
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
        auto cnnNetwork = ie.ReadNetwork(FLAGS_m_yolov5);
        
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
        
        /** YOLOV5-based network should have one input and three output **/
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
		//slog::info << "Ops: " << op->get_friendly_name() << slog::endl;
	        //slog::info << "Type info name: " << std::string(op->get_type_info().name) << slog::endl;
		//slog::info << "Type info version: " << op->get_type_info().version << slog::endl;

		//slog::info << "ngraphFunction get name: " << ngraphFunction->get_name() << slog::endl;
		//slog::info << "Type info is_castable: " << op->get_type_info()->is_castable() << slog::endl;

                auto outputLayer = outputInfo.find(op->get_friendly_name());
		//continue;
                if (outputLayer != outputInfo.end()) {
		    // (TODO) YOLOv5 Changes
                    //auto regionYolo = std::dynamic_pointer_cast<ngraph::op::v0::RegionYolo>(op);
		    auto opAdd = std::dynamic_pointer_cast<ngraph::op::v1::Add>(op);
                    if (!opAdd) {
                        throw std::runtime_error("Invalid output type: " +
                            std::string(opAdd->get_type_info().name) + ". Add expected");
                    }
		    // (TODO) YOLOv5 Changes
                    //yoloParams[outputLayer->first] = YoloParamsV5(opAdd);
		    yoloParams[outputLayer->first] = YoloParamsV5();
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

void YOLOv5::inference(cv::Mat& frame, int frame_number) {
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
                //ParseYOLOV3Output(yoloParams[output_name], output_name, blob, resized_im_h, resized_im_w, height, width, FLAGS_t, objects);
		// (TODO YOLOv5 Changes)
		ParseYOLOV5Output(yoloParams[output_name], output_name, blob, resized_im_h, resized_im_w, height, width, FLAGS_t, objects);
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

#endif /* YOLO_H */
