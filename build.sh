g++ -g -Wall main.cpp -L/usr/local/lib -lavdevice -lm -lxcb -lxcb-shm -lavfilter -pthread -lm -lva -lpostproc -lm -lavformat -lm -lz -lavcodec -pthread -lm -lz -llzma  -lva -lswscale -lm -lswresample -lm -lavutil -pthread -lva-drm -lva -lm -lva -lvdpau -lX11 -lva-x11 -I ./include  -I /opt/intel/openvino_2021/deployment_tools/ngraph/include  -I /opt/intel/openvino_2021/inference_engine/include/ -I /opt/intel/openvino_2021/opencv/include/ -L /opt/intel/openvino_2021/inference_engine/lib/intel64 -linference_engine  -L/opt/intel/openvino_2021/inference_engine/external/tbb/lib/  -ltbb  -L /opt/intel/openvino_2021/deployment_tools/ngraph/lib/ -linference_engine_ir_reader -linference_engine_legacy  -linference_engine_preproc -linference_engine_lp_transformations -lngraph -linference_engine_transformations -L /opt/intel/openvino_2021/opencv/lib -lopencv_imgcodecs -lopencv_imgproc  -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_video  -pthread -lz -o main
