#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov7face.h"

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cout << "Usage:" << argv[0] << " <test image path>\n";
		return -1;
	}

	int num_threads = 1;
	YOLOV7Face yolov7face;
	int ret = yolov7face.init("../models/yolov7-lite-e-opt-fp16", num_threads);
	if(ret < 0)
	{
		std::cout << "yolov7face init failed!!!\n";
		return -1;
	}

	cv::Mat image = cv::imread(argv[1]);
	if(image.empty())
	{
		std::cout << "input image is empty!\n";
		return -1;
	}

	std::vector<Object> results;
	yolov7face.detect(image, results);
	std::cout << "detected " << results.size() << " faces\n";
	for(int i = 0; i < results.size(); i ++)
	{
		float x1 = results[i].x1;
		float y1 = results[i].y1;
		float x2 = results[i].x2;
		float y2 = results[i].y2;
		cv::Rect2f box(x1, y1, x2 - x1, y2 - y1);
		cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2, 8, 0);
		for(int j = 0; j < 5; j ++)
		{
			cv::Point2f pt;
			pt.x = results[i].landmark[j].x;
			pt.y = results[i].landmark[j].y;
			if(results[i].landmark[j].prob > 0.5){
				cv::circle(image, pt, 1, cv::Scalar(0,255,0), 2, 8, 0);
			}
		}
	}
	cv::imwrite("result.jpg", image);
	yolov7face.destroy();

	return 0;
}
