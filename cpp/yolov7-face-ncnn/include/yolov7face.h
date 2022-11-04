// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLOFACE_H
#define YOLOFACE_H

#include <net.h>
#include <opencv2/core/core.hpp>

#define YOLOFACE_INPUT_WIDTH  512
#define YOLOFACE_INPUT_HEIGHT 288
#define NUM_OUTPUTS 3
#define NUM_KEYPOINTS 5

struct Point{
    float x;
	float y;
	float prob;
};

typedef struct Object
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	Point landmark[NUM_KEYPOINTS];
} Object;


class YOLOV7Face
{
public:

    int init(const char* modeltype, int num_threads);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.5f, float nms_threshold = 0.45f);

    int destroy() { return 0; }

private:

	void decode(ncnn::Mat data, std::vector<int> anchor, std::vector<Object> &prebox, float threshold, int stride);
	std::vector<float>LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size);

private:

    ncnn::Net yoloface;
	int num_threads = 4;

    float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
	std::vector<int> anchor0 = {4,5,  6,8,  10,12};
	std::vector<int> anchor1 = {15,19,  23,30,  39,52};
	std::vector<int> anchor2 = {72,97,  123,164,  209,297};
};

#endif // NANODET_H
