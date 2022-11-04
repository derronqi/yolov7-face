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

#include "yolov7face.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


int YOLOV7Face::init(const char* modeltype, int num_threads)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

	this->num_threads = num_threads;
    yoloface.load_param(parampath);
    yoloface.load_model(modelpath);

    return 0;
}

bool cmp(Object b1, Object b2) {
    return b1.score > b2.score;
}

static inline float sigmoid(float x){
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

void nms(std::vector<Object> &input_boxes, float NMS_THRESH)
{
	std::vector<float>vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		for (int j = i + 1; j < int(input_boxes.size());)
		{
			float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
			float w = std::max(float(0), xx2 - xx1 + 1);
			float h = std::max(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);
			if (ovr >= NMS_THRESH)
			{
				input_boxes.erase(input_boxes.begin() + j);
				vArea.erase(vArea.begin() + j);
			}
			else
			{
				j++;
			}
		}
	}
}

std::vector<float> YOLOV7Face::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size)
{
	auto in_h = static_cast<float>(src.rows);
	auto in_w = static_cast<float>(src.cols);
	float out_h = out_size.height;
	float out_w = out_size.width;

	float scale = std::min(out_w / in_w, out_h / in_h);

	int mid_h = static_cast<int>(in_h * scale);
	int mid_w = static_cast<int>(in_w * scale);

	cv::resize(src, dst, cv::Size(mid_w, mid_h), 0, 0, cv::INTER_NEAREST);

	int top = (static_cast<int>(out_h) - mid_h) / 2;
	int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
	int left = (static_cast<int>(out_w)- mid_w) / 2;
	int right = (static_cast<int>(out_w)- mid_w + 1) / 2;

	cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
	return pad_info;
}

void YOLOV7Face::decode(ncnn::Mat data, std::vector<int> anchor, std::vector<Object> &prebox, float threshold, int stride)
{
	int fea_h = data.h;
	int fea_w = data.w;
	int spacial_size = fea_w*fea_h;
	int channels = NUM_KEYPOINTS * 3 + 6;
	float *ptr = (float*)(data.data);
	for(int c = 0; c < anchor.size() / 2; c++)
	{
		float anchor_w = float(anchor[c * 2 + 0]);
		float anchor_h = float(anchor[c * 2 + 1]);
		float *ptr_x = ptr + spacial_size * (c * channels + 0);
		float *ptr_y = ptr + spacial_size * (c * channels + 1);
		float *ptr_w = ptr + spacial_size * (c * channels + 2);
		float *ptr_h = ptr + spacial_size * (c * channels + 3);
		float *ptr_s = ptr + spacial_size * (c * channels + 4);
		float *ptr_c = ptr + spacial_size * (c * channels + 5);

		for(int i = 0; i < fea_h; i++)
		{
			for(int j = 0; j < fea_w; j++)
			{
				int index = i * fea_w + j;
				float confidence = sigmoid(ptr_s[index]) * sigmoid(ptr_c[index]);
				if(confidence > threshold)
				{
					Object temp_box;
					float dx = sigmoid(ptr_x[index]);
					float dy = sigmoid(ptr_y[index]);
					float dw = sigmoid(ptr_w[index]);
					float dh = sigmoid(ptr_h[index]);
					
					float pb_cx = (dx * 2.f - 0.5f + j) * stride;
					float pb_cy = (dy * 2.f - 0.5f + i) * stride;

					float pb_w = pow(dw * 2.f, 2) * anchor_w;
					float pb_h = pow(dh * 2.f, 2) * anchor_h;

					temp_box.score = confidence;
					temp_box.x1 = pb_cx - pb_w * 0.5f;
					temp_box.y1 = pb_cy - pb_h * 0.5f;
					temp_box.x2 = pb_cx + pb_w * 0.5f;
					temp_box.y2 = pb_cy + pb_h * 0.5f;

					for(int l = 0; l < NUM_KEYPOINTS; l ++)
					{
						temp_box.landmark[l].x = (ptr[(spacial_size * (c * channels + l * 3 + 6)) + index] * 2 - 0.5 + j) * stride;
						temp_box.landmark[l].y = (ptr[(spacial_size * (c * channels + l * 3 + 7)) + index] * 2 - 0.5 + i) * stride;
						temp_box.landmark[l].prob = sigmoid(ptr[spacial_size * (c * channels + l * 3 + 8) + index]);
					}
					prebox.push_back(temp_box);
				}
			}
		}
	}
}

int YOLOV7Face::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
	cv::Mat dst;
	std::vector<float> infos = LetterboxImage(rgb, dst, cv::Size(YOLOFACE_INPUT_WIDTH, YOLOFACE_INPUT_HEIGHT));
    ncnn::Mat in = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_RGB, dst.cols, dst.rows);

    in.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yoloface.create_extractor();
	ex.set_num_threads(num_threads);
	ex.set_light_mode(true);
    ex.input("data", in);

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("stride_8", out);
		decode(out, anchor0, objects, prob_threshold, 8);
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("stride_16", out);
		decode(out, anchor1, objects, prob_threshold, 16);
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("stride_32", out);
		decode(out, anchor2, objects, prob_threshold, 32);
    }

	std::sort(objects.begin(), objects.end(), cmp);
	nms(objects, nms_threshold);

	for(int i = 0; i < objects.size(); i ++)
	{
		objects[i].x1 = (objects[i].x1 - infos[0]) / infos[2];
		objects[i].y1 = (objects[i].y1 - infos[1]) / infos[2];
		objects[i].x2 = (objects[i].x2 - infos[0]) / infos[2];
		objects[i].y2 = (objects[i].y2 - infos[1]) / infos[2];
		for(int j = 0; j < NUM_KEYPOINTS; j ++)
		{
			objects[i].landmark[j].x = (objects[i].landmark[j].x - infos[0]) / infos[2];
			objects[i].landmark[j].y = (objects[i].landmark[j].y - infos[1]) / infos[2];
		}
	}

    return 0;
}

