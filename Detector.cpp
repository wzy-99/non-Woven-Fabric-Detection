#include "Detector.h"
#include <opencv2/imgproc.hpp>

Detector::Detector()
{
	;
}

Detector::Detector(Param& param)
{
	this->param = param;
}

Detector::~Detector()
{
	;
}

void Detector::set(Param& param)
{
	this->param = param;
}

void Detector::detect(cv::Mat& img)
{
	this->img = img;

	segmenting();
	searching();
	growing();
	sorting();
}

void Detector::segmenting()
{
	cv::Mat gray;
	cv::cvtColor(this->img, gray, cv::COLOR_BGR2GRAY);

	cv::adaptiveThreshold(gray,
		this->binary,
		255,
		cv::ADAPTIVE_THRESH_MEAN_C,
		cv::THRESH_BINARY,
		this->param.segment_block_size,
		this->param.segment_constant);
}

void Detector::searching()
{
	for (uint16_t i = 0; i < this->binary.rows; i++)
	{
		for (uint16_t j = 0; j < this->binary.cols; j++)
		{
			if (this->binary.at<uchar>(i, j) == DIRTY)
			{
				this->seed.push_back(Point(i, j));
			}
		}
	}
}

void Detector::growing()
{
	uint16_t label = 1;

	uint16_t max_y = this->binary.rows - 1;
	uint16_t max_x = this->binary.cols - 1;

	for (uint16_t i = 0; i < this->seed.size(); i++)
	{
		Point point = this->seed.at(i);

		if (point.label == 0)  // unlabeled
		{
			point.label = label;
			label = label + 1;
		}

		if (point.x < max_x)
		{
			if (this->binary.at<uchar>(point.x + 1, point.y) == DIRTY)  // right
			{
				this->seed.push_back(Point(i, j));
			}
		}
	}
}


