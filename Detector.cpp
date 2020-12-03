#include "Detector.h"
#include <algorithm>
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

std::vector<Detection>& Detector::detect(cv::Mat& img)
{
	this->img = img;
	this->detections = std::vector<Detection>();

	segmenting();
	searching();
	sorting();

	return this->detections;
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
	uint16_t label = 1;
	for (uint16_t i = 0; i < this->binary.rows; i++)
	{
		for (uint16_t j = 0; j < this->binary.cols; j++)
		{
			if (this->binary.at<uchar>(i, j) == DIRTY)
			{
				growing(i, j, label);
				label = label + 1;
			}
		}
	}
}

void Detector::growing(uint16_t y, uint16_t x, uint16_t label)
{
	uint16_t max_y = this->binary.rows - 1;
	uint16_t max_x = this->binary.cols - 1;

	std::stack<Point> points;
	points.push(Point(y, x));

	while (!points.empty())
	{
		// 先拷贝然后POP
		Point point(points.top());
		points.pop();

		// 将该点设为已访问过
		this->binary.at<uchar>(point.y, point.x) = VISIT;

		// 开辟空间，并记录该点
		while (this->detections.size() < label)
		{
			this->detections.push_back(Detection());
		}
		this->detections[label-1].area.push_back(point);

		// 判断
		if (point.x > 0) //left
		{
			if (this->binary.at<uchar>(point.y, point.x - 1) == DIRTY)
			{
				points.push(Point(point.y, point.x - 1));
			}
		}
		if (point.x < max_x) //right
		{
			if (this->binary.at<uchar>(point.y, point.x + 1) == DIRTY)
			{
				points.push(Point(point.y, point.x + 1));
			}
		}
		if (point.y > 0) //up
		{
			if (this->binary.at<uchar>(point.y - 1, point.x) == DIRTY)
			{
				points.push(Point(point.y - 1, point.x));
			}
		}
		if (point.y < 0) //down
		{
			if (this->binary.at<uchar>(point.y + 1, point.x) == DIRTY)
			{
				points.push(Point(point.y + 1, point.x));
			}
		}
	}
}

void Detector::sorting()
{
	for (std::vector<Detection>::iterator iter = this->detections.begin(); iter != this->detections.end();)
	{
		if ((*iter).area.empty())
		{
			iter = this->detections.erase(iter); //在这里获取下一个元素
		}
		else if((*iter).area.size() < this->param.dirty_block_size)
		{
			iter = this->detections.erase(iter); //在这里获取下一个元素
		}
		else
		{
			(*iter).measure = (*iter).area.size();

			uint16_t min_x = this->binary.cols;
			uint16_t min_y = this->binary.rows;
			uint16_t max_x = 0;
			uint16_t max_y = 0;
			for (std::vector<Point>::iterator it = (*iter).area.begin(); it != (*iter).area.end(); it++)
			{
				if ((*it).x < min_x)
				{
					min_x = (*it).x;
				}
				if ((*it).x > max_x)
				{
					max_x = (*it).x;
				}
				if ((*it).y < min_y)
				{
					min_y = (*it).y;
				}
				if ((*it).y > max_y)
				{
					max_y = (*it).y;
				}
			}

			(*iter).width = max_x - min_x > 1 ? max_x - min_x : 1;			// 宽度
			(*iter).height = max_y - min_y > 1 ? max_y - min_y : 1;			// 高度
			(*iter).aspect = (double)(*iter).width / (double)(*iter).height;// 长宽比

			if ((*iter).aspect > this->param.dirty_aspect_thres || 1 / (*iter).aspect > this->param.dirty_aspect_thres)
			{
				(*iter).type = HAIR;
			}
			else
			{
				(*iter).type = DUST;
			}
			++iter;
		}
	}
}
