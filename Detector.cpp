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

std::vector<Detecion>& Detector::detect(cv::Mat& img)
{
	this->img = img;

	segmenting();
	searching();
	sorting();

	return this->detection;
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
	uint16_t label = 0;
	for (uint16_t i = 0; i < this->binary.rows; i++)
	{
		for (uint16_t j = 0; j < this->binary.cols; j++)
		{
			if (this->binary.at<uchar>(i, j) == DIRTY)
			{
				growing(i, j, label);
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
		while (this->detection.size() < label)
		{
			this->detection.push_back(Detecion());
		}
		this->detection[label].area.push_back(point);

		// 判断
		if (point.x > 0) //left
		{
			if (this->binary.at<uchar>(point.y, point.x - 1) == DIRTY)
			{
				points.push(Point(y, x - 1));
			}
		}
		if (point.x < max_x) //right
		{
			if (this->binary.at<uchar>(point.y, point.x + 1) == DIRTY)
			{
				points.push(Point(y, x + 1));
			}
		}
		if (point.y > 0) //up
		{
			if (this->binary.at<uchar>(point.y - 1, point.x) == DIRTY)
			{
				points.push(Point(y - 1, x));
			}
		}
		if (point.y < 0) //down
		{
			if (this->binary.at<uchar>(point.y + 1, point.x) == DIRTY)
			{
				points.push(Point(y + 1, x));
			}
		}
	}
}

void Detector::sorting()
{
	for (std::vector<Detecion>::iterator iter = this->detection.begin(); iter != this->detection.end(); iter++)
	{
		if ((*iter).area.empty())
		{
			this->detection.erase(iter);
		}
		else if((*iter).area.size() < this->param.dirty_block_size)
		{
			this->detection.erase(iter);
		}
		else
		{
			(*iter).measure = (*iter).area.size();

			uint16_t min_x = this->binary.rows;
			uint16_t min_y = this->binary.cols;
			uint16_t max_x = 0;
			uint16_t max_y = 0;
			for (std::vector<Point>::iterator it = (*iter).area.begin(); it != (*iter).area.end(); iter++)
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
		}
	}
}
