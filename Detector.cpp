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
	this->binary= cv::Mat(img.rows, img.cols, CV_8UC1);
	this->detections = std::vector<Detection>();

	segmenting();
	searching();
	sorting();

	return this->detections;
}

void Detector::segmenting()
{
	cv::cvtColor(this->img, this->gray, cv::COLOR_BGR2GRAY);

	//cv::adaptiveThreshold(this->gray,
	//	this->binary,
	//	255,
	//	cv::ADAPTIVE_THRESH_MEAN_C,
	//	cv::THRESH_BINARY,
	//	this->param.segment_block_size,
	//	this->param.segment_constant);

	//for (uint16_t i = 0; i < this->binary.rows; i++)
	//{
	//	for (uint16_t j = 0; j < this->binary.cols; j++)
	//	{
	//		picking(i, j);
	//	}
	//}

	cv::Mat canny(this->gray.rows, this->gray.cols, this->gray.type());

	cv::Canny(this->gray,
		canny,
		param.canny_thres_1,
		param.canny_thres_2);

	cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT,
		cv::Size(param.close_block_size, param.close_block_size));
	cv::morphologyEx(canny, this->binary,
		cv::MORPH_CLOSE,
		close_kernel,
		cv::Point(-1, -1),
		this->param.close_iteration);
}

void Detector::picking(uint16_t y, uint16_t x)
{
	const uint16_t max_y = this->gray.rows - 1;
	const uint16_t max_x = this->gray.cols - 1;

	uint16_t sum = 0;
	uint16_t count = 0;
	double_t mean = 0.0;
	for (uint16_t i = y - this->param.picking_block_size; i <= y + this->param.picking_block_size; i++)
	{
		for (uint16_t j = x - this->param.picking_block_size; j <= x + this->param.picking_block_size; j++)
		{
			// �������ͼ��Χ
			if (i > max_y || j > max_x) // ���Ϊ��������Ϊ����Ϊ�Ǹ����ͣ������Ȼ�ܴ�
			{
				continue;
			}
			else
			{
				// ��ʽ 1
				//sum = sum + this->gray.at<uchar>(i, j);
				//count = count + 1;

				// ��ʽ 2
				if (i == y && j == x)  // ��������ĵ�
				{
					continue;
				}
				else
				{
					sum = sum + this->gray.at<uchar>(i, j);
					count = count + 1;
				}
			}
		}
	}

	mean = (double_t)sum / (double_t)count;
	if (mean - this->gray.at<uchar>(y, x) > this->param.picking_thres)
	{
		this->binary.at<uchar>(y, x) = 0;
	}
	else
	{
		this->binary.at<uchar>(y, x) = 255;
	}
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
		// �ȿ���Ȼ��POP
		Point point(points.top());
		points.pop();

		// ���õ���Ϊ�ѷ��ʹ�
		this->binary.at<uchar>(point.y, point.x) = VISIT;

		// ���ٿռ䣬����¼�õ�
		while (this->detections.size() < label)
		{
			this->detections.push_back(Detection());
		}
		this->detections[label-1].area.push_back(point);

		// �ж�
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
		if (point.y < max_y) //down
		{
			if (this->binary.at<uchar>(point.y + 1, point.x) == DIRTY)
			{
				points.push(Point(point.y + 1, point.x));
			}
		}
		if (point.x > 0 && point.y > 0) //left_up
		{
			if (this->binary.at<uchar>(point.y - 1, point.x - 1) == DIRTY)
			{
				points.push(Point(point.y - 1, point.x - 1));
			}
		}
		if (point.x < max_x && point.y > 0) //right_up
		{
			if (this->binary.at<uchar>(point.y - 1, point.x + 1) == DIRTY)
			{
				points.push(Point(point.y - 1, point.x + 1));
			}
		}
		if (point.x > 0 && point.y < max_y) //left_down
		{
			if (this->binary.at<uchar>(point.y + 1, point.x - 1) == DIRTY)
			{
				points.push(Point(point.y + 1, point.x - 1));
			}
		}
		if (point.x < max_x && point.y < max_y) //right_down
		{
			if (this->binary.at<uchar>(point.y + 1, point.x + 1) == DIRTY)
			{
				points.push(Point(point.y + 1, point.x + 1));
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
			iter = this->detections.erase(iter); //�������ȡ��һ��Ԫ��
		}
		else if((*iter).area.size() < this->param.dirty_block_size)
		{
			iter = this->detections.erase(iter); //�������ȡ��һ��Ԫ��
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

			(*iter).xmin = min_x;
			(*iter).xmax = max_x;
			(*iter).ymin = min_y;
			(*iter).ymax = max_y;
			(*iter).width = max_x - min_x > 1 ? max_x - min_x : 1;			// ���
			(*iter).height = max_y - min_y > 1 ? max_y - min_y : 1;			// �߶�
			(*iter).aspect = (double_t)(*iter).width / (double_t)(*iter).height;// �����

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

cv::Mat& Detector::get_binary()
{
	return this->binary;
}
