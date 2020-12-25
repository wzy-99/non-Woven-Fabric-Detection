#include "Detector.h"
#include <algorithm>
#include <opencv2/imgproc.hpp>

template<typename T>
void arange_1d(cv::Mat& mat, T start, T end, T step=1)
{
	T* p = mat.ptr<T>(0);
	for (T i = start; i < end;)
	{
		*p = i * i;
		i = i + step;
		p = p + 1;
	}
}

void kernel_generate(cv::Mat& k, int16_t x, int16_t y, uint16_t D2, double_t c = 1.0)
{
	cv::Mat kx(1, x, CV_64FC1);
	cv::Mat ky(1, y, CV_64FC1);
	
	arange_1d<double>(kx, -x / 2, x - x / 2);
	arange_1d<double>(ky, -y / 2, y - y / 2);

	kx = cv::repeat(kx, y, 1);
	ky = cv::repeat(ky, x, 1).t();
	
	cv::add(kx, ky, k);
	k = -(c / (2 * D2)) * k;
	cv::exp(k, k);
}

void fft2(const cv::Mat& src, cv::Mat& Fourier)
{
	int mat_type = src.type();
	assert(mat_type < 15); //不支持的数据格式

	if (mat_type < 7)
	{
		cv::Mat planes[] = { cv::Mat_<double>(src), cv::Mat::zeros(src.size(),CV_64F) };
		cv::merge(planes, 2, Fourier);
		cv::dft(Fourier, Fourier);
	}
	else
	{
		cv::Mat tmp;
		cv::dft(src, tmp);
		std::vector<cv::Mat> planes;
		cv::split(tmp, planes);
		cv::magnitude(planes[0], planes[1], planes[0]); //将复数转化为幅值
		Fourier = planes[0];
	}
}

void ifft2(const cv::Mat& src, cv::Mat& Fourier)
{
	int mat_type = src.type();
	assert(mat_type < 15); //不支持的数据格式

	if (mat_type < 7)
	{
		cv::Mat planes[] = { cv::Mat_<double>(src), cv::Mat::zeros(src.size(),CV_64F) };
		cv::merge(planes, 2, Fourier);
		cv::dft(Fourier, Fourier, cv::DFT_INVERSE + cv::DFT_SCALE, 0);
	}
	else // 7<mat_type<15
	{
		cv::Mat tmp;
		dft(src, tmp, cv::DFT_INVERSE + cv::DFT_SCALE, 0);
		std::vector<cv::Mat> planes;
		cv::split(tmp, planes);
		cv::magnitude(planes[0], planes[1], planes[0]); //将复数转化为幅值
		Fourier = planes[0];
	}
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
	this->binary = cv::Mat(img.rows, img.cols, CV_8UC1);
	this->homo = cv::Mat(img.rows, img.cols, CV_8UC1);
	this->median = cv::Mat(img.rows, img.cols, CV_8UC1);
	this->detections = std::vector<Detection>();

	segmenting();
	searching();
	sorting();
	homo_filter();
	median_filter();
	gradient();

	return this->detections;
}

void Detector::segmenting()
{
	cv::cvtColor(this->img, this->gray, cv::COLOR_BGR2GRAY);

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


void Detector::searching()
{
	uint16_t label = 1;

#pragma omp parallel for
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
			iter = this->detections.erase(iter); // 在这里获取下一个元素
		}
		else if((*iter).area.size() < this->param.dirty_block_size)
		{
			iter = this->detections.erase(iter); // 在这里获取下一个元素
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
			(*iter).width = max_x - min_x > 1 ? max_x - min_x : 1;				 // 宽度
			(*iter).height = max_y - min_y > 1 ? max_y - min_y : 1;				 // 高度
			(*iter).aspect = (double_t)(*iter).width / (double_t)(*iter).height; // 长宽比

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

void Detector::homo_filter()
{
	cv::Mat fourier(this->gray.size(), CV_64FC2);
	cv::Mat kernel(this->binary.size(), CV_64FC1);

	// 指数变换
	cv::Mat temp(this->gray);
	temp.convertTo(temp, CV_64FC1);
	cv::log(temp, temp);

	// 二维傅里叶变换
	fft2(temp, fourier);
	
	// L2距离
	uint16_t D2 = this->gray.cols * this->gray.cols + this->gray.rows * this->gray.rows;

	// 核函数
	kernel_generate(kernel, this->gray.cols, this->gray.rows, D2, this->param.homo_constant);
	kernel = (this->param.homo_gamma_low - this->param.homo_gamma_high) * kernel + this->param.homo_gamma_high;
	
	// 卷积
	cv::Mat planes[] = { kernel, kernel };
	cv::merge(planes, 2, kernel);
	fourier = fourier.mul(kernel);
	
	// 逆傅里叶变换
	ifft2(fourier, this->homo);

	// 指数变换
	cv::exp(this->homo, this->homo);
	this->homo.convertTo(this->homo, CV_8UC1);
}

void Detector::median_filter()
{
	cv::medianBlur(homo, this->median, this->param.median_size);
}

void Detector::gradient()
{
	cv::Mat sobel_x, sobel_y;

	cv::Sobel(this->median, sobel_x, CV_8UC1, 1, 0, 5);
	cv::Sobel(~this->median, sobel_y, CV_8UC1, 0, 1, 5);

	cv::Mat mask = (sobel_x > this->param.sobel_thres_x) | (sobel_y > this->param.sobel_thres_y);
}

cv::Mat& Detector::get_binary()
{
	return this->binary;
}

cv::Mat& Detector::get_homo()
{
	return this->homo;
}
