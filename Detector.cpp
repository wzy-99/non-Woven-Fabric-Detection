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

void kernel_generate(cv::UMat& k, int16_t x, int16_t y, uint16_t D2, double_t c = 1.0)
{
	cv::Mat kx(1, x, CV_64FC1);
	cv::Mat ky(1, y, CV_64FC1);
	
	arange_1d<double>(kx, -x / 2, x - x / 2);
	arange_1d<double>(ky, -y / 2, y - y / 2);

	kx = cv::repeat(kx, y, 1);
	ky = cv::repeat(ky, x, 1);

	ky = ky.t();
	
	cv::add(kx, ky, k);

	k = k.mul(-(c / (2 * D2)));

	cv::exp(k, k);
}

void fft2(const cv::UMat& src, cv::UMat& Fourier)
{
	int mat_type = src.type();
	assert(mat_type < 15); //不支持的数据格式

	if (mat_type < 7)
	{
		std::vector<cv::UMat> planes(2);
		planes[0] = src;
		planes[1] = cv::UMat::zeros(src.size(), CV_64F);
		cv::merge(planes, Fourier);
		cv::dft(Fourier, Fourier);
	}
	else
	{
		cv::UMat tmp;
		cv::dft(src, tmp);
		std::vector<cv::UMat> planes;
		cv::split(tmp, planes);
		cv::magnitude(planes[0], planes[1], planes[0]); //将复数转化为幅值
		Fourier = planes[0];
	}
}

void ifft2(const cv::UMat& src, cv::UMat& Fourier)
{
	int mat_type = src.type();
	assert(mat_type < 15); //不支持的数据格式

	if (mat_type < 7)
	{
		std::vector<cv::UMat> planes(2);
		planes[0] = src;
		planes[1] = cv::UMat::zeros(src.size(), CV_64F);
		cv::merge(planes, Fourier);
		cv::dft(Fourier, Fourier, cv::DFT_INVERSE + cv::DFT_SCALE, 0);
	}
	else // 7<mat_type<15
	{
		cv::UMat tmp;
		dft(src, tmp, cv::DFT_INVERSE + cv::DFT_SCALE, 0);
		std::vector<cv::UMat> planes;
		cv::split(tmp, planes);
		cv::magnitude(planes[0], planes[1], planes[0]); //将复数转化为幅值
		Fourier = planes[0];
	}
}

Detector::Detector(Param& param)
{
	this->param = param;
}

void Detector::set(Param& param)
{
	this->param = param;
}

void Detector::detect(cv::Mat& im)
{
	im.copyTo(this->img);

	this->detections = std::vector<Detection>();

	cv::cvtColor(this->img, this->gray, cv::COLOR_BGR2GRAY);
	cv::resize(this->gray, this->resized, img.size() / 3, 0, 0, cv::INTER_LINEAR);

	canny_op();
	close_op();
	searching();
	sorting();
	homo_filter();
	median_filter();
	sobel_op();
}

std::vector<Detection>& Detector::detect_sundry(cv::Mat& im)
{
	im.copyTo(this->img);

	this->detections = std::vector<Detection>();

	cv::cvtColor(this->img, this->gray, cv::COLOR_BGR2GRAY);

	canny_op();
	close_op();
	searching();
	sorting();


	return this->detections;
}

cv::UMat& Detector::detect_crease(cv::Mat& im)
{
	im.copyTo(this->img);

	cv::cvtColor(this->img, this->gray, cv::COLOR_BGR2GRAY);
	cv::resize(this->gray, this->resized, img.size() / 3, 0, 0, cv::INTER_LINEAR);

	homo_filter();
	median_filter();
	sobel_op();

	return this->grad;
}

std::vector<Detection>& Detector::get_sundry()
{
	return this->detections;
}

cv::UMat& Detector::get_crease()
{
	return this->grad;
}

void Detector::canny_op()
{
	cv::Canny(this->gray,
		this->canny,
		param.canny_thres_1,
		param.canny_thres_2);
}

void Detector::close_op()
{
	cv::Mat mask;
	cv::threshold(this->gray, mask, this->param.threshold, 255, cv::THRESH_BINARY);

	cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT,
		cv::Size(param.close_block_size, param.close_block_size));

	cv::morphologyEx(this->canny,
		this->closed,
		cv::MORPH_CLOSE,
		close_kernel,
		cv::Point(-1, -1),
		this->param.close_iteration);

	cv::bitwise_and(this->closed, mask, this->closed);
}

void Detector::searching()
{
	uint16_t label = 1;
	cv::Mat binary = this->closed.getMat(cv::ACCESS_READ);

#pragma omp parallel for
	for (uint16_t i = 0; i < binary.rows; i++)
	{
		for (uint16_t j = 0; j < binary.cols; j++)
		{
			if (binary.at<uchar>(i, j) == DIRTY)
			{
				growing(binary, i, j, label);
				label = label + 1;
			}
		}
	}
}

void Detector::growing(cv::Mat& binary, uint16_t y, uint16_t x, uint16_t label)
{
	uint16_t max_y = binary.rows - 1;
	uint16_t max_x = binary.cols - 1;

	std::stack<Point> points;
	points.push(Point(y, x));

	uint16_t min_x = this->gray.cols;
	uint16_t min_y = this->gray.rows;
	uint16_t max_x = 0;
	uint16_t max_y = 0;

	while (!points.empty())
	{
		// 先拷贝然后POP
		Point point(points.top());
		points.pop();

		// 将该点设为已访问过
		binary.at<uchar>(point.y, point.x) = VISIT;

		// 开辟空间，并记录该点
		while (this->detections.size() < label)
		{
			this->detections.push_back(Detection());
		}
		this->detections[label-1].area.push_back(point);

		// 判断
		if (point.x > 0) //left
		{
			if (binary.at<uchar>(point.y, point.x - 1) == DIRTY)
			{
				points.push(Point(point.y, point.x - 1));
			}
		}
		if (point.x < max_x) //right
		{
			if (binary.at<uchar>(point.y, point.x + 1) == DIRTY)
			{
				points.push(Point(point.y, point.x + 1));
			}
		}
		if (point.y > 0) //up
		{
			if (binary.at<uchar>(point.y - 1, point.x) == DIRTY)
			{
				points.push(Point(point.y - 1, point.x));
			}
		}
		if (point.y < max_y) //down
		{
			if (binary.at<uchar>(point.y + 1, point.x) == DIRTY)
			{
				points.push(Point(point.y + 1, point.x));
			}
		}
		if (point.x > 0 && point.y > 0) //left_up
		{
			if (binary.at<uchar>(point.y - 1, point.x - 1) == DIRTY)
			{
				points.push(Point(point.y - 1, point.x - 1));
			}
		}
		if (point.x < max_x && point.y > 0) //right_up
		{
			if (binary.at<uchar>(point.y - 1, point.x + 1) == DIRTY)
			{
				points.push(Point(point.y - 1, point.x + 1));
			}
		}
		if (point.x > 0 && point.y < max_y) //left_down
		{
			if (binary.at<uchar>(point.y + 1, point.x - 1) == DIRTY)
			{
				points.push(Point(point.y + 1, point.x - 1));
			}
		}
		if (point.x < max_x && point.y < max_y) //right_down
		{
			if (binary.at<uchar>(point.y + 1, point.x + 1) == DIRTY)
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
			uint16_t min_x = this->gray.cols;
			uint16_t min_y = this->gray.rows;
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
			(*iter).width = max_x - min_x > 1 ? max_x - min_x : 1;									// 宽度
			(*iter).height = max_y - min_y > 1 ? max_y - min_y : 1;									// 高度
			(*iter).measure = (*iter).area.size();													// 面积
			(*iter).aspect = (double_t)(*iter).width / (double_t)(*iter).height;					// 长宽比
			(*iter).iou = (double_t)(*iter).measure / (double_t)((*iter).width * (*iter).height);	// 面积比

			if ((*iter).iou < this->param.dirty_area_thres)
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
	cv::UMat fourier(this->resized.size(), CV_64FC2);
	cv::UMat kernel(this->resized.size(), CV_64FC1);

	// 指数变换
	cv::UMat temp(this->resized.size(), this->resized.type());
	cv::copyTo(this->resized, temp, cv::noArray());
	temp.convertTo(temp, CV_64FC1);
	cv::log(temp, temp);

	// 二维傅里叶变换
	fft2(temp, fourier);
	
	// L2距离
	uint16_t D2 = this->resized.cols * this->resized.cols + this->resized.rows * this->resized.rows;

	// 核函数
	kernel_generate(kernel, this->resized.cols, this->resized.rows, D2, this->param.homo_constant);
	kernel = kernel.mul((this->param.homo_gamma_low - this->param.homo_gamma_high));
	cv::add(kernel, this->param.homo_gamma_high, kernel);
	
	// 卷积
	std::vector<cv::UMat> planes(2);
	planes[0] = kernel;
	planes[1] = kernel;
	cv::merge(planes, kernel);
	fourier = fourier.mul(kernel);
	
	// 逆傅里叶变换
	ifft2(fourier, this->homo);

	// 指数变换
	cv::exp(this->homo, this->homo);

	this->homo = this->homo.mul(this->param.homo_gain);

	this->homo.convertTo(this->homo, CV_8UC1);
}

void Detector::median_filter()
{
	cv::medianBlur(homo, this->median, this->param.median_size);
}

void Detector::sobel_op()
{
	cv::Mat sobel_x, sobel_y;

	cv::Sobel(this->median, sobel_x, CV_8UC1, 1, 0, 5);
	cv::Sobel(~this->median.getMat(cv::ACCESS_WRITE), sobel_y, CV_8UC1, 0, 1, 5);

	cv::Mat((sobel_x > this->param.sobel_thres_x) | (sobel_y > this->param.sobel_thres_y)).copyTo(this->grad);

	cv::resize(this->grad, this->grad, this->gray.size(), 0, 0, cv::INTER_LINEAR);
}

cv::UMat& Detector::get_gray()
{
	return this->gray;
}

cv::UMat& Detector::get_canny()
{
	return this->canny;
}

cv::UMat& Detector::get_binary()
{
	return this->closed;
}

cv::UMat& Detector::get_homo()
{
	return this->homo;
}

cv::UMat& Detector::get_median()
{
	return this->median;
}

cv::UMat& Detector::get_grad()
{
	return this->grad;
}
