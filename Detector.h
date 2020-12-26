#pragma once

/**
	无纺布检测
**/

#include <iostream>
#include <vector>
#include <stack>
#include <opencv2/core.hpp>

#define CLEAN	0
#define DIRTY	255
#define VISIT	127

#define DUST	1
#define HAIR	2


/**
	@class:		算法参数
	@describe:	具体详见注释。
**/
struct Param
{
	uint16_t dirty_block_size;		// 污染物的大小阈值
	double_t dirty_aspect_thres;	// 污染物的长宽比阈值
	double_t dirty_area_thres;		// 污染物的面积比阈值
	double_t canny_thres_1;			// canny算子的阈值1
	double_t canny_thres_2;			// canny算子的阈值2
	uint16_t close_block_size;		// 闭运算核大小
	uint16_t close_iteration;		// 闭运算迭代次数
	double_t homo_gamma_high;		// 同态滤波高频增益
	double_t homo_gamma_low;		// 同源滤波低频增益
	double_t homo_constant;			// 同态滤波常量
	double_t homo_gain;				// 同态滤波增益
	uint16_t median_size;			// 中值滤波大小
	uint16_t sobel_thres_x;			// sobel算子X阈值
	uint16_t sobel_thres_y;			// sobel算子Y阈值
	uint16_t threshold;				// 阈值分割阈值

	void operator=(const Param& param)
	{
		this->dirty_block_size = param.dirty_block_size;
		this->dirty_aspect_thres = param.dirty_aspect_thres;
		this->canny_thres_1 = param.canny_thres_1;
		this->canny_thres_2 = param.canny_thres_2;
		this->close_block_size = param.close_block_size;
		this->close_iteration = param.close_iteration;
		this->homo_gamma_high = param.homo_gamma_high;
		this->homo_gamma_low = param.homo_gamma_low;
		this->homo_constant = param.homo_constant;
		this->homo_gain = param.homo_gain;
		this->median_size = param.median_size;
		this->sobel_thres_x = param.sobel_thres_x;
		this->sobel_thres_y = param.sobel_thres_y;
		this->threshold = param.threshold;
	}
};

/**
	@class:		二维点
	@describe:	记录Y和X坐标。
**/
struct Point
{
	Point(uint16_t y, uint16_t x)
	{
		this->x = x;
		this->y = y;
	};
	uint16_t x;
	uint16_t y;
};

/**
	@class:		检测结果
	@describe:	包括类型、面积、长宽比、区域。
**/
struct Detection
{
	Detection()
	{
		this->type = 0;
		this->measure = 0;
		this->iou = 0.0;
		this->aspect = 0.0;
		this->width = 0;
		this->height = 0;
		this->xmin = 0;
		this->xmax = 0;
		this->ymin = 0;
		this->ymax = 0;
		this->area = std::vector<Point>();
	}
	uint16_t type;								// 类型
	uint16_t measure;							// 面积
	uint16_t width;								// 宽度
	uint16_t height;							// 长度
	uint16_t xmin, xmax, ymin, ymax;			// 左上右下
	double_t iou;								// 面积比
	double_t aspect;							// 长宽比
	std::vector<Point> area;					// 连通域
};

/** 
	@class:		检测类
	@describe:	将检测算法封装，使其与环境无关。
**/
class Detector
{
public:
	// config variable
	Param param;

	// class base fuction
	Detector(Param& param);

	// interface fuction
	void set(Param& param);

	// detect fuction
	void detect(cv::Mat& im);								// 检测所有
	std::vector<Detection>& detect_sundry(cv::Mat& im);		// 检测杂物
	cv::UMat& detect_crease(cv::Mat& im);					// 检测折痕

	// get result
	std::vector<Detection>& get_sundry();					// 获取检测杂物结果
	cv::UMat& get_crease();									// 获取检测折痕结果

	// get temp data for debug
	cv::UMat& get_gray();
	cv::UMat& get_canny();
	cv::UMat& get_binary();
	cv::UMat& get_homo();
	cv::UMat& get_median();
	cv::UMat& get_grad();

private:
	// local variable
	cv::UMat img;
	cv::UMat gray;
	cv::UMat canny;
	cv::UMat closed;
	cv::UMat resized;
	cv::UMat homo;
	cv::UMat median;
	cv::UMat grad;
	std::vector<Detection> detections;

	// inner fuction
	void canny_op();									// 边缘提取
	void close_op();									// 闭运算
	void searching();									// 搜索种子点
	void growing(cv::Mat&, uint16_t, uint16_t, uint16_t);// 区域生长
	void sorting();										// 结果分类
	void homo_filter();									// 同态滤波
	void median_filter();								// 中值滤波
	void sobel_op();									// 梯度计算
};