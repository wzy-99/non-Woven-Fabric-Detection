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
	uint16_t segment_block_size;
	uint16_t segment_constant;
	uint16_t growing_block_size;
	uint16_t picking_block_size;
	double_t picking_thres;
	uint16_t dirty_block_size;
	double_t dirty_aspect_thres;
	double_t canny_thres_1;
	double_t canny_thres_2;
	uint16_t close_block_size;
	uint16_t close_iteration;

	void operator=(const Param& param)
	{
		this->segment_block_size = param.segment_block_size;
		this->segment_constant = param.segment_constant;
		this->growing_block_size = param.growing_block_size;
		this->dirty_block_size = param.dirty_block_size;
		this->dirty_aspect_thres = param.dirty_aspect_thres;
		this->picking_block_size = param.picking_block_size;
		this->picking_thres = param.picking_thres;
		this->canny_thres_1 = param.canny_thres_1;
		this->canny_thres_2 = param.canny_thres_2;
		this->close_block_size = param.close_block_size;
		this->close_iteration = param.close_iteration;
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
	// class base fuction
	Detector();
	Detector(Param& param);
	~Detector();

	// interface fuction
	void set(Param& param);
	std::vector<Detection>& detect(cv::Mat&);

	// get data
	cv::Mat& get_binary();

private:
	// config variable
	Param param;

	// local variable
	cv::Mat img;
	cv::Mat gray;
	cv::Mat binary;
	std::vector<Detection> detections;

	// inner fuction
	void segmenting();									// 分割图像
	void picking(uint16_t, uint16_t);					// 像素采样
	void searching();									// 搜索种子点
	void growing(uint16_t, uint16_t, uint16_t);			// 区域生长
	void researching();									// 
	void regrowing();
	void sorting();										// 结果分类
};


