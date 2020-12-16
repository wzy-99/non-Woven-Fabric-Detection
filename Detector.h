#pragma once

/**
	�޷Ĳ����
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
	@class:		�㷨����
	@describe:	�������ע�͡�
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
	@class:		��ά��
	@describe:	��¼Y��X���ꡣ
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
	@class:		�����
	@describe:	�������͡����������ȡ�����
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
	uint16_t type;								// ����
	uint16_t measure;							// ���
	uint16_t width;								// ���
	uint16_t height;							// ����
	uint16_t xmin, xmax, ymin, ymax;			// ��������
	double_t aspect;							// �����
	std::vector<Point> area;					// ��ͨ��
};

/** 
	@class:		�����
	@describe:	������㷨��װ��ʹ���뻷���޹ء�
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
	void segmenting();									// �ָ�ͼ��
	void picking(uint16_t, uint16_t);					// ���ز���
	void searching();									// �������ӵ�
	void growing(uint16_t, uint16_t, uint16_t);			// ��������
	void researching();									// 
	void regrowing();
	void sorting();										// �������
};


