#pragma once

/**
	�޷Ĳ����
**/

#include <iostream>
#include <vector>
#include <stack>
#include <opencv2/core.hpp>

#define CLEAN	255
#define DIRTY	0
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
	uint16_t dirty_block_size;
	double dirty_aspect_thres;

	void operator=(const Param& param)
	{
		this->segment_block_size = param.segment_block_size;
		this->segment_constant = param.segment_constant;
		this->dirty_block_size = param.dirty_block_size;
		this->dirty_aspect_thres = param.dirty_aspect_thres;
	}
};

/**
	@class:		�����
	@describe:	�������͡����������ȡ�����
**/
struct Detecion
{
	Detecion()
	{
		this->type = 0;
		this->measure = 0;
		this->aspect = 0.0;
		this->width = 0;
		this->height = 0;
	}
	uint16_t type;								// ����
	uint16_t measure;							// ���
	uint16_t width;								// ���
	uint16_t height;							// ����
	double aspect;								// �����
	std::vector<Point> area;					// ��ͨ��
};

/**
	@class:		��ά��
	@describe:	��¼Y��X���ꡣ
**/
struct Point
{
	Point(Point& point)
	{
		this->x = point.x;
		this->y = point.y;
	};
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
	std::vector<Detecion>& detect(cv::Mat& img);

private:
	// config variable
	Param param;

	// local variable
	cv::Mat img;
	cv::Mat binary;
	std::vector<Detecion> detection;

	// inner fuction
	void segmenting();									// �ָ�ͼ��
	void searching();									// �������ӵ�
	void growing(uint16_t, uint16_t, uint16_t);			// ��������
	void sorting();										// �������
};


