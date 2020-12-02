#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>

#define CLEAN	255
#define DIRTY	0

struct Param
{
	uint16_t segment_block_size;
	uint16_t segment_constant;
};

struct Detecion
{
	uint16_t type;								// ����
	uint16_t measure;							// ���
	double aspect;								// �����
	std::vector<std::vector<cv::Point>> area;	// ����
};

struct Point
{
	Point(uint16_t x, uint16_t y)
	{
		this->label = 0;
		this->x = x;
		this->y = y;
	};
	Point(uint16_t label, uint16_t x, uint16_t y)
	{
		this->label = label;
		this->x = x;
		this->y = y;
	};
	uint16_t label;
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
	void detect(cv::Mat& img);

private:
	// config variable
	Param param;

	// local variable
	cv::Mat img;
	cv::Mat binary;
	std::vector<Point> seed;
	std::vector<Detecion> detection;

	// inner fuction
	void segmenting();		// �ָ�ͼ��
	void searching();		// �������ӵ�
	void growing();			// ��������
	void sorting();			// �������
};


