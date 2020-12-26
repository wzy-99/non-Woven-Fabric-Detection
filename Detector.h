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
	uint16_t dirty_block_size;		// ��Ⱦ��Ĵ�С��ֵ
	double_t dirty_aspect_thres;	// ��Ⱦ��ĳ������ֵ
	double_t dirty_area_thres;		// ��Ⱦ����������ֵ
	double_t canny_thres_1;			// canny���ӵ���ֵ1
	double_t canny_thres_2;			// canny���ӵ���ֵ2
	uint16_t close_block_size;		// ������˴�С
	uint16_t close_iteration;		// �������������
	double_t homo_gamma_high;		// ̬ͬ�˲���Ƶ����
	double_t homo_gamma_low;		// ͬԴ�˲���Ƶ����
	double_t homo_constant;			// ̬ͬ�˲�����
	double_t homo_gain;				// ̬ͬ�˲�����
	uint16_t median_size;			// ��ֵ�˲���С
	uint16_t sobel_thres_x;			// sobel����X��ֵ
	uint16_t sobel_thres_y;			// sobel����Y��ֵ
	uint16_t threshold;				// ��ֵ�ָ���ֵ

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
	uint16_t type;								// ����
	uint16_t measure;							// ���
	uint16_t width;								// ���
	uint16_t height;							// ����
	uint16_t xmin, xmax, ymin, ymax;			// ��������
	double_t iou;								// �����
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
	// config variable
	Param param;

	// class base fuction
	Detector(Param& param);

	// interface fuction
	void set(Param& param);

	// detect fuction
	void detect(cv::Mat& im);								// �������
	std::vector<Detection>& detect_sundry(cv::Mat& im);		// �������
	cv::UMat& detect_crease(cv::Mat& im);					// ����ۺ�

	// get result
	std::vector<Detection>& get_sundry();					// ��ȡ���������
	cv::UMat& get_crease();									// ��ȡ����ۺ۽��

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
	void canny_op();									// ��Ե��ȡ
	void close_op();									// ������
	void searching();									// �������ӵ�
	void growing(cv::Mat&, uint16_t, uint16_t, uint16_t);// ��������
	void sorting();										// �������
	void homo_filter();									// ̬ͬ�˲�
	void median_filter();								// ��ֵ�˲�
	void sobel_op();									// �ݶȼ���
};