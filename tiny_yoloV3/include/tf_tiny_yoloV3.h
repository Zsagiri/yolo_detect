// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� TF_TINY_YOLOV3_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// TF_TINY_YOLOV3_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
//#pragma once

//#define COMPILER_MSVC
//#define NOMINMAX

//#ifdef TF_TINY_YOLOV3_EXPORTS
//#define TF_TINY_YOLOV3_API __declspec(dllexport)
//#else
//#define TF_TINY_YOLOV3_API __declspec(dllimport)
//#endif

#include <vector>

#ifdef TF_TINY_YOLOV3_EXPORTS
#define API extern  __declspec(dllexport) 
#else
#define API extern  __declspec(dllimport) 
#endif


/******************
model_path��pb�ļ�·��
classes_number�����Ŀ����������
full_conv_type��ģ����� 0��tiny_yoloV3 ; 1: tiny_yoloV3_full_conv;
score_: ��������ֵ
iou_��Ŀ����غ϶���ֵ
******************/
API int yolo_config(std::string model_path, int classes_number, int model_type, float score_, float iou_);

/******************
wigth��height: ͼ��Ŀ�͸�
imageB: ͼ���buffer
boxes: [[lu_x, lu_y, rd_x, rd_y, probability, class],]
       lu_x,lu_yΪ���Ͻ�x,y����
	   rd_x,rd_yΪ���½�x,y����
	   probabilityΪ��������Ŀ����ĸ���
	   classΪ������������
******************/
API int yolo_img_detect(int height, int wigth, unsigned char *imageB, std::vector<std::vector<float>> &boxes);


