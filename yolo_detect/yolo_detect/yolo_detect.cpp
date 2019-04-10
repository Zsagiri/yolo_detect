// medicine_detect.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "time.h"
#include "io.h"
#include "tf_tiny_yoloV3.h"

#include "opencv2/opencv.hpp"

#include <vector>
#include <string>


void get_class(std::string path, std::vector<std::string> *class_names)
{
	std::ifstream infile;
	infile.open(path.data());
	assert(infile.is_open());

	std::string class_name;
	while (std::getline(infile, class_name))
	{
		class_names->push_back(class_name);
	}
}


int main()
{
	int image_size[2] = { 416, 416 };
	float score_ = 0.10;
	float iou_ = 0.3;
	clock_t start, finish;

	std::string model_path = "./model_data/tiny_yolov3_model3.pb";
	int classes_number = 1;

	int model_type = 2;   //0:tiny yoloV3   1:tiny yoloV3 full conv   2:tiny yoloV3 model3

	yolo_config(model_path, classes_number, model_type, score_, iou_);

	std::cout << "yoloV3 config successfully" << std::endl;
	
	//std::string filePath = "./model_data/imgnames.txt";
	//std::vector<std::string> files;
	//get_class(filePath, &files);
	
	while (true)
	//int count = 0;
	//for(int i = 0; i<files.size(); i++)
	{
		std::string imgname, img, img_path, save_path;
		std::cout << "Input image filename: ";
		std::cin >> imgname;
		//imgname = files[i];
		img = "Input image name: " + imgname + ".jpg";
		std::cout << img << std::endl;
		
		img_path = "./VOCdevkit/VOC2007/JPEGImages/" + imgname + ".jpg";
		save_path = "./output_img/" + imgname + "_d.jpg";
		//std::cout << "read image...   " << "count : " << count << std::endl;
		//count++;
		cv::Mat image = cv::imread(img_path, 1);
		std::cout << "read image successfully" << std::endl;
		if (image.empty())
		{
			std::cout << "Open the image failed" << std::endl;
			continue;
		}
		cv::Mat image_0;
		cv::cvtColor(image, image_0, CV_BGR2RGB);

		int height = image.rows;
		int width = image.cols;

		unsigned char *imgBuff = new unsigned char[height * width];
		if (image_0.isContinuous())
			imgBuff = image_0.data;

		std::vector<std::vector<float>> boxes;

		start = clock();
		yolo_img_detect(height, width, imgBuff, boxes);
		finish = clock();
		std::cout << "time: " << (float)(finish - start) / CLOCKS_PER_SEC << " s" << std::endl;

		std::cout << "boxes.size(): " << boxes.size() << std::endl;
		if (boxes.size() == 0)
		{
			std::cout << "Found 0 boxes for " << imgname << ".jpg" << std::endl;
			continue;
		}

		std::cout << "Found " << boxes.size() << " boxes for " << imgname << ".jpg" << std::endl;

		int blemish_count = 0;
		for (int i = 0; i < boxes.size(); i++)
		{
			cv::Point2i lu, rd;
			lu.x = boxes[i][0];
			lu.y = boxes[i][1];
			rd.x = boxes[i][2];
			rd.y = boxes[i][3];
			if (boxes[i][5] == 0)   //boxes[i][5]: 目标类别 
			{
				std::cout << "object: ";
				std::cout << boxes[i][0] << ", " << boxes[i][1] << ", ";
				std::cout << boxes[i][2] << ", " << boxes[i][3] << ", ";
				std::cout << "class probability: " << boxes[i][4] << std::endl;
				blemish_count++;
				cv::rectangle(image, lu, rd, cv::Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		std::cout << "Number of object: " << blemish_count << std::endl;
		std::cout << "\n";
		cv::imwrite(save_path, image);
		std::vector<std::vector<float>>().swap(boxes);
	}

    return 0;
}

