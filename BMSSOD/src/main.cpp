/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Saliency Detection: A Boolean Map Approach", Jianming Zhang, 
*	Stan Sclaroff, ICCV, 2013
*	
*	Copyright (C) 2013 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: jmzhang@bu.edu
*******************************************************************************/

#include <iostream>
#include <ctime>

#include "opencv2/opencv.hpp"
#include "BMS.h"
#include "fileGettor.h"

#define MAX_IMG_DIM 300

using namespace cv;
using namespace std;

void help()
{
	cout<<"Usage: \n"
		<<"MBS <input_path> <output_path> <use_color_whitening>\n"
		<< "example:\n"
		<< "MBS your/input/image/ your/output/image/ 1\n"
		<<"Press ENTER to continue ..."<<endl;
	getchar();
}

Mat getDisMap()
{
	Mat ret(MAX_IMG_DIM, MAX_IMG_DIM, CV_32FC1);
	float centerX = MAX_IMG_DIM / 2.0f;
	float centerY = MAX_IMG_DIM / 2.0f;

	for (int r = 0; r < MAX_IMG_DIM; r++)
	{
		for (int c = 0; c < MAX_IMG_DIM; c++)
		{
			ret.at<float>(r, c) = sqrt((r - centerY)*(r - centerY) + (c - centerX)*(c - centerX));
		}
	}

	normalize(ret, ret, 0.0, 1.0, NORM_MINMAX);
	ret = 1.0 - ret;

	return ret;
}

void doWork(
	const string& in_path,
	const string& out_path,
	bool use_cws
	)
{
	/*namedWindow("debug1");
	namedWindow("debug2");*/
	if (in_path.compare(out_path)==0)
		cerr<<"output path must be different from input path!"<<endl;
	FileGettor fg(in_path.c_str());
	vector<string> file_list=fg.getFileList();

	Mat disMat = getDisMap();

	clock_t ttt;
	double avg_time=0;
	//#pragma omp parallel for
	int img_count = 0;
	for (int i=0;i<file_list.size();i++)
	{
		/* get file name */
		string ext=getExtension(file_list[i]);
		if (!(ext.compare("jpg")==0 || ext.compare("jpeg")==0 || ext.compare("JPG")==0 || ext.compare("tif")==0 || ext.compare("png")==0 || ext.compare("bmp")==0))
			continue;
		cout<<file_list[i]<<"..."<<endl;

		/* Preprocessing */
		Mat src=imread(in_path+file_list[i]);


		ttt = clock();

		Mat src_small;
		float w = (float)src.cols, h = (float)src.rows;
		float maxD = max(w,h);
		resize(src,src_small,Size((int)(MAX_IMG_DIM*w/maxD),(int)(MAX_IMG_DIM*h/maxD)),0.0,0.0,INTER_AREA);// standard: width: 300 pixel
		Mat srcRoi;
		Rect roi;
		removeFrame(src_small, srcRoi, roi);

		cvtColor(srcRoi, srcRoi, CV_RGB2Lab);
		
		/* Computing saliency */
		BMS bms(srcRoi);
		bms.computeSaliency(use_cws);
		
		Mat resultRoi=bms.getSaliencyMap(disMat);
		Mat result = Mat::zeros(src_small.size(), CV_32FC1);


		/* Post-processing */
		int postprocess_width = (int)MAX(floor(sqrt(sum(resultRoi)[0] / (255.0*resultRoi.rows*resultRoi.cols))*MAX_IMG_DIM/6.0),3);
		postProcessByRec8u(resultRoi, postprocess_width, -1.0);
		resultRoi.convertTo(resultRoi, CV_32FC1);
		normalize(resultRoi, resultRoi, 0.0, 1.0, NORM_MINMAX);
		Mat bmsMap = bms.getBMSMap();
		bmsMap.convertTo(bmsMap,CV_8UC1,255.0);

#ifdef IMDEBUG
		imdebug("lum b=32f w=%d h=%d %p", bms.getCWSMap().cols, bms.getCWSMap().rows, bms.getCWSMap().data);
		imdebug("lum b=32f w=%d h=%d %p", bms.getBMSMap().cols, bms.getBMSMap().rows, bms.getBMSMap().data);
		imdebug("lum w=%d h=%d %p", bms.getSaliencyMap().cols, bms.getSaliencyMap().rows, bms.getSaliencyMap().data);
		imdebug("lum b=32f w=%d h=%d %p", resultRoi.cols, resultRoi.rows, resultRoi.data);
#endif

		
		double mVal1 = mean(resultRoi, bmsMap > 127)[0];
		double mVal2 = mean(resultRoi, bmsMap <= 127)[0];

		exp(-10*(resultRoi - 0.5*(mVal1+mVal2)), resultRoi);
		resultRoi += 1.0;
		resultRoi = 1.0 / resultRoi;

#ifdef IMDEBUG
		imdebug("lum b=32f w=%d h=%d %p", resultRoi.cols, resultRoi.rows, resultRoi.data);
#endif

		normalize(resultRoi, Mat(result, roi), 0.0, 255.0, NORM_MINMAX);
		result.convertTo(result, CV_8UC1);
		
		resize(result, result, src.size());

		img_count++;
		ttt=clock()-ttt;
		float process_time=(float)ttt/CLOCKS_PER_SEC;
		avg_time+=process_time;
		

		/* Save the saliency map*/
		if (use_cws)
			imwrite(out_path+rmExtension(file_list[i])+"_MB+.png",result);		
		else
			imwrite(out_path + rmExtension(file_list[i]) + "_MB.png", result);
	}
	cout << "average_time: " << avg_time / img_count << endl;
	getchar();
}


int main(int args, char** argv)
{
	if (args != 4)
	{
		cout<<"wrong number of input arguments."<<endl;
		help();
		return 1;
	}

	/* initialize system parameters */
	string INPUT_PATH		=	argv[1];
	string OUTPUT_PATH		=	argv[2];
	bool USE_CWS			=	(bool)atoi(argv[3]);
	

	doWork(INPUT_PATH,OUTPUT_PATH, USE_CWS);

	return 0;
}