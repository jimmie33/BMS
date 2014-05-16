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

#define MAX_IMG_DIM 400

using namespace cv;
using namespace std;

void help()
{
	cout<<"Usage: \n"
		<<"BMSSOD <input_path> <output_path> <step_size> <postprocess_width>\n"
		<<"Press ENTER to continue ..."<<endl;
	getchar();
}


void doWork(
	const string& in_path,
	const string& out_path,
	int sample_step,
	int postprocess_width
	)
{
	if (in_path.compare(out_path)==0)
		cerr<<"output path must be different from input path!"<<endl;
	FileGettor fg(in_path.c_str());
	vector<string> file_list=fg.getFileList();

	clock_t ttt;
	double avg_time=0;
	//#pragma omp parallel for
	for (int i=0;i<file_list.size();i++)
	{
		/* get file name */
		string ext=getExtension(file_list[i]);
		if (!(ext.compare("jpg")==0 || ext.compare("jpeg")==0 || ext.compare("JPG")==0 || ext.compare("tif")==0 || ext.compare("png")==0 || ext.compare("bmp")==0))
			continue;
		cout<<file_list[i]<<"...";

		/* Preprocessing */
		Mat src=imread(in_path+file_list[i]);
		Mat src_small;
		float w = (float)src.cols, h = (float)src.rows;
		float maxD = max(w,h);
		resize(src,src_small,Size((int)(MAX_IMG_DIM*w/maxD),(int)(MAX_IMG_DIM*h/maxD)),0.0,0.0,INTER_AREA);// standard: width: 600 pixel
		cvtColor(src_small, src_small, CV_RGB2Lab);

		/* Computing saliency */
		ttt=clock();

		BMS bms(src_small);
		bms.computeSaliency((double)sample_step);
		
		Mat result=bms.getSaliencyMap();

		/* Post-processing */
		postProcessByRec8u(result, postprocess_width);
		normalize(result, result, 0.0, 255.0, NORM_MINMAX);

		ttt=clock()-ttt;
		float process_time=(float)ttt/CLOCKS_PER_SEC;
		avg_time+=process_time;
		cout<<"average_time: "<<avg_time/(i+1)<<endl;

		/* Save the saliency map*/
		resize(result,result,src.size());
		imwrite(out_path+rmExtension(file_list[i])+".png",result);		
	}
}


int main(int args, char** argv)
{
	if (args != 5)
	{
		cout<<"wrong number of input arguments."<<endl;
		help();
		return 1;
	}

	/* initialize system parameters */
	string INPUT_PATH		=	argv[1];
	string OUTPUT_PATH		=	argv[2];
	int SAMPLE_STEP			=	atoi(argv[3]);//8: delta

	/*Note: we transform the kernel width to the equivalent iteration 
	number for OpenCV's **dilate** and **erode** functions**/	
	int POSTPROCESS_WIDTH	=	atoi(argv[4]);//3: omega_d1
	

	doWork(INPUT_PATH,OUTPUT_PATH,SAMPLE_STEP,POSTPROCESS_WIDTH);

	return 0;
}