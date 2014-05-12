#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>

#include "CWS.h"
#include "fileGettor.h"

#define MAX_IMG_DIM 400

using namespace cv;
using namespace std;

void help()
{
	cout << "Usage: \n"
		<< "CWS <input_path> <output_path> <reg> <marginRatio> <openWidth> <centerBias>\n"
		<< "Press ENTER to continue ..." << endl;
	getchar();
}


void doWork(
	const string& in_path,
	const string& out_path,
	float reg,
	float marginRatio,
	int openWidth,
	int centerBias
	)
{
	if (in_path.compare(out_path) == 0)
		cerr << "output path must be different from input path!" << endl;
	FileGettor fg(in_path.c_str());
	vector<string> file_list = fg.getFileList();

	clock_t ttt;
	double avg_time = 0;
	//#pragma omp parallel for
	for (int i = 0; i<file_list.size(); i++)
	{
		/* get file name */
		string ext = getExtension(file_list[i]);
		if (!(ext.compare("jpg") == 0 || ext.compare("jpeg") == 0 || ext.compare("JPG") == 0 || ext.compare("tif") == 0 || ext.compare("png") == 0 || ext.compare("bmp") == 0))
			continue;
		cout << file_list[i] << "...";

		/* Preprocessing */
		Mat src = imread(in_path + file_list[i]);
		Mat src_small;
		float w = (float)src.cols, h = (float)src.rows;
		float maxD = max(w, h);
		resize(src, src_small, Size((int)(MAX_IMG_DIM*w / maxD), (int)(MAX_IMG_DIM*h / maxD)), 0.0, 0.0, INTER_AREA);// standard: width: 600 pixel
		GaussianBlur(src_small,src_small,Size(5,5),0.5,0.5);// removing noise 1

		/* Computing saliency */
		ttt = clock();


		Mat result = computeCWS(src_small,reg,marginRatio);

		/* Post-processing */
		if (openWidth > 0)
			postProcessByRec(result, openWidth);

		normalize(result, result, 0.0, 255.0, NORM_MINMAX);
		result.convertTo(result, CV_8UC1);

		

		ttt = clock() - ttt;
		float process_time = (float)ttt / CLOCKS_PER_SEC;
		avg_time += process_time;
		cout << "average_time: " << avg_time / (i + 1) << endl;

		/*imshow("display", result);
		waitKey(0);*/

		/* Save the saliency map*/
		resize(result, result, src.size());
		imwrite(out_path + rmExtension(file_list[i]) + ".png", result);
	}
}


int main(int args, char** argv)
{
	if (args != 7)
	{
		cout << "wrong number of input arguments." << endl;
		help();
		return 1;
	}

	/* initialize system parameters */
	string INPUT_PATH = argv[1];
	string OUTPUT_PATH = argv[2];

	float reg = (float)atof(argv[3]);
	float marginRatio = (float)atof(argv[4]);
	int openWidth = atoi(argv[5]);
	int centerBias = atoi(argv[6]);


	doWork(INPUT_PATH, OUTPUT_PATH, reg, marginRatio, openWidth, centerBias);

	return 0;
}