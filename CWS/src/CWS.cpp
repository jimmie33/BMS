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

#include "CWS.h"

#include <vector>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;


Mat computeCWS(const Mat src,float reg, float marginRatio)
{
	assert(mSrc.channels() == 3);

	vector<Mat> sampleVec(4);
	Mat srcF;
	Mat ret(src.size(), CV_32FC1);

	src.convertTo(srcF, CV_64FC3);
	int rowMargin = (int)(marginRatio*src.rows);
	int colMargin = (int)(marginRatio*src.cols);
	
	sampleVec[0] = Mat(srcF,Range(0,rowMargin)).clone();
	sampleVec[1] = Mat(srcF,Range(src.rows-rowMargin,src.rows)).clone();
	sampleVec[2] = Mat(srcF,Range::all(),Range(0,colMargin)).clone();
	sampleVec[3] = Mat(srcF,Range::all(),Range(src.cols-colMargin,src.cols)).clone();

	Mat maxMap(src.size(), CV_32FC1);

	for (int i = 0; i < 4; i++)
	{
		Mat meanF, covF;
		Mat samples = sampleVec[i].reshape(1, sampleVec[i].rows*sampleVec[i].cols);
		calcCovarMatrix(samples, covF, meanF, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);

		covF += Mat::eye(covF.rows, covF.cols, CV_64FC1)*reg;
		

		Mat srcFTemp = srcF - Scalar(meanF.at<double>(0, 0), meanF.at<double>(0, 1), meanF.at<double>(0, 2));
		srcFTemp = srcFTemp.reshape(1, src.rows*src.cols);
		Mat whitenedSrc = (srcFTemp*covF.inv()).mul(srcFTemp);
		whitenedSrc.convertTo(whitenedSrc, CV_32FC1);
		reduce(whitenedSrc, whitenedSrc, 1, CV_REDUCE_SUM);
		whitenedSrc = whitenedSrc.reshape(1, src.rows);
		sqrt(whitenedSrc, whitenedSrc);
		normalize(whitenedSrc, whitenedSrc, 1.0, 0.0, NORM_MINMAX);
		if (i == 1)
			maxMap = max(ret, whitenedSrc);
		else if (i > 1)
			maxMap = max(maxMap, whitenedSrc);

		if (i == 0)
			ret = whitenedSrc;
		else
			ret += whitenedSrc;
		
	}

	ret -= maxMap;

	return ret;
}

#ifdef USE_IPP
IppStatus ippiDilate32fWrapper(const Mat src, Mat& dst, int kernelWidth)
{
	int step = (int)src.cols*sizeof(float);
	Ipp32f *pSrc = (Ipp32f*)src.data, *pDst = (Ipp32f*)dst.data;
	IppStatus status;
	IppiSize roiSize = { src.cols, src.rows };
	IppiSize maskSize = {kernelWidth,kernelWidth};

	Mat kernel = Mat::ones(kernelWidth, kernelWidth, CV_8UC1);
	Ipp8u* pMask = kernel.data;
	IppiPoint anchor = {kernelWidth/2, kernelWidth/2};

	status = ippiDilate_32f_C1R(pSrc, step, pDst, step, roiSize, pMask, maskSize, anchor);

	return status;
}

IppStatus ippiErode32fWrapper(const Mat& src, Mat& dst, int kernelWidth)
{
	int step = (int)src.cols*sizeof(float);
	Ipp32f *pSrc = (Ipp32f*)src.data, *pDst = (Ipp32f*)dst.data;
	IppStatus status;
	IppiSize roiSize = { src.cols, src.rows };
	IppiSize maskSize = { kernelWidth, kernelWidth };

	Mat kernel = Mat::ones(kernelWidth, kernelWidth, CV_8UC1);
	Ipp8u* pMask = kernel.data;
	IppiPoint anchor = { kernelWidth / 2, kernelWidth / 2 };

	status = ippiErode_32f_C1R(pSrc, step, pDst, step, roiSize, pMask, maskSize, anchor);

	cout << src.step << "," << dst.step << endl;

	return status;
}

IppStatus ippiRecDilateWrapper(const Mat& src, Mat& srcDst, int kernelWidth)
{
	int step = (int)src.cols*sizeof(float), buffSize;
	Ipp32f *pSrc = (Ipp32f*)src.data, *pSrcDst = (Ipp32f*)srcDst.data;
	IppStatus status;
	IppiSize roiSize = { src.cols, src.rows };
	IppiSize maskSize = { kernelWidth, kernelWidth };
	Ipp8u *pBuff = NULL;
	
	status = ippiMorphReconstructGetBufferSize_32f_C1(roiSize, &buffSize);
	if (status != ippStsNoErr) return status;
	
	pBuff = ippsMalloc_8u(buffSize);
	status = ippiMorphReconstructDilate_32f_C1IR(pSrc, step, pSrcDst, step, roiSize, (Ipp32f*)pBuff, ippiNormL1);

	ippsFree(pBuff);
	return status;
}
IppStatus ippiRecErodeWrapper(const Mat& src, Mat& srcDst, int kernelWidth)
{
	int step = (int)src.cols*sizeof(float), buffSize;
	Ipp32f *pSrc = (Ipp32f*)src.data, *pSrcDst = (Ipp32f*)srcDst.data;
	IppStatus status;
	IppiSize roiSize = { src.cols, src.rows };
	IppiSize maskSize = { kernelWidth, kernelWidth };
	Ipp8u *pBuff = NULL;

	status = ippiMorphReconstructGetBufferSize_32f_C1(roiSize, &buffSize);
	if (status != ippStsNoErr) return status;

	pBuff = ippsMalloc_8u(buffSize);
	status = ippiMorphReconstructErode_32f_C1IR(pSrc, step, pSrcDst, step, roiSize, (Ipp32f*)pBuff, ippiNormL1);

	ippsFree(pBuff);
	return status;
}
#endif

void postProcessByRec(cv::Mat& salmap, int kernelWidth)
{
	assert(salmap.type() == CV_32FC1);
#ifdef USE_IPP
	Mat temp(salmap.size(),CV_32FC1);
	IppStatus status;
	
	/*status = ippiErode32fWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Erosion: " << ippGetStatusString(status) << endl;*/
	erode(salmap, temp, Mat(), Point(-1,-1), kernelWidth/2);

	status = ippiRecDilateWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: DilationRec: " << ippGetStatusString(status) << endl;

	/*status = ippiDilate32fWrapper(temp, salmap, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Dilation: " << ippGetStatusString(status) << endl;*/
	dilate(temp, salmap, Mat(), Point(-1, -1), kernelWidth / 2);

	status = ippiRecErodeWrapper(temp, salmap, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: ErosionRec: " << ippGetStatusString(status) << endl;
#endif
}