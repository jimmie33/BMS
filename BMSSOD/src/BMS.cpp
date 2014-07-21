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

#include "BMS.h"

#include <vector>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;

#define COV_MAT_REG 50.0f
#define BORDER_MARGIN 30
#define CLUSTER_THRESH 0
#define TOLERANCE 0.01
#define FRAME_MAX 20
#define SOBEL_THRESH 0.4

Mat computeCWS(const Mat src, float reg, float marginRatio)
{
	assert(mSrc.channels() == 3);

	vector<Mat> sampleVec(4);
	vector<Mat> means(4);
	vector<Mat> covs(4);
	//vector<double> pixNum(4);
	//vector<Mat> clusterMeans;
	//vector<Mat> clusterCovs;

	Mat srcF;
	Mat ret(src.size(), CV_32FC1);

	src.convertTo(srcF, CV_32FC3);
	int rowMargin = (int)(marginRatio*src.rows);
	int colMargin = (int)(marginRatio*src.cols);

	sampleVec[0] = Mat(srcF, Range(0, rowMargin)).clone();
	sampleVec[1] = Mat(srcF, Range(src.rows - rowMargin, src.rows)).clone();
	sampleVec[2] = Mat(srcF, Range::all(), Range(0, colMargin)).clone();
	sampleVec[3] = Mat(srcF, Range::all(), Range(src.cols - colMargin, src.cols)).clone();

	/*sampleVec[0] = Mat(srcF, Range(0, rowMargin), Range(0, src.cols/2)).clone();
	sampleVec[1] = Mat(srcF, Range(0, src.rows/2), Range(0, colMargin)).clone();

	sampleVec[2] = Mat(srcF, Range(0, rowMargin), Range(src.cols/2, src.cols)).clone();
	sampleVec[3] = Mat(srcF, Range(0, src.rows / 2), Range(src.cols - colMargin, src.cols)).clone();

	sampleVec[4] = Mat(srcF, Range(src.rows - rowMargin, src.rows), Range(0, src.cols/2)).clone();
	sampleVec[5] = Mat(srcF, Range(src.rows/2, src.rows), Range(0, colMargin)).clone();

	sampleVec[6] = Mat(srcF, Range(src.rows - rowMargin, src.rows), Range(src.cols/2, src.cols)).clone();
	sampleVec[7] = Mat(srcF, Range(src.rows / 2, src.rows), Range(src.cols - colMargin, src.cols)).clone();*/

	Mat maxMap(src.size(), CV_32FC1);

	for (int i = 0; i < 4; i++)
	{
		/*Mat samples;
		vconcat(sampleVec[2*i].reshape(1, sampleVec[2*i].rows*sampleVec[2*i].cols), 
			sampleVec[2*i+1].reshape(1, sampleVec[2*i+1].rows*sampleVec[2*i+1].cols),
			samples);*/
		calcCovarMatrix(sampleVec[i].reshape(1, sampleVec[i].rows*sampleVec[i].cols), covs[i], means[i], CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F);
		//pixNum[i] = samples.rows;
	}

	for (int i = 0; i < means.size(); i ++)
	{
		Mat covF = covs[i], meanF = means[i];
		covF += Mat::eye(covF.rows, covF.cols, CV_32FC1)*reg;
		SVD svd(covF);
		Mat sqrtW;
		sqrt(svd.w, sqrtW);
		Mat sqrtInvCovF = svd.u * Mat::diag(1.0 / sqrtW);

		Mat srcFTemp = srcF - Scalar(meanF.at<float>(0, 0), meanF.at<float>(0, 1), meanF.at<float>(0, 2));
		srcFTemp = srcFTemp.reshape(1, src.rows*src.cols);
		//Mat whitenedSrc = (srcFTemp*covF.inv()).mul(srcFTemp);
		Mat whitenedSrc = srcFTemp*sqrtInvCovF;
		whitenedSrc = abs(whitenedSrc);

		//whitenedSrc.convertTo(whitenedSrc, CV_32FC1);
		reduce(whitenedSrc, whitenedSrc, 1, CV_REDUCE_SUM);
		whitenedSrc = whitenedSrc.reshape(1, src.rows);
		//sqrt(whitenedSrc, whitenedSrc);
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

	normalize(ret, ret, 0.0, 1.0, NORM_MINMAX);
	//ret.convertTo(ret, CV_8U);
	return ret;
}

BMS::BMS(const Mat& src)
:mAttMapCount(0)
{
	mSrc=src.clone();
	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);

	split(mSrc, mFeatureMaps);

	for (int i = 0; i < mFeatureMaps.size(); i++)
	{
		//normalize(mFeatureMaps[i], mFeatureMaps[i], 255.0, 0.0, NORM_MINMAX);
		medianBlur(mFeatureMaps[i], mFeatureMaps[i], 5);
	}
	//mBorderPriorMap = Mat::zeros(src.size(), CV_32FC1);

	//whitenFeatMap(COV_MAT_REG);
	//computeBorderPriorMap(10.0f, 0.25);
	/*Mat lab;
	cvtColor(mSrc,lab,CV_RGB2Lab);

	vector<Mat> maps;
	maps.push_back(lab);

	for (int i=0;i<maps.size();i++)
	{
		vector<Mat> sp;
		split(maps[i],sp);
		mFeatureMaps.push_back(sp[0]);
		mFeatureMaps.push_back(sp[1]);
		mFeatureMaps.push_back(sp[2]);
	}*/

}

void BMS::computeSaliency(double step)
{
	mCWSMap = computeCWS(mSrc, 50.0f, 0.1f);
	//for (int i=0;i<mFeatureMaps.size();++i)
	//{
	//	Mat bm;
	//	double max_,min_;
	//	minMaxLoc(mFeatureMaps[i],&min_,&max_);
	//	//step = (max_ - min_) / 30.0f;
	//	for (double thresh = 0; thresh < 255; thresh += step)
	//	{
	//		bm=mFeatureMaps[i]>thresh;
	//		Mat am = getAttentionMap(bm);
	//		mSaliencyMap += am;
	//		mAttMapCount++;
	//		//bm=_feature_maps[i]<=thresh;
	//		//registerPosition(bm);
	//	}
	//}

	mBMSMap = fastBMS(mFeatureMaps);
	normalize(mBMSMap, mBMSMap, 0.0, 1.0, NORM_MINMAX);
	/*Mat intersection = cws.mul(mSaliencyMap);
	normalize(intersection, intersection, 0.0, 1.0, NORM_MINMAX);*/
	mSaliencyMap = mBMSMap + mCWSMap;
}


cv::Mat BMS::getAttentionMap(const cv::Mat& bm) 
{
	Mat ret = bm.clone();

	int jump;

	/*Scalar sumWhole = sum(bm);
	Scalar sumCenter = sum(Mat(bm, 
		Rect(Point(BORDER_MARGIN, BORDER_MARGIN), Point(bm.cols-BORDER_MARGIN-1, bm.rows-BORDER_MARGIN-1))));

	double sumBorder = (sumWhole[0]-sumCenter[0])/255.0;
	double areaBorder = bm.rows*bm.cols - (bm.rows - 2*BORDER_MARGIN)*(bm.cols - 2*BORDER_MARGIN);
	double white = pow(sumBorder,2.0);
	double black = pow(areaBorder - sumBorder,2.0);
	double whiteValue = black > white ? (black / (white + black) - 0.5) * 1 + 1 : 1;
	double blackValue = white > black ? (white / (white + black) - 0.5) * 1 + 1 : 1;*/

	Point seed;
	for (int i=0;i<bm.rows;i++)
	{
		jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
		seed = Point(0 + jump, i);
		if (ret.at<uchar>(seed) != 1)
		{
			floodFill(ret, seed, Scalar(1.0), 0, Scalar(0), Scalar(0), 4);
		}
		
			
		jump = BMS_RNG.uniform(0.0,1.0)>0.99 ?BMS_RNG.uniform(5,25):0;
		seed = Point(bm.cols - 1 - jump, i);
		if (ret.at<uchar>(seed) != 1)
		{
			floodFill(ret, seed, Scalar(1.0), 0, Scalar(0), Scalar(0), 4);
		}
		
	}
	for (int j=0;j<bm.cols;j++)
	{
		jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
		seed = Point(j, 0 + jump);
		if (ret.at<uchar>(seed) != 1)
		{
			floodFill(ret, seed, Scalar(1.0), 0, Scalar(0), Scalar(0), 4);
		}
		

		jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
		seed = Point(j, bm.rows - 1 - jump);
		if (ret.at<uchar>(seed) != 1)
		{
			floodFill(ret, seed, Scalar(1.0), 0, Scalar(0), Scalar(0), 4);
		}
	}


	
	//double max_, min_;
	//minMaxLoc(ret,&min_,&max_);
	//ret.setTo(Scalar(255.0), ret == 0);
	//imshow("bm", bm);
	//imshow("display", ret);
	//waitKey();
	ret = ret != 1;

	ret.convertTo(ret,CV_32FC1);

	return ret;
}

Mat BMS::getSaliencyMap(const Mat& disMap)
{
	Mat ret; 

	normalize(mSaliencyMap, ret, 0.0, 1.0, NORM_MINMAX);
	Mat _disMap;
	resize(disMap, _disMap, mSaliencyMap.size());
	mSaliencyMap = mSaliencyMap.mul(_disMap);
	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);
	ret.convertTo(ret,CV_8UC1);
	return ret;
}

Mat BMS::getSaliencyMap()
{
	Mat ret;

	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

void BMS::whitenFeatMap(float reg)
{
	assert(mSrc.channels() == 3);
	
	Mat srcF,meanF,covF;
	mSrc.convertTo(srcF, CV_32FC3);
	Mat samples = srcF.reshape(1, mSrc.rows*mSrc.cols);
	calcCovarMatrix(samples, covF, meanF, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F);

	covF += Mat::eye(covF.rows, covF.cols, CV_32FC1)*reg;
	SVD svd(covF);
	Mat sqrtW;
	sqrt(svd.w,sqrtW);
	Mat sqrtInvCovF = svd.u * Mat::diag(1.0/sqrtW);

	srcF = srcF - Scalar(meanF.at<float>(0, 0), meanF.at<float>(0, 1), meanF.at<float>(0, 2));
	Mat whitenedSrc = srcF.reshape(1, mSrc.rows*mSrc.cols)*sqrtInvCovF;
	whitenedSrc = whitenedSrc.reshape(3, mSrc.rows);
	//whitenedSrc.convertTo(whitenedSrc, CV_8U, 64.0, 127);
	split(whitenedSrc, mFeatureMaps);

	for (int i = 0; i < mFeatureMaps.size(); i++)
	{
		normalize(mFeatureMaps[i], mFeatureMaps[i], 255.0, 0.0, NORM_MINMAX);
		mFeatureMaps[i].convertTo(mFeatureMaps[i], CV_8U);
		medianBlur(mFeatureMaps[i], mFeatureMaps[i], 5);
	}
}

void BMS::computeBorderPriorMap(float reg, float marginRatio)
{
	assert(mSrc.channels() == 3);

	vector<Mat> sampleVec(4);
	Mat srcF;

	mSrc.convertTo(srcF, CV_64FC3);
	int rowMargin = (int)(marginRatio*mSrc.rows);
	int colMargin = (int)(marginRatio*mSrc.cols);
	
	sampleVec[0] = Mat(srcF,Range(0,rowMargin)).clone();
	sampleVec[1] = Mat(srcF,Range(mSrc.rows-rowMargin,mSrc.rows)).clone();
	sampleVec[2] = Mat(srcF,Range::all(),Range(0,colMargin)).clone();
	sampleVec[3] = Mat(srcF,Range::all(),Range(mSrc.cols-colMargin,mSrc.cols)).clone();

	for (int i = 0; i < 4; i++)
	{
		Mat meanF, covF;
		Mat samples = sampleVec[i].reshape(1, sampleVec[i].rows*sampleVec[i].cols);
		calcCovarMatrix(samples, covF, meanF, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);

		covF += Mat::eye(covF.rows, covF.cols, CV_64FC1)*reg;
		

		Mat srcFTemp = srcF - Scalar(meanF.at<double>(0, 0), meanF.at<double>(0, 1), meanF.at<double>(0, 2));
		srcFTemp = srcFTemp.reshape(1, mSrc.rows*mSrc.cols);
		Mat whitenedSrc = (srcFTemp*covF.inv()).mul(srcFTemp);
		whitenedSrc.convertTo(whitenedSrc, CV_32FC1);
		reduce(whitenedSrc, whitenedSrc, 1, CV_REDUCE_SUM);
		normalize(whitenedSrc.reshape(1, mSrc.rows), whitenedSrc, 1.0, 0.0, NORM_MINMAX);
		mBorderPriorMap += whitenedSrc;
	}
	normalize(mBorderPriorMap, mBorderPriorMap, 1.0, 0.0, NORM_MINMAX);
}

#ifdef USE_IPP
IppStatus ippiDilate32fWrapper(const Mat src, Mat& dst, int kernelWidth)
{
	int step = (int)src.cols*sizeof(float);
	Ipp32f *pSrc = (Ipp32f*)src.data, *pDst = (Ipp32f*)dst.data;
	IppStatus status;
	IppiSize roiSize = { src.cols, src.rows };
	IppiSize maskSize = { kernelWidth, kernelWidth };

	Mat kernel = Mat::ones(kernelWidth, kernelWidth, CV_8UC1);
	Ipp8u* pMask = kernel.data;
	IppiPoint anchor = { kernelWidth / 2, kernelWidth / 2 };

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



IppStatus ippiRecDilate8uWrapper(const Mat& src, Mat& srcDst, int kernelWidth)
{
	int step = (int)src.cols*sizeof(char), buffSize;
	Ipp8u *pSrc = (Ipp8u*)src.data, *pSrcDst = (Ipp8u*)srcDst.data;
	IppStatus status;
	IppiSize roiSize = { src.cols, src.rows };
	IppiSize maskSize = { kernelWidth, kernelWidth };
	Ipp8u *pBuff = NULL;

	status = ippiMorphReconstructGetBufferSize_8u_C1(roiSize, &buffSize);
	if (status != ippStsNoErr) return status;

	pBuff = ippsMalloc_8u(buffSize);
	status = ippiMorphReconstructDilate_8u_C1IR(pSrc, step, pSrcDst, step, roiSize, (Ipp8u*)pBuff, ippiNormL1);

	ippsFree(pBuff);
	return status;
}
IppStatus ippiRecErode8uWrapper(const Mat& src, Mat& srcDst, int kernelWidth)
{
	int step = (int)src.cols*sizeof(char), buffSize;
	Ipp8u *pSrc = (Ipp8u*)src.data, *pSrcDst = (Ipp8u*)srcDst.data;
	IppStatus status;
	IppiSize roiSize = { src.cols, src.rows };
	IppiSize maskSize = { kernelWidth, kernelWidth };
	Ipp8u *pBuff = NULL;

	status = ippiMorphReconstructGetBufferSize_8u_C1(roiSize, &buffSize);
	if (status != ippStsNoErr) return status;

	pBuff = ippsMalloc_8u(buffSize);
	status = ippiMorphReconstructErode_8u_C1IR(pSrc, step, pSrcDst, step, roiSize, (Ipp8u*)pBuff, ippiNormL1);

	ippsFree(pBuff);
	return status;
}
#endif

void postProcessByRec(cv::Mat& salmap, int kernelWidth)
{
	assert(salmap.type() == CV_32FC1);
#ifdef USE_IPP
	Mat temp(salmap.size(), CV_32FC1);
	IppStatus status;

	/*status = ippiErode32fWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Erosion: " << ippGetStatusString(status) << endl;*/
	erode(salmap, temp, Mat(), Point(-1, -1), kernelWidth / 2);

	status = ippiRecDilateWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: DilationRec: " << ippGetStatusString(status) << endl;

	/*status = ippiDilate32fWrapper(temp, salmap, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Dilation: " << ippGetStatusString(status) << endl;*/
	dilate(temp, salmap, Mat(), Point(-1, -1), kernelWidth / 2);

	status = ippiRecErodeWrapper(temp, salmap, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: ErosionRec: " << ippGetStatusString(status) << endl;
#else
	cerr << "IPP Not enabled." << endl;
#endif
}

void postProcessByRec8u(cv::Mat& salmap, int kernelWidth, double thresh)
{
	assert(salmap.type() == CV_8UC1);
#ifdef USE_IPP
	Mat temp(salmap.size(), CV_8UC1);
	IppStatus status;

	/*status = ippiErode32fWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Erosion: " << ippGetStatusString(status) << endl;*/
	erode(salmap, temp, Mat(), Point(-1, -1), kernelWidth / 2);

	// do better work with the seed map
	if (thresh > 0)
	{
		Mat maskTop = salmap < thresh;
		temp.setTo(Scalar(0.0), maskTop);
	}


	status = ippiRecDilate8uWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: DilationRec: " << ippGetStatusString(status) << endl;

	/*status = ippiDilate32fWrapper(temp, salmap, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Dilation: " << ippGetStatusString(status) << endl;*/
	dilate(temp, salmap, Mat(), Point(-1, -1), kernelWidth / 2);

	status = ippiRecErode8uWrapper(temp, salmap, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: ErosionRec: " << ippGetStatusString(status) << endl;
#else
	cerr<<"IPP Not enabled."<<endl;
#endif
}


void postProcessByRec8u(cv::Mat& salmap, int kernelWidth, const cv::Mat& disMat)
{
	assert(salmap.type() == CV_8UC1);

	Mat _disMat;
	resize(disMat, _disMat, salmap.size());
#ifdef USE_IPP
	Mat temp(salmap.size(), CV_8UC1);
	IppStatus status;

	/*status = ippiErode32fWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Erosion: " << ippGetStatusString(status) << endl;*/
	erode(salmap, temp, Mat(), Point(-1, -1), kernelWidth / 2);
	temp.convertTo(temp, CV_32FC1);
	temp = temp.mul(_disMat);
	temp.convertTo(temp, CV_8UC1);

	status = ippiRecDilate8uWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: DilationRec: " << ippGetStatusString(status) << endl;

	/*status = ippiDilate32fWrapper(temp, salmap, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Dilation: " << ippGetStatusString(status) << endl;*/
	dilate(temp, salmap, Mat(), Point(-1, -1), kernelWidth / 2);

	status = ippiRecErode8uWrapper(temp, salmap, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: ErosionRec: " << ippGetStatusString(status) << endl;
#else
	cerr << "IPP Not enabled." << endl;
#endif
}

void doCluster(const Mat& distMat, double thresh, std::vector<std::vector<int>>& clusters)
{
	clusters.clear();
	int nmel = distMat.rows;
	Mat mask = Mat::zeros (1, distMat.rows, CV_8UC1);
	for (int i = 0; i < nmel; i++)
	{
		if (mask.at<uchar>(0,i) == 1)
			continue;
		vector<int> newCluster(1, i);
		mask.at<uchar>(0, i) = 1;
		int npos = 0;
		while (npos < newCluster.size())
		{
			for (int j = i + 1; j < nmel; j++)
			{
				if (mask.at<uchar>(0, j) == 0 && distMat.at<float>(newCluster[npos], j) < thresh)
				{
					newCluster.push_back(j);
					mask.at<char>(0, j) = 1;
				}
			}
			npos++;
		}
		clusters.push_back(newCluster);
	}

	// print
	/*cout << distMat << endl;
	for (int i = 0; i < clusters.size(); i++)
	{
		for (int j = 0; j < clusters[i].size(); j++)
		{
			cout << clusters[i][j];
		}
		cout << endl;
	}*/
}

void rasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub)
{
	Size sz = featMap.size();
	float *pMapup = (float*)map.data + 1;
	float *pMap = pMapup + sz.width;
	uchar *pFeatup = featMap.data + 1;
	uchar *pFeat = pFeatup + sz.width;
	uchar *pLBup = lb.data + 1;
	uchar *pLB = pLBup + sz.width;
	uchar *pUBup = ub.data + 1;
	uchar *pUB = pUBup + sz.width;

	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float lfV, upV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap - 1);
		featPrev = *(pFeat - 1);
		lbPrev = *(pLB - 1);
		ubPrev = *(pUB - 1);


		for (int c = 1; c < sz.width - 1; c++)
		{
			lfV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			upV = MAX(*pFeat, *pUBup) - MIN(*pFeat, *pLBup);//(*pFeat >= *pLBup && *pFeat <= *pUBup) ? *pMapup : *pMapup + abs((float)(*pFeat) - (float)(*pFeatup));

			flag = 0;
			if (lfV < *pMap)
			{
				*pMap = lfV;
				flag = 1;
			}
			if (upV < *pMap)
			{
				*pMap = upV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from left
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from up
				*pLB = MIN(*pFeat, *pLBup);
				*pUB = MAX(*pFeat, *pUBup);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap++; pMapup++;
			featPrev = *pFeat;
			pFeat++; pFeatup++;
			lbPrev = *pLB;
			pLB++; pLBup++;
			ubPrev = *pUB;
			pUB++; pUBup++;
		}
		pMapup += 2; pMap += 2;
		pFeat += 2; pFeatup += 2;
		pLBup += 2; pLB += 2;
		pUBup += 2; pUB += 2;
	}
}

void invRasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub)
{
	Size sz = featMap.size();
	int datalen = sz.width*sz.height;
	float *pMapdn = (float*)map.data + datalen - 2;
	float *pMap = pMapdn - sz.width;
	uchar *pFeatdn = featMap.data + datalen - 2;
	uchar *pFeat = pFeatdn - sz.width;
	uchar *pLBdn = lb.data + datalen - 2;
	uchar *pLB = pLBdn - sz.width;
	uchar *pUBdn = ub.data + datalen - 2;
	uchar *pUB = pUBdn - sz.width;
	
	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float rtV, dnV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap + 1);
		featPrev = *(pFeat + 1);
		lbPrev = *(pLB + 1);
		ubPrev = *(pUB + 1);

		for (int c = 1; c < sz.width - 1; c++)
		{
			rtV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			dnV = MAX(*pFeat, *pUBdn) - MIN(*pFeat, *pLBdn);//(*pFeat >= *pLBdn && *pFeat <= *pUBdn) ? *pMapdn : *pMapdn + abs((float)(*pFeat) - (float)(*pFeatdn));

			flag = 0;
			if (rtV < *pMap)
			{
				*pMap = rtV;
				flag = 1;
			}
			if (dnV < *pMap)
			{
				*pMap = dnV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from right
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from down
				*pLB = MIN(*pFeat, *pLBdn);
				*pUB = MAX(*pFeat, *pUBdn);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap--; pMapdn--;
			featPrev = *pFeat;
			pFeat--; pFeatdn--;
			lbPrev = *pLB;
			pLB--; pLBdn--;
			ubPrev = *pUB;
			pUB--; pUBdn--;
		}


		pMapdn -= 2; pMap -= 2;
		pFeatdn -= 2; pFeat -= 2;
		pLBdn -= 2; pLB -= 2;
		pUBdn -= 2; pUB -= 2;
	}
}

cv::Mat fastBMS(const std::vector<cv::Mat> featureMaps)
{
	assert(featureMaps[0].type() == CV_8UC1);

	Size sz = featureMaps[0].size();
	Mat ret = Mat::zeros(sz, CV_32FC1);
	if (sz.width < 3 || sz.height < 3)
		return ret;

	for (int i = 0; i < featureMaps.size(); i++)
	{
		Mat map = Mat::zeros(sz, CV_32FC1);
		Mat mapROI(map, Rect(1, 1, sz.width - 2, sz.height - 2));
		mapROI.setTo(Scalar(100000));
		Mat lb = featureMaps[i].clone();
		Mat ub = featureMaps[i].clone();

		rasterScan(featureMaps[i], map, lb, ub);
		invRasterScan(featureMaps[i], map, lb, ub);
		rasterScan(featureMaps[i], map, lb, ub);
		
		ret += map;
	}

	return ret;
	
}

int findFrameMargin(const Mat& img, bool reverse)
{
	Mat edgeMap, edgeMapDil, edgeMask;
	Sobel(img, edgeMap, CV_16SC1, 0, 1);
	edgeMap = abs(edgeMap);
	edgeMap.convertTo(edgeMap, CV_8UC1);
	edgeMask = edgeMap < (SOBEL_THRESH * 255.0);
	dilate(edgeMap, edgeMapDil, Mat(), Point(-1, -1), 2);
	edgeMap = edgeMap == edgeMapDil;
	edgeMap.setTo(Scalar(0.0), edgeMask);


	if (!reverse)
	{
		for (int i = edgeMap.rows - 1; i >= 0; i--)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return i + 1;
	}
	else
	{
		for (int i = 0; i < edgeMap.rows; i++)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return edgeMap.rows - i;
	}

	return 0;
}

bool removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi)
{
	if (inImg.rows < 2 * (FRAME_MAX + 3) || inImg.cols < 2 * (FRAME_MAX + 3))
	{
		roi = Rect(0, 0, inImg.cols, inImg.rows);
		outImg = inImg;
		return false;
	}

	Mat imgGray;
	cvtColor(inImg, imgGray, CV_RGB2GRAY);
	// fast rejection
	/*if (abs(mean(imgGray.row(0))[0] - mean(imgGray.row(imgGray.rows - 1))[0]) > TOLERANCE * 255.0)
	{
		roi = Rect(0, 0, imgGray.cols, imgGray.rows);
		outImg = inImg;
		return false;
	}*/

	int up, dn, lf, rt;
	
	up = findFrameMargin(imgGray.rowRange(0, FRAME_MAX), false);
	dn = findFrameMargin(imgGray.rowRange(imgGray.rows - FRAME_MAX, imgGray.rows), true);
	lf = findFrameMargin(imgGray.colRange(0, FRAME_MAX).t(), false);
	rt = findFrameMargin(imgGray.colRange(imgGray.cols - FRAME_MAX, imgGray.cols).t(), true);

	int margin = MAX(up, MAX(dn, MAX(lf, rt)));
	if ( margin == 0 )
	{
		roi = Rect(0, 0, imgGray.cols, imgGray.rows);
		outImg = inImg;
		return false;
	}

	int count = 0;
	count = up == 0 ? count : count + 1;
	count = dn == 0 ? count : count + 1;
	count = lf == 0 ? count : count + 1;
	count = rt == 0 ? count : count + 1;

	// cut four border region if at least 2 border frames are detected
	if (count > 1)
	{
		margin += 2;
		roi = Rect(margin, margin, inImg.cols - 2*margin, inImg.rows - 2*margin);
		outImg = Mat(inImg, roi);

		return true;
	}

	// otherwise, cut only one border
	up = up == 0 ? up : up + 2;
	dn = dn == 0 ? dn : dn + 2;
	lf = lf == 0 ? lf : lf + 2;
	rt = rt == 0 ? rt : rt + 2;

	
	roi = Rect(lf, up, inImg.cols - lf - rt, inImg.rows - up - dn);
	outImg = Mat(inImg, roi);

	return true;
	
}



void getTrainData(const cv::Mat& img, const cv::Mat& ref, int nBin, cv::Mat& X, cv::Mat& Y, cv::Mat& W)
{
	int histSize[] = {nBin};
	float range[] = {0,1.0};
	const float* ranges[] = {range};
	int channels[] = {0};

	Mat histPos, histNeg;
	calcHist(&img, 1, channels, ref > 127.0, histPos, 1, histSize, ranges);
	calcHist(&img, 1, channels, ref <= 127.0, histNeg, 1, histSize, ranges);
	histPos = histPos / sum(histPos)[0];
	histNeg = histNeg / sum(histNeg)[0];

	X = Mat::zeros(2*nBin, 1, CV_32FC1);
	Y = Mat::zeros(2*nBin, 1, CV_32FC1);
	W = Mat::zeros(2*nBin, 1, CV_32FC1);

	float xStep = 1.0f / nBin;
	float *pX = (float*)X.data, *pY = (float*)Y.data, *pW = (float*)W.data;
	for (int i = 0; i < nBin; i++, pX++, pY++, pW++)
	{
		*pX = xStep*(i + 0.5f);
		*pY = 1.0f;
		*pW = histPos.at<float>(i, 0);
	}
	for (int i = 0; i < nBin; i++, pX++, pY++, pW++)
	{
		*pX = xStep*(i + 0.5f);
		*pY = 0.0f;
		*pW = histNeg.at<float>(i, 0);
	}
}

void getLRParam(const cv::Mat& X, const cv::Mat& Y, const cv::Mat& W, float& a, float& b)
{
	Mat beta = (Mat_<float>(2,1) << 0.0f, 0.0f), betaOld;
	Mat temp, err, p, s = Mat::zeros(2, 1, CV_32FC1), J = Mat::zeros(2, 2, CV_32FC1);
	float lambda = 0.00001f;
	Mat reg = (Mat_<float>(2,2) << 0.0f, 0.0f, 0.0f, sum(W)[0]*lambda);
	double diff = 1.0f;
	int count = 0;
	while (diff > 0.0001f && count < 10)
	{
		betaOld = beta.clone();
		exp(-(beta.at<float>(1,0)*X + beta.at<float>(0,0)), p);
		p = 1.0 / (1.0 + p);

		err = Y - p;
		temp = W.t()*err;
		s.at<float>(0, 0) = temp.at<float>(0, 0);
		temp = W.t()*(err.mul(X) - lambda*beta.at<float>(1,0));
		s.at<float>(1, 0) = temp.at<float>(0, 0);

		err = p.mul(1.0 - p);
		temp = W.t()*err;
		J.at<float>(0, 0) = temp.at<float>(0, 0);
		temp = W.t()*err.mul(X);
		J.at<float>(0, 1) = temp.at<float>(0, 0);
		J.at<float>(1, 0) = temp.at<float>(0, 0);
		temp = W.t()*err.mul(X).mul(X);
		J.at<float>(1, 1) = temp.at<float>(0, 0);

		J += reg;

		beta = betaOld + J.inv()*s;
		
		diff = sum(abs(beta - betaOld))[0];
		count++;
	}

	a = beta.at<float>(1,0);
	b = beta.at<float>(0,0);
}