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

Mat computeCWS(const Mat src, float reg, float marginRatio)
{
	assert(mSrc.channels() == 3);

	vector<Mat> sampleVec(8);
	vector<Mat> means(4);
	vector<Mat> covs(4);
	vector<double> pixNum(4);
	vector<Mat> clusterMeans;
	vector<Mat> clusterCovs;

	Mat srcF;
	Mat ret(src.size(), CV_32FC1);

	src.convertTo(srcF, CV_32FC3);
	int rowMargin = (int)(marginRatio*src.rows);
	int colMargin = (int)(marginRatio*src.cols);

	sampleVec[0] = Mat(srcF, Range(0, rowMargin), Range(0, src.cols/2)).clone();
	sampleVec[1] = Mat(srcF, Range(0, src.rows/2), Range(0, colMargin)).clone();

	sampleVec[2] = Mat(srcF, Range(0, rowMargin), Range(src.cols/2, src.cols)).clone();
	sampleVec[3] = Mat(srcF, Range(0, src.rows / 2), Range(src.cols - colMargin, src.cols)).clone();

	sampleVec[4] = Mat(srcF, Range(src.rows - rowMargin, src.rows), Range(0, src.cols/2)).clone();
	sampleVec[5] = Mat(srcF, Range(src.rows/2, src.rows), Range(0, colMargin)).clone();

	sampleVec[6] = Mat(srcF, Range(src.rows - rowMargin, src.rows), Range(src.cols/2, src.cols)).clone();
	sampleVec[7] = Mat(srcF, Range(src.rows / 2, src.rows), Range(src.cols - colMargin, src.cols)).clone();

	Mat maxMap(src.size(), CV_32FC1);

	for (int i = 0; i < 4; i++)
	{
		Mat samples;
		vconcat(sampleVec[2*i].reshape(1, sampleVec[2*i].rows*sampleVec[2*i].cols), 
			sampleVec[2*i+1].reshape(1, sampleVec[2*i+1].rows*sampleVec[2*i+1].cols),
			samples);
		calcCovarMatrix(samples, covs[i], means[i], CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F);
		pixNum[i] = samples.rows;
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

	normalize(ret, ret, 0.0, 1.0, NORM_MINMAX);
	//ret.convertTo(ret, CV_8U);
	return ret;
}

BMS::BMS(const Mat& src)
:mAttMapCount(0)
{
	mSrc=src.clone();
	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);
	mBorderPriorMap = Mat::zeros(src.size(), CV_32FC1);

	whitenFeatMap(COV_MAT_REG);
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
	Mat cws = computeCWS(mSrc, 50.0f, 0.1f);
	for (int i=0;i<mFeatureMaps.size();++i)
	{
		Mat bm;
		double max_,min_;
		minMaxLoc(mFeatureMaps[i],&min_,&max_);
		//step = (max_ - min_) / 30.0f;
		for (double thresh = 0; thresh < 255; thresh += step)
		{
			bm=mFeatureMaps[i]>thresh;
			Mat am = getAttentionMap(bm);
			mSaliencyMap += am;
			mAttMapCount++;
			//bm=_feature_maps[i]<=thresh;
			//registerPosition(bm);
		}
	}

	normalize(mSaliencyMap, mSaliencyMap, 0.0, 1.0, NORM_MINMAX);
	/*Mat intersection = cws.mul(mSaliencyMap);
	normalize(intersection, intersection, 0.0, 1.0, NORM_MINMAX);*/
	mSaliencyMap =  cws + mSaliencyMap;
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

Mat BMS::getSaliencyMap()
{
	Mat a,b,ret; 

	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);
	ret.convertTo(ret,CV_8UC1);
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

void postProcessByRec8u(cv::Mat& salmap, int kernelWidth)
{
	assert(salmap.type() == CV_8UC1);
#ifdef USE_IPP
	Mat temp(salmap.size(), CV_8UC1);
	IppStatus status;

	/*status = ippiErode32fWrapper(salmap, temp, kernelWidth);
	if (status != ippStsNoErr) cerr << "postProcessByRec: Erosion: " << ippGetStatusString(status) << endl;*/
	erode(salmap, temp, Mat(), Point(-1, -1), kernelWidth / 2);

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