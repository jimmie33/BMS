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

#ifndef BMS_H
#define BMS_H

#ifdef IMDEBUG
#include <imdebug.h>
#endif
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#ifdef USE_IPP
#include <ipp.h>
#endif
#ifdef IMDEBUG
#include "imdebug.h"
#endif


static cv::RNG BMS_RNG;

class BMS
{
public:
	BMS (const cv::Mat& src);
	cv::Mat getSaliencyMap(const cv::Mat& disMat);
	cv::Mat getSaliencyMap();
	void computeSaliency(bool use_cws);
	cv::Mat getBMSMap() const { return mBMSMap; }
	cv::Mat getCWSMap() const { return mCWSMap; }
private:
	cv::Mat mSaliencyMap;
	cv::Mat mBMSMap;
	cv::Mat mCWSMap;
	int mAttMapCount;
	cv::Mat mBorderPriorMap;
	cv::Mat mSrc;
	std::vector<cv::Mat> mFeatureMaps;
	cv::Mat getAttentionMap(const cv::Mat& bm);
	void whitenFeatMap(float reg);
	void computeBorderPriorMap(float reg, float marginRatio);
};

cv::Mat computeCWS(const cv::Mat src, float reg, float marginRatio);
cv::Mat fastBMS(const std::vector<cv::Mat> featureMaps);
cv::Mat fastGeodesic(const std::vector<cv::Mat> featureMaps);

int findFrameMargin(const cv::Mat& img, bool reverse);
bool removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi);

void postProcessByRec8u(cv::Mat& salmap, int kernelWidth, double thresh);
void postProcessByRec8u(cv::Mat& salmap, int kernelWidth, const cv::Mat& disMat);
void postProcessByRec(cv::Mat& salmap, int kernelWidth);

void doCluster(const cv::Mat& distMat, double thresh, std::vector<std::vector<int>>& clusters);

void getTrainData(const cv::Mat& img, const cv::Mat& ref, int nBin, cv::Mat& X, cv::Mat& Y, cv::Mat& W);
void getLRParam(const cv::Mat& X, const cv::Mat& Y, const cv::Mat& W, float& a, float& b);



#endif


