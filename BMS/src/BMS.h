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



static cv::RNG BMS_RNG;

class BMS
{
public:
	BMS (const cv::Mat& src, const int dw1, const int ow, const bool nm, const bool hb);
	cv::Mat getSaliencyMap();
	void computeSaliency(double step);
private:
	cv::Mat mSaliencyMap;
	int mAttMapCount;
	cv::Mat mBorderPriorMap;
	cv::Mat mSrc;
	std::vector<cv::Mat> mFeatureMaps;
	int mDilationWidth_1;
	int mOpeningWidth;
	bool mHandleBorder;
	bool mNormalize;
	cv::Mat getAttentionMap(const cv::Mat& bm, int dilation_width_1, bool toNormalize, bool handle_border);
	void whitenFeatMap(float reg);
	void computeBorderPriorMap(float reg, float marginRatio);
};



#endif


