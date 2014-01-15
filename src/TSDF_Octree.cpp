/*
Copyright (C) 2014 Steven Hickson

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

*/
// TestVideoSegmentation.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <cpu_tsdf/tsdf_volume_octree.h>
#include <cpu_tsdf/marching_cubes_tsdf_octree.h>

#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/segmentation/extract_clusters.h>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <string>
#include <vector>

#include "Microsoft_grabber.h"
#include <pcl/visualization/cloud_viewer.h>
/*#include <FaceTrackLib.h>
#include <KinectInteraction.h>
#include <NuiKinectFusionApi.h>
#include <NuiKinectFusionDepthProcessor.h>
#include <NuiKinectFusionVolume.h>*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace pcl;
using namespace cv;
using namespace cpu_tsdf;

inline int Round (float a)  
{
	assert( !_isnan( a ) );
	//    int b;
	//    _asm
	//    { // assumes that we are in 'round mode', which is the default; should probably check
	//      FLD a   ; load floating-point value
	//      FIST b  ; store integer
	//    };
	//#ifndef NDEBUG
	//    int c = static_cast<int>(a>=0 ? a+0.5f : a-0.5f);
	//#endif
	//    assert( b == static_cast<int>(a>=0 ? a+0.5f : a-0.5f) );
	//    return b;   
	return static_cast<int>(a>=0 ? a+0.5f : a-0.5f); 
} 

void CreatePointCloudFromRegisteredData(const Mat &img, const Mat &depth, PointCloud<PointXYZRGBA>::Ptr &cloud) {
	assert(!img.empty() && !depth.empty());
	//take care of old cloud to prevent memory leak/corruption
	PointCloud<PointXYZRGBA>::iterator pCloud = cloud->begin();
	Mat_<int>::const_iterator pDepth = depth.begin<int>();
	Mat_<Vec3b>::const_iterator pImg = img.begin<Vec3b>();
	for(int j = 0; j < cloud->height; j++) {
		for(int i = 0; i < cloud->width; i++) {
			PointXYZRGBA loc;
			loc.z = *pDepth / 1000.0f;
			loc.x = float((i - 319.5f) * loc.z / 525.0f);
			loc.y = float((j - 239.5f) * loc.z / 525.0f);
			loc.b = (*pImg)[0];
			loc.g = (*pImg)[1];
			loc.r = (*pImg)[2];
			loc.a = 255;
			*pCloud = loc;
			pImg++; pDepth++; pCloud++;
		}
	}
}

int main (int argc, char** argv) {
	try {
		vector<Eigen::Affine3d > poses;
		visualization::PCLVisualizer viewer("Cloud Viewer");

		TSDFVolumeOctree::Ptr tsdf (new TSDFVolumeOctree);
		tsdf->setGridSize (10., 10., 10.); // 10m x 10m x 10m
		tsdf->setResolution (2048, 2048, 2048); // Smallest cell size = 10m / 2048 = about half a centimeter
		Eigen::Affine3d tsdf_center; // Optionally offset the center
		tsdf->setGlobalTransform (tsdf_center);
		tsdf->setIntegrateColor(true);
		tsdf->reset (); // Initialize it to be empty

		int i = 0, N = atoi(argv[3]);
		string inFolder = string(argv[1]);
		string orderFile = string(argv[2]);
		ifstream file(inFolder + orderFile);
		string line, it1, rgbFile, it2, depthFile;
		while(getline(file, line) && i < N) {
			stringstream linestream(line);
			linestream >> it1 >> depthFile >> it2 >> rgbFile;
			Mat image, depth;
			//image = cvLoadImage(string(inFolder + rgbFile).c_str());
			image = cv::imread(string(inFolder + rgbFile));
			depth = cvLoadImage(string(inFolder + depthFile).c_str(),CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
			depth.convertTo(depth, 4);
			int* pOut = (int*)depth.data;
			for (int y=0; y < image.rows; y++) {
				for (int x=0; x < image.cols; x++) {
					*pOut = Round(*pOut * 0.2f);
					++pOut;
				}
			}
			/*imshow("image", image);
			imshow("depth", depth);
			cvWaitKey();*/
			PointCloud<PointXYZRGBA>::Ptr cloud(new PointCloud<PointXYZRGBA>);

			cloud->header.frame_id =  "/microsoft_rgb_optical_frame";
			cloud->height = 640;
			cloud->width = 480;
			cloud->is_dense = false;
			cloud->points.resize (cloud->height * cloud->width);
			CreatePointCloudFromRegisteredData(image,depth,cloud);
			/*viewer.removePointCloud("cloud");
			viewer.addPointCloud(cloud);*/
			pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

			pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
			ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
			ne.setMaxDepthChangeFactor(0.02f);
			ne.setNormalSmoothingSize(10.0f);
			ne.setInputCloud(cloud);
			ne.compute(*normals);
			tsdf->integrateCloud (*cloud, *normals); // Integrate the cloud
			// Note, the normals aren't being used in the default settings. Feel free to pass in an empty cloud
			cout << "Finished Number " << i << endl;
			i++;
		}
		// Now what do you want to do with it? 
		//float distance; pcl::PointXYZ query_point (1.0, 2.0, -1.0);
		//tsdf->getFxn (query_point, distance); // distance is normalized by the truncation limit -- goes from -1 to 1
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr raytraced = tsdf->renderColoredView(); // Optionally can render it
		//tsdf->save ("output.vol"); // Save it?  
		// Mesh with marching cubes
		MarchingCubesTSDFOctree mc;
		mc.setInputTSDF (tsdf);
		pcl::PolygonMesh mesh;
		mc.reconstruct (mesh);
		viewer.addPointCloud<PointXYZRGBNormal>(raytraced);
		while(!viewer.wasStopped())
			viewer.spinOnce();
		pcl::io::savePolygonFilePLY("test.ply",mesh);


	} catch (pcl::PCLException e) {
		cout << e.detailedMessage() << endl;
	} catch (std::exception &e) {
		cout << e.what() << endl;
	}
	cin.get();
	return 0;
}