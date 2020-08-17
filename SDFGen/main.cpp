#include <iostream>
#include <tclap/CmdLine.h>
#include "obj/ObjReader.h"
#include "container/Array3D.h"
#include "container/Space.h"
#include "test/PolygonTest.h"
#include "util/Hdf5Writer.h"
#include <mutex>

struct CamPoses {

	dPoint camPos;
	dPoint towardsPose;
	dPoint upPos;
	double xFov;
	double yFov;

};

std::vector<CamPoses> readCameraPoses(std::string cameraPositionsFile) {
    std::fstream positions_file(cameraPositionsFile, std::istream::in);
    std::vector<CamPoses> camPoses;
    if(positions_file.is_open()){
        std::string line;
        unsigned int currentIndex = 0;
        while(positions_file.good()){
            std::getline(positions_file, line); // skip this line
            if(line.length() > 1){
                CamPoses pose;
                unsigned int eleCamPos = 0;
                std::string elementString;
                std::istringstream stringStreamEle(line);
                while(getline(stringStreamEle, elementString, ' ') && eleCamPos < 11){
                    if(elementString.length() <= 1){
                        continue;
                    }
                    const double ele = std::atof(elementString.c_str());
                    if(eleCamPos < 3){
                        pose.camPos[eleCamPos] = ele;
                    }else if(eleCamPos >= 3 && eleCamPos < 6){
                        pose.towardsPose[eleCamPos - 3] = ele;
                    }else if(eleCamPos >= 6 && eleCamPos < 9){
                        pose.upPos[eleCamPos - 6] = ele;
                    }else if(eleCamPos == 9){
                        pose.xFov = ele;
                    }else if(eleCamPos == 10){
                        pose.yFov = ele;
                    }
                    ++eleCamPos;
                }
                camPoses.emplace_back(pose);
                ++currentIndex;
            }
        }
    }else{
        printError("The position file could not be opened: " + cameraPositionsFile);
    }
    return camPoses;
}


std::mutex readInMutex;

Polygons readInAndPreparePolygons(const CamPoses& camPose, std::string objFile, double scaling, double projectionNearClipping, double projectionFarClipping, double nearClipping, double farClipping, double surfacelessPolygonsThreshold) {
	readInMutex.lock();
	ObjReader reader;
	reader.read(objFile);

	Polygons polygons = reader.getPolygon();
	if (scaling != 1)
		scalePoints(polygons, scaling);
	printVars(camPose.camPos, camPose.towardsPose);
	dTransform camTrans;
	camTrans.setAsCameraTransTowards(camPose.camPos, camPose.towardsPose, camPose.upPos);

	dTransform projectionTrans;
	projectionTrans.setAsProjectionWith(camPose.xFov, camPose.yFov, projectionNearClipping, projectionFarClipping);

	transformPoints(polygons, camTrans);
	polygons = nearFarPolygonClipping(polygons, nearClipping, farClipping);

	transformPoints(polygons, projectionTrans, true);

	polygons = removePolygonsOutOfFrustum(polygons);
	polygons = frustumClipping(polygons);
	polygons = removeFlatPolygons(polygons, surfacelessPolygonsThreshold);
	writeToDisc(polygons, "/tmp/test.obj");
	readInMutex.unlock();
	return polygons;
}

void convertCamPoses(unsigned int start, unsigned int end, const std::vector<CamPoses>& camPoses, std::string objFile, std::string outputFolder, double scaling, double projectionNearClipping, double projectionFarClipping, double nearClipping, double farClipping, double surfacelessPolygonsThreshold, unsigned int spaceResolution, double minimumOctreeVoxelSize, double maxDistanceToMinPosDist, int boundaryWidth, unsigned int threads, bool useExactAlgorithm, int approximationAccuracy, double truncationThreshold) {
	for(int camPoseIndex = start; camPoseIndex < end; camPoseIndex++) {
		const CamPoses& camPose = camPoses[camPoseIndex];
		Polygons polygons(readInAndPreparePolygons(camPose, objFile, scaling, projectionNearClipping, projectionFarClipping, nearClipping, farClipping, surfacelessPolygonsThreshold));


		Space space({spaceResolution, spaceResolution, spaceResolution}, d_negOnes, d_ones * 2);
		if (useExactAlgorithm)
			space.calcDistsExactly(polygons, minimumOctreeVoxelSize, maxDistanceToMinPosDist, threads, truncationThreshold);
		else
			space.calcDistsApproximately(polygons, maxDistanceToMinPosDist, truncationThreshold, approximationAccuracy);

		space.correctInnerVoxel(boundaryWidth, truncationThreshold);

		const auto size = space.getData().getSize();
		Array3D<float> rotatedAndFlipped(size);
		for(unsigned int i = 0; i < size[0]; ++i){
			for(unsigned int j = 0; j < size[1]; ++j){
				for(unsigned int k = 0; k < size[2]; ++k){
					rotatedAndFlipped(i, j, size[2] - 1 - k) = space.getData()(j, size[1] - 1 - i,k);
				}
			}
		}

		// convert to 16 bit -> to save memory
		Array3D<unsigned short> compressedArray(size);
		for(unsigned int i = 0; i < size[0]; ++i){
			for(unsigned int j = 0; j < size[1]; ++j){
				for(unsigned int k = 0; k < size[2]; ++k){
					// clipping some of the values are below the negative trunc value
					const auto newVal = (std::max(-(float)truncationThreshold, rotatedAndFlipped(i,j,k)) + truncationThreshold) / (2 * truncationThreshold);
					compressedArray(i,j,k) = (unsigned short)(newVal * 65535.F);
				}
			}
		}
		const std::string outputFilePath = outputFolder + "/output_" + Utility::toString(camPoseIndex) + ".hdf5";
		Hdf5Writer::writeArrayToFile(outputFilePath, compressedArray);
		printMsg("Save output file!");
	}
}

int main(int argc, char** argv){

	//PolygonTest::testAll();
	TCLAP::CmdLine cmd("Generate Voxels passed on a list of camera postions and an .obj file", ' ', "1.0");
	const bool required = true;
	const bool notRequired = false;
	TCLAP::ValueArg<std::string> objFile("o", "obj", "File path to the obj file", required, "", "string");
	TCLAP::ValueArg<std::string> cameraPositionsFile("c", "cameraPosFile", "File path to camera position file", required, "", "string");
	TCLAP::ValueArg<std::string> outputFolder("f", "folder", "Folder path for output files", required, "", "string");
	TCLAP::ValueArg<double> minimumOctreeVoxelSize("d", "depth", "Minimum octree voxel size. Will be used to determie octree depth.", notRequired, 0.1, "double");
	TCLAP::ValueArg<int> boundaryWidth("b", "boundary", "Additional boundary width for inner voxel detection", notRequired, 2, "int");
	TCLAP::ValueArg<double> surfacelessPolygonsThreshold("t", "thres", "Threshold for detection of polygons with no surface", notRequired, 1e-4, "double");
	TCLAP::ValueArg<unsigned int> spaceResolution("r", "res", "Resolution of the voxel space", notRequired, 128, "int");
	TCLAP::ValueArg<double> scaling("s", "scale", "Polygon scaling", notRequired, 1, "double");
	TCLAP::ValueArg<double> farClipping("", "far", "Far clipping threshold used for removing outside polygons", notRequired, 4, "double");
	TCLAP::ValueArg<double> nearClipping("", "near", "Near clipping threshold used for removing outside polygons", notRequired, 1, "double");
	TCLAP::ValueArg<double> projectionFarClipping("", "proj_far", "Far clipping threshold used for building the projection matrix", notRequired, 4, "double");
	TCLAP::ValueArg<double> projectionNearClipping("", "proj_near", "Near clipping threshold used for building the projection matrix", notRequired, 1, "double");
	TCLAP::ValueArg<double> frustumBorder("", "frustum_bor", "Additional border around frustum when clipping polygons", notRequired, 0, "double");
	TCLAP::ValueArg<double> maxDistanceToMinPosDist("", "pos_threshold", "Maximum amount the minimum positive distance is allowed to be smaller than the minimum negative distance to a polygon, s.t. the voxel should be still positive.", notRequired, 4e-3, "double");
	TCLAP::ValueArg<unsigned int> threads("", "threads", "Number of threads to use", notRequired, 0, "int");
	TCLAP::ValueArg<bool> useExactAlgorithm("", "exact", "Use exact algorithm", notRequired, false, "bool");
	TCLAP::ValueArg<int> approximationAccuracy("", "accuracy", "When using the approximate algorithm, this determines the area around each voxel which is searched.", notRequired, 2, "int");
	TCLAP::ValueArg<double> truncationThreshold("", "trunc", "The truncation threshold to use", notRequired, 0.1, "double");

	cmd.add(objFile);
	cmd.add(cameraPositionsFile);
	cmd.add(outputFolder);
    cmd.add(minimumOctreeVoxelSize);
    cmd.add(boundaryWidth);
    cmd.add(surfacelessPolygonsThreshold);
    cmd.add(spaceResolution);
	cmd.add(scaling);
    cmd.add(farClipping);
    cmd.add(nearClipping);
    cmd.add(projectionFarClipping);
    cmd.add(projectionNearClipping);
    cmd.add(frustumBorder);
    cmd.add(maxDistanceToMinPosDist);
	cmd.add(threads);
	cmd.add(useExactAlgorithm);
	cmd.add(approximationAccuracy);
	cmd.add(truncationThreshold);
	cmd.parse(argc, argv);

    std::vector<CamPoses> camPoses = readCameraPoses(cameraPositionsFile.getValue());

	if (useExactAlgorithm.getValue()) {
		convertCamPoses(0, camPoses.size(), camPoses, objFile.getValue(), outputFolder.getValue(), scaling.getValue(), projectionNearClipping.getValue(), projectionFarClipping.getValue(), nearClipping.getValue(), farClipping.getValue(), surfacelessPolygonsThreshold.getValue(), spaceResolution.getValue(), minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(), boundaryWidth.getValue(), threads.getValue(), useExactAlgorithm.getValue(), approximationAccuracy.getValue(), truncationThreshold.getValue());
	} else {
		unsigned int amountOfThreads =threads.getValue();

		if (amountOfThreads == 0)
			amountOfThreads = std::thread::hardware_concurrency();
		std::vector<std::thread> threads;

		amountOfThreads = std::min(amountOfThreads, (unsigned int) camPoses.size());
		for (unsigned int i = 0; i < amountOfThreads; ++i) {
			unsigned int start = (unsigned int) (i * camPoses.size() / (float) amountOfThreads);
			unsigned int end = (unsigned int) ((i + 1) * camPoses.size() / (float) amountOfThreads);
			if (i + 1 == amountOfThreads) {
				end = camPoses.size();
			}
			threads.emplace_back(
					std::thread(&convertCamPoses, start, end, std::cref(camPoses), objFile.getValue(), outputFolder.getValue(), scaling.getValue(), projectionNearClipping.getValue(), projectionFarClipping.getValue(), nearClipping.getValue(), farClipping.getValue(), surfacelessPolygonsThreshold.getValue(), spaceResolution.getValue(), minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(), boundaryWidth.getValue(), 1, useExactAlgorithm.getValue(), approximationAccuracy.getValue(), truncationThreshold.getValue())
			);
		}
		for (auto &thread : threads) {
			thread.join();
		}
	}

	std::cout << "Done" << std::endl;
	return 0;
}