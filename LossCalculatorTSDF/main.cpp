/************************************************************

  This example shows how to read and write data to a dataset
  using gzip compression (also called zlib or deflate).  The
  program first checks if gzip compression is available,
  then if it is it writes integers to a dataset using gzip,
  then closes the file.  Next, it reopens the file, reads
  back the data, and outputs the type of compression and the
  maximum value in the dataset to the screen.

  This file is intended for use with HDF5 Library version 1.8

 ************************************************************/

#include "hdf5.h"
#include <iostream>
#include <vector>
#include "Array3D.h"
#include "Hdf5ReaderAndWriter.h"
#include "util/StopWatch.h"
#include <tclap/CmdLine.h>
#include <thread>
#include <iterator>
#include <regex>

Array3D<double> average(const Array3D<double>& input){
    /**
     * Averages an input array to a size of 32x32x32.
     */
	Array3D<double> res(32);
	const unsigned int blockSize = input.length() / 32;
	const double amountOfBlocks = blockSize * blockSize * blockSize;
	for(unsigned int x = 0; x < 32; ++x){
		for(unsigned int y = 0; y < 32; ++y){
			for(unsigned int z = 0; z < 32; ++z){
				double mean_value = 0;
				for(unsigned int i = 0; i < blockSize; ++i){
					for(unsigned int j = 0; j < blockSize; ++j){
						for(unsigned int k = 0; k < blockSize; ++k){
							mean_value += input(x * blockSize + i, y * blockSize + j, z * blockSize + k);
						}
					}
				}
				res(x,y,z) = mean_value / amountOfBlocks;
			}
		}
	}
	return res;
}


void createLossMap(const std::string& file, const std::string& goalFile, unsigned int usedSize){
    /**
     * Creates a loss map for the given .hdf5 file, the file must contain a "voxelgrid" with a resolution of usedSize.
     * The resulting loss map will be saved to the goalFile.
     */
	StopWatch global;
	Array3D<double> voxel;
	try{
	    // at first the voxel grid will be read from the file
		Hdf5ReaderAndWriter::readVoxel(file, voxel, usedSize);
		// as the voxel grid is stored as 16 bit unsigned short it is converted back to a float
		// with a range from -1 to 1
		voxel.scaleToRange();


		Array3D<double> res(usedSize);
		// we then have to calculate the free space between the camera and the first obstacle
		// furthermore, we set the values of the wall, which are slightly before and after the first surface
		// the rest of the values are not changed
		voxel.projectWallValueIntoZ(res);
		// we now select a free position in the first plane (closest to the camera) to start our flood fill algorithm,
		// as it might be that there are obstacle intersecting with the near plane of the camera, so we have to search
		// for free space right in front of the camera
		auto pose = voxel.findFreeElementIn(2, usedSize - 1);

        // from this pose out we perform a flood fill to find all voxels which are not visible from the camera, but are
        // not filled. We set the losses for the free space behind obstacles and we set the surfaces for all hidden
        // obstacles. Only obstacles, which can not be reached, because they are behind a hole free wall, are neglected.
		voxel.performFloodFill(pose, res);

		// we then reduce the size from usedSize to 32x32x32
		Array3D<double> averaged_loss = average(res);

		// and store it in the goalFile, which is also an .hdf5 container
		auto fileId = Hdf5ReaderAndWriter::openFile(goalFile, 't');
		Hdf5ReaderAndWriter::writeArrayToFile(fileId, "lossmap", averaged_loss);
		H5Fclose(fileId);

		std::cout << "Done in " << global.elapsed_time() << ", " << goalFile << std::endl;
	}catch(UnusableException e){
		std::cout << "This file is broke: " << file << std::endl;
	}
}

class WordDelimitedByComma : public std::string {};

std::istream& operator>>(std::istream& is, WordDelimitedByComma& output){
	std::getline(is, output, ',');
	return is;
}

int main(int argc, char** argv){
	TCLAP::CmdLine cmd("Loss Calculator and TSDF smoother", ' ', "0.9");
	TCLAP::ValueArg<std::string> pathArg("p","path","Paths to the files, separated by a comma, for each file a new thread is started",true,"","string");
	TCLAP::ValueArg<int> maxAmountOfThreads("t","maxThreads","Maximum amount of threads",true,0,"int");
	TCLAP::ValueArg<int> resolution("r","resolution","Used resolution 128, 256 or 512",true,0,"int");

	cmd.add(pathArg);
	cmd.add(maxAmountOfThreads);
	cmd.add(resolution);
	cmd.parse( argc, argv );

	if(pathArg.getValue().length() == 0){
		std::cout << "Set the path value!" << std::endl;
		exit(1);
	}

	const unsigned int amountOfThreads = maxAmountOfThreads.getValue();
	const unsigned int usedSize = resolution.getValue();
	if(usedSize == 0){
		std::cout << "The resolution can not be zero" << std::endl;
		exit(1);
	}

	const std::string allPaths = pathArg.getValue();
	std::istringstream iss(allPaths);
	std::vector<std::string> files((std::istream_iterator<WordDelimitedByComma>(iss)),
								   std::istream_iterator<WordDelimitedByComma>());
    for(auto& filePath: files) {
        filePath = std::regex_replace(filePath, std::regex("~"), std::string(getenv("HOME")));
    }
    std::vector<std::string> goalFiles;
    for(const auto& file : files){
        auto pos = file.rfind('.');
        goalFiles.emplace_back(file.substr(0, pos) + "_loss_avg.hdf5");
    }

    unsigned int maxNr = std::min(amountOfThreads, (unsigned int) files.size());
    unsigned int fileCounter = 0;
    while(fileCounter < files.size()){
        std::vector<std::thread> threads;
        for(unsigned int i = 0; i < maxNr; ++i){
            if(fileCounter == files.size()){
                break;
            }
            const std::string filePath = files[fileCounter];
            std::cout << "Use file: " << filePath << std::endl;
            threads.emplace_back(&createLossMap, filePath, goalFiles[fileCounter], usedSize);
            ++fileCounter;
        }
        for(auto& thread : threads){
            thread.join();
        }
    }
	return 0;
}
