//
// Created by denn_ma on 2019-06-17.
//

#ifndef LOSSCALCULATOR_HDF5READERANDWRITER_H
#define LOSSCALCULATOR_HDF5READERANDWRITER_H


#include "util/StopWatch.h"
#include "Image.h"


class Hdf5ReaderAndWriter {
public:
    static std::mutex m_mutex;

	static hid_t openFile(const std::string& filePath, const char mode){
	    m_mutex.lock();
	    hid_t fileId;
		if(mode == 'r'){
			fileId = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		}else if(mode == 't'){
            fileId = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		}else{
            fileId = H5Fopen(filePath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		}
		m_mutex.unlock();
		return fileId;
	}

	static hid_t openReadDataSet(const hid_t fileId, const std::string& dataset){
		if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
			return H5Dopen(fileId, dataset.c_str(), H5P_DEFAULT);
		}
			throw unusableException;
		}

	static void read3DArray(const hid_t fileId, const std::string& dataset, Array3D<double>& array,
								  unsigned int size){
        m_mutex.lock();
        hid_t dset = openReadDataSet(fileId, dataset);

        const auto maxNr = size * size * size;
        auto* inner_array = new unsigned short[maxNr];
		H5Dread(dset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, (void*) inner_array);
		H5Dclose(dset);
		array.init(inner_array, size);
		delete[](inner_array);
        m_mutex.unlock();
	}

	static void readVoxel(const std::string& filePath, Array3D<double>& voxelGrid, unsigned int usedSize, const std::string name="voxelgrid"){
		hid_t fileId = openFile(filePath, 'r');
		read3DArray(fileId, name, voxelGrid, usedSize);
        m_mutex.lock();
		H5Fclose(fileId);
        m_mutex.unlock();
	}

	static void writeArrayToFile(const hid_t fileId, const std::string& dataset, Array3D<double>& array){
        m_mutex.lock();
		if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
			H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
		}
		hsize_t chunk[3] = {array.length(), array.length(), array.length()};
		auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
		H5Pset_deflate(dcpl, 9);
		H5Pset_chunk(dcpl, 3, chunk);

		hsize_t dims[3] = {array.length(), array.length(), array.length()};
		auto space = H5Screate_simple(3, dims, NULL);
		auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
		H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &array.getInner(0));
		H5Pclose(dcpl);
		H5Dclose(dset);
		H5Sclose(space);
		m_mutex.unlock();
	}

};

std::mutex Hdf5ReaderAndWriter::m_mutex = std::mutex();

#endif //LOSSCALCULATOR_HDF5READERANDWRITER_H
