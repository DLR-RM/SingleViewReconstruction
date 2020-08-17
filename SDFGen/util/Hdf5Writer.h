//
// Created by denn_ma on 2019-07-30.
//

#ifndef SDFGEN_HDF5WRITER_H
#define SDFGEN_HDF5WRITER_H

#include "hdf5.h"

namespace Hdf5Writer {

	static hid_t openFile(const std::string& filePath){
		return H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	}


	static void writeArrayToFileId(const hid_t fileId, const std::string& dataset, Array3D<float>& array){
		if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
			H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
		}
		const auto size = array.getSize();
		hsize_t chunk[3] = {size[0], size[1], size[2]};
		auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
		H5Pset_deflate(dcpl, 9);
		H5Pset_chunk(dcpl, 3, chunk);

		hsize_t dims[3] = {size[0], size[1], size[2]};
		auto space = H5Screate_simple(3, dims, NULL);
		auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
		H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array.getData()[0]));
		H5Pclose(dcpl);
		H5Dclose(dset);
		H5Sclose(space);
	}

	static void writeArrayToFileId(const hid_t fileId, const std::string& dataset, Array3D<unsigned short>& array){
		if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
			H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
		}
		const auto size = array.getSize();
		hsize_t chunk[3] = {size[0], size[1], size[2]};
		auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
		H5Pset_deflate(dcpl, 9);
		H5Pset_chunk(dcpl, 3, chunk);

		hsize_t dims[3] = {size[0], size[1], size[2]};
		auto space = H5Screate_simple(3, dims, NULL);
		auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_STD_U16LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
		H5Dwrite(dset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array.getData()[0]));
		H5Pclose(dcpl);
		H5Dclose(dset);
		H5Sclose(space);
	}

	static void writeArrayToFile(const std::string& filePath, Array3D<unsigned short>& voxelgrid){
		const auto fileId = openFile(filePath);
		Hdf5Writer::writeArrayToFileId(fileId, "voxelgrid", voxelgrid);
		H5Fclose(fileId);
	}

}

#endif //SDFGEN_HDF5WRITER_H
