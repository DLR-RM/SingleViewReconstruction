//
// Created by Maximilian Denninger on 13.08.18.
//

#ifndef SDFGEN_ARRAY3D_I_H
#define SDFGEN_ARRAY3D_I_H

#include "Array3D.h"

#ifndef __ARRAY3D_INTERNAL__
#error "Don't include Array3D_i.h directly. Include Array3D.h instead."
#endif

template<typename dataType>
Array3D<dataType>::Array3D(const uiPoint& size){
	setSize(size);
}

template<typename dataType>
void Array3D<dataType>::setSize(const uiPoint& size){
	m_size = size;
	m_data.resize(internalLength());
}

template<typename dataType>
dataType& Array3D<dataType>::operator()(unsigned int i, unsigned int j, unsigned int k){
	return m_data[getInternVal(i,j,k)];
}

template<typename dataType>
const dataType& Array3D<dataType>::operator()(unsigned int i, unsigned int j, unsigned int k) const {
	return m_data[getInternVal(i,j,k)];
}

template<typename dataType>
void Array3D<dataType>::fill(const dataType& defaultValue){
	for(unsigned int i = 0; i < m_data.size(); ++i){
		m_data[i] = defaultValue;
	}
}

#endif //SDFGEN_ARRAY3D_I_H
