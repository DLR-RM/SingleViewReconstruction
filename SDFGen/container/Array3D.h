//
// Created by Maximilian Denninger on 13.08.18.
//

#ifndef SDFGEN_ARRAY3D_H
#define SDFGEN_ARRAY3D_H

#include <vector>
#include "../geom/math/Point.h"

template<typename dataType>
class Array3D {
public:
	using InternalStorage = std::vector<dataType>;
	using Iterator = typename InternalStorage::iterator;
	using ConstIterator = typename InternalStorage::const_iterator;

	explicit Array3D(const uiPoint& size);

	~Array3D() = default;

	dataType& operator()(unsigned int i, unsigned int j, unsigned int k);

	const dataType& operator()(unsigned int i, unsigned int j, unsigned int k) const;

	void setSize(const uiPoint& size);

	const uiPoint& getSize() const { return m_size; }

	void fill(const dataType& defaultValue);

	InternalStorage& getData(){ return m_data; }

	Iterator begin(){ return m_data.begin(); }

	ConstIterator begin() const{ return m_data.begin(); }

	Iterator end(){ return m_data.end(); }

	ConstIterator end() const{ return m_data.end(); }

private:
	unsigned int internalLength() const { return m_size[0] * m_size[1] * m_size[2]; }

	unsigned int getInternVal(unsigned int i, unsigned int j, unsigned int k) const {
		return i * m_size[0] * m_size[1] + j * m_size[1] + k;
	}

	InternalStorage m_data;
	uiPoint m_size;
};

#define __ARRAY3D_INTERNAL__

#include "Array3D_i.h"

#undef __ARRAY3D_INTERNAL__


#endif //SDFGEN_ARRAY3D_H
