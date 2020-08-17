//
// Created by Maximilian Denninger on 9/3/18.
//

#ifndef SDFGEN_TRANSFORM_H
#define SDFGEN_TRANSFORM_H

#include <array>
#include "Point.h"
#include "MinMaxValue.h"
#include <cfloat>


template<typename dataType>
class Transform {
public:

	Transform(): m_values{}{
	};


	dataType& operator()(unsigned int x, unsigned int y);

	const dataType& operator()(unsigned int x, unsigned int y) const;


	void
	setAsProjectionWith(const double xFov, const double yFov, const double near, const double far);

	void setAsCameraTransTowards(const dPoint& pos, const dPoint& towardsPose, const dPoint& up);

	template<typename otherType>
	Transform<dataType> operator*(const Transform<otherType>& rhs) const;

	template<typename otherType>
	Point<dataType> operator*(const Point<otherType>& rhs) const;

	template<typename otherType>
	void transform(Point<otherType>& point) const;

	void transpose();

private:

	dataType& at(unsigned int x, unsigned int y);

	const dataType& at(unsigned int x, unsigned int y) const;

	// row sorted -> 0 1 2 3
	// 				 4 5 6 7
	// 				 8 9 10 11
	// 				 12 13 14 15
	std::array<dataType, 16> m_values;
};

#define __TRANSFORM_INTERNAL__

#include "Transform_i.h"

#undef __TRANSFORM_INTERNAL__

using dTransform = Transform<double>;

#endif //SDFGEN_TRANSFORM_H
