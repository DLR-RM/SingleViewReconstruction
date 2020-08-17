//
// Created by Maximilian Denninger on 09.08.18.
//

#ifndef SDFGEN_POINT_H
#define SDFGEN_POINT_H

#include <array>
#include <iostream>
#include <cmath>
#include "../../util/Utility.h"

template<typename dataType>
class Point {
public:

	using InternalStorage = std::array<dataType, 3>;
	using Iterator = typename InternalStorage::iterator;
	using ConstIterator = typename InternalStorage::const_iterator;

	Point() : m_data({0, 0, 0}){};

	explicit Point(InternalStorage data) : m_data(std::move(data)){};

	explicit Point(InternalStorage&& data) : m_data(std::move(data)){};

	Point(const Point<dataType>& rhs) : m_data(rhs.m_data){};

	Point(Point<dataType>&& rhs) : m_data(std::move(rhs.m_data)){};

	Point(dataType x, dataType y, dataType z) : m_data({x, y, z}){};

	template<typename differentType>
	explicit Point(const Point<differentType>& rhs);

	dataType& operator[](int i){ return m_data[i]; }

	const dataType operator[](int i) const{ return m_data[i]; }

	Point<dataType>& operator=(const Point<dataType>& rhs){
		m_data = rhs.m_data;
		return *this;
	}

	Point<dataType>& operator=(Point<dataType>&& rhs){
		m_data = std::move(rhs.m_data);
		return *this;
	}

	Point<dataType> operator+(dataType rhs) const;

	Point<dataType> operator-(dataType rhs) const;

	Point<dataType> operator+(const Point<dataType>& rhs) const;

	Point<dataType> operator-(const Point<dataType>& rhs) const;

	Point<dataType> operator/(dataType rhs) const;

	Point<dataType> operator*(dataType rhs) const;

	Point<dataType> operator/(const Point<dataType>& rhs) const;

	Point<dataType> operator*(const Point<dataType>& rhs) const;

	Point<dataType>& operator/=(dataType rhs);

	Point<dataType>& operator*=(dataType rhs);

	Point<dataType>& operator+=(const Point<dataType>& rhs);

	Point<dataType>& operator-=(const Point<dataType>& rhs);

	Point<dataType>& operator/=(const Point<dataType>& rhs);

	Point<dataType>& operator*=(const Point<dataType>& rhs);

	bool operator==(const Point<dataType>& rhs) const;

	bool operator!=(const Point<dataType>& rhs) const;

	Iterator begin(){ return m_data.begin(); }

	ConstIterator begin() const{ return m_data.begin(); }

	Iterator end(){ return m_data.end(); }

	ConstIterator end() const{ return m_data.end(); }

	double length() const{ return sqrt(m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2]); }

	double lengthSquared() const{ return m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2]; }

	double sumEle() const{ return m_data[0] + m_data[1] + m_data[2]; };

	double l1norm() const { return fabs(m_data[0]) + fabs(m_data[1]) + fabs(m_data[2]); }

	void normalizeThis(){
		const auto len = length();
		if(len > 0.0){
			for(unsigned int i = 0; i < 3; ++i){
				m_data[i] /= len;
			}
		}
	}

	Point<dataType> normalize() const{
		Point<dataType> res(m_data);
		const auto len = length();
		if(len > 0.0){
			for(unsigned int i = 0; i < 3; ++i){
				res.m_data[i] /= len;
			}
		}
		return res;
	}

private:

	InternalStorage m_data;
};

using dPoint = Point<double>;
using iPoint = Point<int>;
using uiPoint = Point<unsigned int>;


class Point3D : public Point<double> {
public:

	Point3D() : Point(), m_index(0), m_used(true){};

	Point3D(double x, double y, double z, unsigned int index) : Point<double>(x, y, z), m_index(index), m_used(true){};

	Point3D(const dPoint& p, unsigned int i) : Point<double>(p), m_index(i), m_used(true){};

	Point3D(const Point3D& rhs) : Point<double>(rhs[0], rhs[1], rhs[2]), m_index(0), m_used(rhs.m_used){};

	unsigned int getIndex(){ return m_index; }

	void setIndex(unsigned int index){ m_index = index; };

	bool used() const{ return m_used; }

	void setUsed(bool used){ m_used = used; }

private:
	unsigned int m_index;

	bool m_used;
};

#define POINT_CMP_FCT(lhs, rhs, op, outputType) \
    {lhs[0] op rhs[0] ? lhs[0] : rhs[0], \
		lhs[1] op rhs[1] ? lhs[1] : rhs[1], \
		lhs[2] op rhs[2] ? lhs[2] : rhs[2]} \


template<typename dataType, typename differentType>
static Point<dataType> eMax(const Point<dataType>& lhs, const Point<differentType>& rhs){
	return POINT_CMP_FCT(lhs, rhs, >, Point<dataType>);
}

template<typename dataType, typename differentType>
static Point<dataType> eMin(const Point<dataType>& lhs, const Point<differentType>& rhs){
	return POINT_CMP_FCT(lhs, rhs, <, Point<dataType>);
}

template<typename dataType>
static Point<dataType> eAbs(const Point<dataType>& lhs){
    return {fabs(lhs[0]), fabs(lhs[1]), fabs(lhs[2])};
}

template<typename dataType, typename differentType>
static Point<dataType> eMultiply(const Point<dataType>& lhs, const Point<differentType>& rhs){
	return {lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]};
}

template<typename dataType, typename differentType>
static Point<dataType> eDivide(const Point<dataType>& lhs, const Point<differentType>& rhs){
    return {lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2]};
}

template<typename dataType, typename differentType>
static Point<dataType> eMinEqual(const Point<dataType>& lhs, const Point<differentType>& rhs){
	return POINT_CMP_FCT(lhs, rhs, <=, Point<dataType>);
}

template<typename dataType, typename differentType>
static Point<dataType> eMaxEqual(const Point<dataType>& lhs, const Point<differentType>& rhs){
	return POINT_CMP_FCT(lhs, rhs, >=, Point<dataType>);
}

template<typename dataType, typename differentType>
static Point<dataType> cross(const Point<dataType>& lhs, const Point<differentType>& rhs){
	Point<dataType> res;
	res[0] = lhs[1] * rhs[2] - rhs[1] * lhs[2];
	res[1] = lhs[2] * rhs[0] - rhs[2] * lhs[0];
	res[2] = lhs[0] * rhs[1] - rhs[0] * lhs[1];
	return res;
};

template<typename dataType, typename differentType>
static double dot(const Point<dataType>& lhs, const Point<differentType>& rhs){
	return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
};

static dPoint d_zeros{0.0, 0.0, 0.0};
static dPoint d_ones{1.0, 1.0, 1.0};
static dPoint d_negOnes{-1.0, -1.0, -1.0};
static iPoint i_zeros{0, 0, 0};
static iPoint i_ones{1, 1, 1};
#define __POINT_INTERNAL__

#include "Point_i.h"

#undef __POINT_INTERNAL__

#endif //SDFGEN_POINT_H
