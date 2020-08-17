//
// Created by Maximilian Denninger on 9/3/18.
//


#ifndef __TRANSFORM_INTERNAL__
#error "Don't include Transform_i.h directly. Include Transform.h instead."
#endif

#include "../../util/Utility.h"


template<typename dataType>
std::ostream& operator<<(std::ostream& os, const Transform<dataType>& point){
	os << "\n";
	for(unsigned int i = 0; i < 4; ++i){
		for(unsigned int j = 0; j < 4; ++j){
			if(fabs(point(i, j)) < 1e-5){
				os << 0;
			}else{
				os << point(i, j);
			}
			if(j < 3){
				os << " ";
			}
		}
		os << "\n";
	}
	return os;
}

template<typename dataType>
void Transform<dataType>::setAsProjectionWith(const double xFov, const double yFov, const double near, const double far){
	auto width = 1. / tan(xFov);
	auto height = 1. / tan(yFov);
	at(0, 0) = width;
	at(1, 1) = height;
	at(2, 2) = (near + far) / (near - far);
	at(2, 3) = (2 * near * far) / (near - far);
	at(3, 2) = -1.0;
	at(3, 3) = 0;
}

template<typename dataType>
void Transform<dataType>::setAsCameraTransTowards(const dPoint& pos, const dPoint& towardsPose, const dPoint& up){
	dPoint zaxis = towardsPose.normalize() * -1.0;
	dPoint xaxis = (cross(up, zaxis)).normalize();
	dPoint yaxis = (cross(zaxis, xaxis)).normalize();
	for(unsigned int i = 0; i < 3; ++i){
		at(0, i) = xaxis[i];
		at(1, i) = yaxis[i];
		at(2, i) = zaxis[i];
	}
	at(0, 3) = -dot(xaxis, pos);
	at(1, 3) = -dot(yaxis, pos);
	at(2, 3) = -dot(zaxis, pos);
	at(3, 3) = 1;
}

template<typename dataType>
dataType& Transform<dataType>::at(unsigned int x, unsigned int y){
	return m_values[x * 4 + y];
}

template<typename dataType>
const dataType& Transform<dataType>::at(unsigned int x, unsigned int y) const{
	return m_values[x * 4 + y];
}

template<typename dataType>
dataType& Transform<dataType>::operator()(unsigned int x, unsigned int y){
	return m_values[x * 4 + y];
}

template<typename dataType>
const dataType& Transform<dataType>::operator()(unsigned int x, unsigned int y) const{
	return m_values[x * 4 + y];
}

template<typename dataType>
template<typename otherType>
Transform<dataType> Transform<dataType>::operator*(const Transform<otherType>& rhs) const{
	Transform<dataType> res;
	for(unsigned int i = 0; i < 4; ++i){
		for(unsigned int j = 0; j < 4; ++j){
			dataType value = 0;
			for(unsigned int m = 0; m < 4; ++m){
				value += at(i, m) * (dataType) rhs(m, j);
			}
			res(i, j) = value;
		}
	}
	return res;
}

template<typename dataType>
template<typename otherType>
Point<dataType> Transform<dataType>::operator*(const Point<otherType>& rhs) const{
	Point<dataType> res;
	for(unsigned int i = 0; i < 4; ++i){
		dataType value = 0;
		for(unsigned int j = 0; j < 3; ++j){
			value += rhs[j] * at(i, j);
		}
		value += at(i, 3);
		if(i < 3){
			res[i] = value;
		}else if(fabs(value) > 1e-5){
			res /= value; // divide by the w, to normalize it again
		}else{
			res = {0,0,0};
		}
	}
	return res;
}

template<typename dataType>
template<typename otherType>
void Transform<dataType>::transform(Point<otherType>& point) const{
	Point<long double> res;
	for(unsigned int i = 0; i < 4; ++i){
		long double value = 0;
		for(unsigned int j = 0; j < 3; ++j){
			value += point[j] * at(i, j);
		}
		value += at(i, 3);
		if(i < 3){
			res[i] = value;
		}else if(fabs(value) > 1e-5){
			res /= value; // divide by the w, to normalize it again
		}else{
			printError("This is dangerous");
		}
	}
	for(unsigned int i = 0; i < 3; ++i){
		point[i] = res[i];
	}
}

template<typename dataType>
void Transform<dataType>::transpose(){
	for(unsigned int i = 1; i < 4; ++i){
		std::swap(at(i, 0), at(0, i));
	}
	for(unsigned int i = 1; i < 3; ++i){
		std::swap(at(i, 3), at(3, i));
	}
	std::swap(at(1, 2), at(2, 1));
}
