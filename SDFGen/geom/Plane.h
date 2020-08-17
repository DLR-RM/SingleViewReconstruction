//
// Created by Maximilian Denninger on 9/3/18.
//

#ifndef SDFGEN_PLANE_H
#define SDFGEN_PLANE_H

template <typename T>
int sgn(T val) {
    return (T(0) <= val) - (val < T(0));
}

#include "math/Point.h"
#include "Line.h"

struct Plane {

	Plane(): m_dist(0.0){};

	explicit Plane(dPoint normal, const dPoint& pointOnPlane) : m_normal(std::move(normal)), m_dist(0.0){
		m_normal.normalizeThis();
		calcNormalDist(pointOnPlane);
	}

	Plane(dPoint normal, double dist) : m_normal(std::move(normal)), m_dist(dist){};

	void calcNormal(const dPoint& first, const dPoint& second){
		m_normal = cross(first, second);
		m_normal.normalizeThis();
	}

	void calcNormalDist(const dPoint& pointOnPlane){
		m_pointOnPlane = pointOnPlane;
		m_dist = dot(m_normal, pointOnPlane);
	}

	double getDist(const dPoint& point) const {
		return dot(m_normal, point) - m_dist;
	}

	bool intersectBy(const Line& line) const {
//		printVars(getDist(*line.m_first), getDist(*line.m_second));
		// check if points are both on the same side of the plane -> no intersect
		return sgn(getDist(*line.m_first)) != sgn(getDist(*line.m_second));
	}

	dPoint intersectionPoint(const Line& line) const {
		double d = dot(m_pointOnPlane - *line.m_first, m_normal) / dot(line.m_normal, m_normal);
		return line.m_normal * d + *line.m_first;
	}

	dPoint m_normal;
	dPoint m_pointOnPlane;

	double m_dist;
};


#endif //SDFGEN_PLANE_H
