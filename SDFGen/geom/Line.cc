//
// Created by Maximilian Denninger on 9/3/18.
//

#include "Line.h"

double Line::getDist(const dPoint& toPoint) const{
	const auto vecToPoint = *m_first - toPoint;
	return (vecToPoint - m_normal * dot(vecToPoint, m_normal)).length();
}

Line::Line(Point3D& first, Point3D& second) : m_first(&first), m_second(&second) {
	m_normal = *m_second - *m_first;
	m_normal.normalizeThis();
}
