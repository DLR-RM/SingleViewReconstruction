//
// Created by Maximilian Denninger on 9/3/18.
//

#ifndef SDFGEN_LINE_H
#define SDFGEN_LINE_H

#include "math/Point.h"

struct Line {
	Line() = default;

	Line(Point3D& first, Point3D& second);

	double getDist(const dPoint& toPoint) const;

	Point3D* m_first;
	Point3D* m_second;
	dPoint m_normal;
};


#endif //SDFGEN_LINE_H
