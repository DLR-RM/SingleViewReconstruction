//
// Created by Maximilian Denninger on 13.08.18.
//

#ifndef SDFGEN_POLYGON_H
#define SDFGEN_POLYGON_H


#include "math/Point.h"
#include <vector>
#include "Line.h"
#include "Plane.h"
#include "math/Transform.h"
#include <fstream>
#include <list>
#include "BoundingBox.h"

using Points = std::vector<Point3D>;

class Polygon {
public:
	explicit Polygon(Points points): m_points(std::move(points)), m_calcNormal(false){
	};

	Polygon(const iPoint& indices, std::vector<Point3D>& pointsRef);

	Points& getPoints(){
		return m_points;
	}

	const Points& getPoints() const{
		return m_points;
	}

	double calcDistance3(const dPoint& point) const;

	double calcDistanceConst(const dPoint& point) const;

	double calcDistance(const dPoint& point);

	double calcDistance2(const dPoint& point);

	BoundingBox getBB() const {
		BoundingBox box;
		box.addPolygon(*this);
		return box;
	}

	double size(){
		const auto first = m_points[0] - m_points[1];
		const auto second = m_points[2] - m_points[1];
		return 0.5 * cross(first, second).length();
	}

	void calcNormal();

	void flipPointOrder();

private:

	Points m_points;

	bool m_calcNormal;

	Plane m_main;

	std::array<Plane, 3> m_edgePlanes;

	// for the perpendicular borders at the points
	std::array<std::array<Plane, 2>, 3> m_borderPlanes;

	std::array<Line, 3> m_edgeLines;

};

using Polygons = std::vector<Polygon>;


static void writeToDisc(Polygons& polygons, const std::string& filePath){

	std::ofstream output2(filePath, std::ios::out);

	int counter = 1;
//	BoundingBox cube({-2, -2, -2}, {2, 2, 2});
	for(auto& poly : polygons){
//		bool isInside = false;
//		for(auto& point : poly.getPoints()){
//			if(cube.isPointIn(point)){
//				isInside = true;
//				break;
//			}
//		}
////		isInside = true;
//		if(isInside){
		for(auto& point : poly.getPoints()){
			output2 << "v " << (point)[0] << " " << (point)[1] << " " << (point)[2] << "\n";
			point.setIndex(counter);
			counter += 1;
		}
//		}
	}
	int insideCounter = 0;
	for(auto& poly : polygons){
//		bool isInside = false;
//		for(auto& point : poly.getPoints()){
//			if(cube.isPointIn(point)){
//				isInside = true;
//				break;
//			}
//		}
//
////		isInside = true;
//		if(isInside){
		++insideCounter;
		output2 << "f";
		for(auto& point : poly.getPoints()){
			output2 << " " << point.getIndex();
		}
		output2 << "\n";

//		}
	}

	output2.close();
}

static void transformPoints(Polygons& polygons, const dTransform& trans, bool flipPointOrder = false){
	for(auto& poly: polygons){
		for(unsigned int i = 0; i < 3; ++i){
			auto& point = poly.getPoints()[i];
			trans.transform(point);
		}

		if (flipPointOrder) {
			poly.flipPointOrder();
		}
	}
}

static void scalePoints(Polygons& polygons, double scaling){
    for(auto& poly: polygons){
        for(unsigned int i = 0; i < 3; ++i){
            auto& point = poly.getPoints()[i];
            point *= scaling;
        }
    }
}

Polygons removePolygonsOutOfFrustum(Polygons& polys);
Polygons removeFlatPolygons(Polygons& polys, double threshold);
Polygons nearFarPolygonClipping(Polygons& polygons, double nearClipping, double farClipping);
Polygons frustumClipping(Polygons &polys);
void clipAtPlane(Polygons &newPolys, Points &polygonPoints, const Plane &clippingPlane, int needsClipping);

#endif //SDFGEN_POLYGON_H
