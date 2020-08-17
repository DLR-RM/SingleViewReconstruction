//
// Created by Maximilian Denninger on 09.08.18.
//

#ifndef SDFGEN_OBJREADER_H
#define SDFGEN_OBJREADER_H

#include <string>
#include <fstream>
#include "../util/Utility.h"
#include "../geom/math/Point.h"
#include <vector>
#include "../geom/Polygon.h"
#include "../geom/BoundingBox.h"

class ObjReader {
public:
	ObjReader() = default;

	void read(const std::string& filePath);

	Polygons& getPolygon(){ return m_polygons; }

	BoundingBox& getBoundingBox() { return m_box; }

	std::vector<Point3D>& getPoints() { return m_points;}

private:
	std::vector<Point3D> m_points;
	Polygons m_polygons;

	BoundingBox m_box;

	bool startsWith(const std::string& line, const std::string& start);

	void removeStartAndTrailingWhiteSpaces(std::string& line);

};


#endif //SDFGEN_OBJREADER_H
