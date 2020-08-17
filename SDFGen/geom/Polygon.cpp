//
// Created by Maximilian Denninger on 13.08.18.
//

#include <cfloat>
#include "Polygon.h"
#include "../util/Utility.h"
#include "BoundingBox.h"
#include "PolygonCubeIntersection.h"

Polygon::Polygon(const iPoint &indices, Points &pointsRef) : m_calcNormal(false) {
    for (const auto &ele : indices) {
        m_points.emplace_back(pointsRef[ele - 1]);
    }
}

double Polygon::calcDistanceConst(const dPoint &point) const {
    const double planeDist = m_main.getDist(point);
    const double desiredSign = sgn(planeDist);
    const double zeroVal = 0.000;
    for (unsigned int currentEdgeId = 0; currentEdgeId < 3; ++currentEdgeId) {
        const double dist = m_edgePlanes[currentEdgeId].getDist(point);
        if (dist < zeroVal) { // is outside
            const double firstBorder = m_borderPlanes[currentEdgeId][0].getDist(point);
            const double secondBorder = m_borderPlanes[currentEdgeId][1].getDist(point);
            if (firstBorder < zeroVal) {
                // use the point dist to the first point
                return desiredSign * (point - m_borderPlanes[currentEdgeId][0].m_pointOnPlane).length();
            } else if (secondBorder < zeroVal) {
                // use the point dist to second point
                return desiredSign * (point - m_borderPlanes[currentEdgeId][1].m_pointOnPlane).length();
            } else {

                return desiredSign * m_edgeLines[currentEdgeId].getDist(point);
            }
        }
    }
    return planeDist;
}

double Polygon::calcDistance(const dPoint &point) {
    if (m_calcNormal) {
        return calcDistanceConst(point);
    } else {
        calcNormal();
        if (m_calcNormal) {
            return calcDistanceConst(point);
        }
    }
    return -1;
}

void Polygon::calcNormal() {
    if (m_points.size() == 3) {
        const auto vec01 = m_points[1] - m_points[0];
        const auto vec02 = m_points[2] - m_points[0];

        m_main.calcNormal(vec01, vec02);
        m_main.calcNormalDist(m_points[0]);

        // edgeId = 0 -> edge between 0 and 1
        for (unsigned int edgeId = 0; edgeId < 3; ++edgeId) {
            const unsigned int nextId = (edgeId + 1) % 3;
            const unsigned int nextNextId = (edgeId + 2) % 3;
            const auto first = m_points[nextNextId] - m_points[nextId];
            const auto second = m_points[edgeId] - m_points[nextId];
            const double alpha = acos(dot(first, second) / (first.length() * second.length()));
            dPoint normal;
            if (fabs(alpha - M_PI_2) > 1e-5) {
                const double length = cos(alpha) * second.length();
                const auto dir = first.normalize();
                auto newPoint = dir * length + m_points[nextId];
                normal = m_points[edgeId] - newPoint;
            } else { // they are perpendicular
                normal = second;
            }
            m_edgePlanes[edgeId] = Plane(normal, m_points[nextId]);
            m_edgeLines[edgeId] = Line(m_points[nextNextId], m_points[nextId]);
            m_borderPlanes[edgeId][0] = Plane(first, m_points[nextId]);
            m_borderPlanes[edgeId][1] = Plane(first * (-1), m_points[nextNextId]);
        }

        m_calcNormal = true;
    } else {
        printError("Polygons have to be triangles!");
    }
}


void Polygon::flipPointOrder() {
    std::swap(m_points[0], m_points[1]);
}


Polygons removePolygonsOutOfFrustum(Polygons &polys) {
    Polygons newPolys;
    for (unsigned int i = 0; i < polys.size(); ++i) {
        if (t_c_intersection(polys[i], d_zeros + d_ones + 1e-6, d_zeros - d_ones - 1e-6)) {
            newPolys.emplace_back(polys[i].getPoints());
        }
    }
    return newPolys;
}

Polygons frustumClipping(Polygons &polys) {
    Polygons tmpPolys;
    Polygons* newPolys = &tmpPolys;
    Polygons* oldPolys = &polys;

    for (int j = 0; j < 3; j++) {
        for (int d = -1; d <= 1; d += 2) {
            for (unsigned int i = 0; i < oldPolys->size(); ++i) {
                int needsClipping = 0;
                for (auto &point : (*oldPolys)[i].getPoints()) {
                    if (point[j] * d > 1) {
                        needsClipping++;
                    }
                }

                if (needsClipping > 0) {
                    dPoint normal(0, 0, 0);
                    dPoint pointOnPlane(0, 0, 0);
                    normal[j] = d;
                    pointOnPlane[j] = d;
                    Plane clippingPlane(normal, pointOnPlane);
                    clipAtPlane(*newPolys, (*oldPolys)[i].getPoints(), clippingPlane, needsClipping);
                } else {
                    newPolys->emplace_back((*oldPolys)[i].getPoints());

                }
            }

            std::swap(newPolys, oldPolys);
            newPolys->clear();
        }
    }
    return *oldPolys;
}


Polygons removeFlatPolygons(Polygons &polys, double threshold) {
    Polygons newPolys;

    for (unsigned int i = 0; i < polys.size(); ++i) {
        auto points = polys[i].getPoints();
        Line line(points[0], points[1]);

        if ((points[0] - points[1]).length() > threshold && line.getDist(points[2]) > threshold) {
            newPolys.emplace_back(polys[i].getPoints());
        }
    }

    printMsg("Removed polygons with all three vertices on a line: " + std::to_string(polys.size() - newPolys.size()));
    return newPolys;
}


Polygons nearFarPolygonClipping(Polygons &polygons, double nearClipping, double farClipping) {
    Polygons newPolys;
    Plane nearClippingPlane({0, 0, -1}, {0, 0, -nearClipping});
    for (unsigned int i = 0; i < polygons.size(); ++i) {
        bool usePoly = false;
        auto &points = polygons[i].getPoints();

        for (const auto &point : points) {
            if (point[2] < -nearClipping && point[2] > -farClipping) {
                usePoly = true;
            }
        }
        auto &first = points[0];
        if (!usePoly) {
            for (int j = 1; j < 3; ++j) {
                // poly stretches through volume
                if (first[2] >= -nearClipping && points[j][2] <= -farClipping) {
                    usePoly = true;
                    break;
                } else if (first[2] <= -farClipping && points[j][2] >= -nearClipping) {
                    usePoly = true;
                    break;
                }
            }
        }
        int needsClipping = 0;
        for (const auto &point : points) {
            if (usePoly && point[2] > -nearClipping) {
                ++needsClipping;
            }
        }
        if (needsClipping > 0) {
            clipAtPlane(newPolys, points, nearClippingPlane, needsClipping);
        } else if (usePoly) {
            newPolys.emplace_back(points);
        }
    }

    return newPolys;
}

void clipAtPlane(Polygons &newPolys, Points &polygonPoints, const Plane &clippingPlane, int needsClipping) {
    std::array<Line, 3> lines;
    for (unsigned int i = 0; i < 3; ++i) {
        lines[i] = Line(polygonPoints[i], polygonPoints[(i + 1) % 3]);
    }
    if (needsClipping == 1) {
        int notUsed = -1;
        for (unsigned int i = 0; i < 3; ++i) {
            if (!clippingPlane.intersectBy(lines[i])) {
                notUsed = i;
            }
        }

        dPoint firstCut = clippingPlane.intersectionPoint(lines[(notUsed + 1) % 3]);
        dPoint secondCut = clippingPlane.intersectionPoint(lines[(notUsed + 2) % 3]);
        Point3D firstCut3D(firstCut, 0);
        Point3D secondCut3D(secondCut, 0);
        Points p1 = {polygonPoints[notUsed], polygonPoints[(notUsed + 1) % 3], firstCut3D};
        Points p2 = {firstCut3D, secondCut3D, polygonPoints[notUsed]};
        newPolys.emplace_back(p1);
        newPolys.emplace_back(p2);
    } else if (needsClipping == 2) {
        int used = -1;
        for (unsigned int i = 0; i < 3; ++i) {
            if (!clippingPlane.intersectBy(lines[i])) {
                used = (i + 2) % 3;
                break;
            }
        }

        dPoint firstCut = clippingPlane.intersectionPoint(lines[used]);
        dPoint secondCut = clippingPlane.intersectionPoint(lines[(used + 2) % 3]); // +2 == -1
        Point3D firstCut3D(firstCut, 0);
        Point3D secondCut3D(secondCut, 0);
        Points p1 = {polygonPoints[used], firstCut3D, secondCut3D};
        newPolys.emplace_back(p1);
    }
}


