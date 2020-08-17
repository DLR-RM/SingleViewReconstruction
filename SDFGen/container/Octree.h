//
// Created by Dominik Winkelbauer on 27.09.18.
//

#ifndef SDFGEN_OCTREE_H
#define SDFGEN_OCTREE_H

#include <vector>
#include "../geom/Polygon.h"
#include "Array3D.h"
#include <thread>

enum OctreeState {
    OCTREE_EMPTY = 0,
    OCTREE_FILLED = 1,
    OCTREE_MIXED = 2
};

class Octree {
public:

    Octree(const dPoint& origin, const dPoint& size, int level = 0, const Octree* parent = nullptr);

	void changeState(OctreeState newState);
	std::vector<Octree>& getChilren();
	bool intersectsWith(const Polygon& polygon) const;
	const std::vector<Polygon*>& getIntersectingPolygons() const;
	void addIntersectingPolygon(Polygon& polygon);
	void setNeighbour(int direction, Octree& octree);
	const Octree& findNodeContainingPoint(const dPoint& point) const;
	bool containsPoint(const dPoint &point) const;
	OctreeState getState() const;
	double calcMinDistanceToPoint(const dPoint &point) const;
	double calcMaxDistanceToPoint(const dPoint &point) const;
	void findLeafNeighbours(std::vector<const Octree *> &neighbours) const;
	int resetIds(int nextId = 0);
	int getId() const;
	const dPoint& getOrigin() const;
	void collectLeafChilds(std::vector<const Octree*>& childs) const;


	const Octree* m_parent;
	std::vector<Polygon*> m_intersectingPolygons;
private:
    std::vector<Octree> m_children;
    std::vector<Octree*> m_neighbours;
    OctreeState m_state;
    int m_level;
	int m_id;

    dPoint m_origin;
    dPoint m_size;

	static const std::vector<dPoint> m_positions;
	static const std::vector<dPoint> m_directions;

	int indexFromPosition(const dPoint &position);

	const dPoint& positionFromIndex(int index);

	void findLeafChildsInDirection(const dPoint &direction, std::vector<const Octree *> &neighbours) const;

	int flipDirectionIndex(int direction);
	const Octree* getClosestNeighbour(int direction) const;

};

void buildOctree(Octree& octree, int maxLevel);

#endif //SDFGEN_OCTREE_H
