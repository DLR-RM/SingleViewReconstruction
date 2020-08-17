//
// Created by Maximilian Denninger on 15.08.18.
//

#ifndef SDFGEN_SPACE_H
#define SDFGEN_SPACE_H


#include "Array3D.h"
#include "../geom/Polygon.h"
#include "Octree.h"
#include <mutex>
#include <thread>
#include <queue>
#include <map>

struct QueueElement {
    double minDistanceToBlock;
    const Octree* node;

    QueueElement(double distance, const Octree* node) : minDistanceToBlock(distance), node(node) {
    };
};


class Space {
public:

	Space(const uiPoint& iSize, const dPoint& origin, const dPoint& size): m_data(iSize), m_origin(origin), m_size(size) {};

	void internCalcDistApproximately(const Array3D<std::vector<Polygon*>*>& initialContainedPolygons, Array3D<Polygon*>& closestPolygons,const Polygons &polygons, unsigned int size, double maxDistanceToMinPosDist, double truncationThreshold, int approximationAccuracy);
	void fillVoxel(const Array3D<std::vector<Polygon*>*>& initialContainedPolygons, Array3D<Polygon*>& closestPolygons, const uiPoint &voxel, std::vector<Polygon *> &polygonChoices, unsigned int size, double maxDistanceToMinPosDist, double truncationThreshold, int approximationAccuracy);

	void calcDistsApproximately(Polygons &polys, double maxDistanceToMinPosDist, double truncationThreshold, int approximationAccuracy);
	void calcDistsExactly(Polygons& polys, double minimumOctreeVoxelSize, double maxDistanceToMinPosDist, unsigned int amountOfThreads=0, double truncationThreshold=0);

	double getI(unsigned int i, unsigned int j, unsigned int k) { return m_data(i,j,k); }

    void correctInnerVoxel(int mode, double truncationThreshold);

	Array3D<float>& getData(){ return m_data; }
private:

	void internCalcDistExactly(const Octree &octree, unsigned int start, unsigned int end, int totalNumberOfLeafs, double maxDistanceToMinPosDist, double truncationThreshold);

	dPoint getCenterOf(unsigned int i, unsigned int j, unsigned int k);

    Octree createOctree(Polygons& polys, double minimumOctreeVoxelSize);

	Array3D<float> m_data;

	dPoint m_origin;
	dPoint m_size;

	double interCalcDistExactlyForPoint(const dPoint &point, const Octree &octree, std::vector<QueueElement> &openList, std::vector<const Octree *> &neighbours, std::vector<bool> &visitedNodes, const Polygon *&closestPoly, double maxDistanceToMinPosDist);

	enum VoxelLocation {
	    VOXEL_LOCATION_AMBIGUOUS,
        VOXEL_LOCATION_INSIDE,
        VOXEL_LOCATION_BORDER,
        VOXEL_LOCATION_OUTSIDE,
        VOXEL_LOCATION_INSIDE_RECALCULATED
	};
    void floodFillVisibleVoxels(Array3D<VoxelLocation> &voxelLocations, int borderWidth);

    void fillNegativeVoxelsAndAddBorder(Array3D<VoxelLocation> &voxelLocations, int borderWidth);

    void removeOutsideBorder(Array3D<VoxelLocation> &voxelLocations, int borderWidth);

    void recalculateInnerVoxel(const uiPoint& pos, const uiPoint& size, Array3D<VoxelLocation> &voxelLocations, double truncationThreshold);
    int recalculateInnerVoxels(Array3D<VoxelLocation> &voxelLocations, double truncationThreshold);

};


#endif //SDFGEN_SPACE_H
