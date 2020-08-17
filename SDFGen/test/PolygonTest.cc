//
// Created by Maximilian Denninger on 8/30/18.
//

#include "PolygonTest.h"
#include <iostream>
#include "../util/Utility.h"
#include "../geom/Polygon.h"
#include "../container/Space.h"


#define TEST_FUNCTION(fct_name) \
	do{\
  		printMsg("Test: " #fct_name); \
		fct_name(); \
	}while(false); \

void PolygonTest::testAll(){
	TEST_FUNCTION(testPolygonInPlane);
}

void PolygonTest::testPolygonInPlane(){
	std::vector<Point3D> points = {{0,0,0,1}, {1,1,0,2}, {2,0,0,2}};

	Polygon poly({1,2,3}, points);
	bool worked = true;

	for(double i = -1; i <= 3; i += 0.5){
		dPoint p(i, 0, 0);
		auto dist = poly.calcDistance(p);
		if(i >= 0 && i <= 2){
			if(fabs(dist) > 1e-5){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}else if(i < 0){
			if(dist != fabs(i)){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}else{
			if(dist != fabs(i - 2)){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}
	}

	for(double d = -1; d <= 3; d += 0.5){
		dPoint p(d, -1, 0);
		auto dist = poly.calcDistance(p);
		if(d >= 0 && d <= 2){
			if(fabs(dist - 1) > 1e-5){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}else if(d < 0){
			double real = (points[0] - p).length();
			if(fabs(real - dist) > 1e-5){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}else{
			double real = (points[2] - p).length();
			if(fabs(real - dist) > 1e-5){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}

	}


	for(double d = -1; d <= 3; d += 0.5){
		dPoint p(d, 0, 10);
		auto dist = poly.calcDistance(p);
		if(d >= 0 && d <= 2){
			if(fabs(dist - -10) > 1e-5){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}else if(d < 0){
			double real = -(points[0] - p).length();
			if(fabs(real - dist) > 1e-5){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}else{
			double real = -(points[2] - p).length();
			if(fabs(real - dist) > 1e-5){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}
	}


	// Test flat polygon (Not necessary anymore, as we filter them out before)
	/*points = {{0,0,0,1}, {1,0,0,2}, {2,0,0,2}};
	poly = Polygon({1,2,3}, points);

	for(double i = -1; i <= 3; i += 0.5){
		dPoint p(i, 0, 0);
		auto dist = poly.calcDistance(p);
		if(i >= 0 && i <= 2){
			if(fabs(dist) > 1e-5){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}else if(i < 0){
			if(dist != fabs(i)){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}else{
			if(dist != fabs(i - 2)){
				printError("Dist is wrong: " << dist << " for: " << p); worked = false;
			}
		}
	}*/

	// Test correct distance sign, when using two polygons
	std::vector<Polygon> polys;
	points = {{0,0,0,1}, {1,1,0,2}, {2,0,0,2}};
	polys.push_back(Polygon({1, 2, 3}, points));

	points = {{2,0,0,1}, {1,1,0,2}, {0,0,1,2}};
	polys.push_back(Polygon({1, 2, 3}, points));

	Space space({1, 1, 1}, {4, 0, 0.1}, {4, 0, 0.1});
	space.calcDistsExactly(polys, 7, 1e-4);

	dPoint p(6, 0, 0.15);
	points = {{0,0,0,1}, {1,1,0,2}, {2,0,0,2}};
	double dist = space.getI(0, 0, 0);
	double real = (points[2] - p).length();
	if (fabs(real - dist) > 1e-5){
		printError("Dist is wrong: " << dist << " for: " << p); worked = false;
	}


	polys.clear();
	unsigned int numberOfVoxel = 10;
	points = {{0,0,0,1}, {1,1,0,2}, {2,0,0,2}};
	polys.push_back(Polygon({1, 2, 3}, points));
	points = {{0,0,2,1}, {1,1,2,2}, {2,0,2,2}};
	polys.push_back(Polygon({1, 3, 2}, points));

	Space spaceDist({1, 1, numberOfVoxel}, {-1, -1, -1}, {4, 4, 4});
	spaceDist.calcDistsExactly(polys, 7, 1e-4);

	float realDist = 0.8;
	for (unsigned int i = 0; i < numberOfVoxel; ++i) {
		if(fabs(realDist - spaceDist.getI(0, 0, i)) > 1e-5){
			printError("Dist is wrong: " << spaceDist.getI(0, 0, i) << " for: " << i << " - true: " << realDist); worked = false;
		}
		if (i < 4)
			realDist -= 0.4;
		else if (i > 4)
			realDist += 0.4;
	}


	if(!worked){
		printError("One of the test failed! Stop execution!");
		exit(1);
	}

}
