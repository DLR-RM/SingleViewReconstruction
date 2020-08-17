/*
 * Code from MathGeoLib (https://github.com/juj/MathGeoLib/blob/master/src/Geometry/Triangle.cpp#L624)
 *
 * Copyright Jukka Jylänki
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PolygonCubeIntersection.h"
#include <cmath>

bool t_c_intersection(const Polygon& tr, dPoint cubeMax, dPoint cubeMin) {

    dPoint tMin = eMin(tr.getPoints()[0], eMin(tr.getPoints()[1], tr.getPoints()[2]));
    dPoint tMax = eMax(tr.getPoints()[0], eMax(tr.getPoints()[1], tr.getPoints()[2]));

    if (tMin[0] >= cubeMax[0] || tMax[0] <= cubeMin[0]
        || tMin[1] >= cubeMax[1] || tMax[1] <= cubeMin[1]
        || tMin[2] >= cubeMax[2] || tMax[2] <= cubeMin[2])
        return false;

    dPoint center = (cubeMin + cubeMax) * 0.5f;
    dPoint h = cubeMax - center;

    const dPoint t[3] = { tr.getPoints()[1]-tr.getPoints()[0], tr.getPoints()[2]-tr.getPoints()[0], tr.getPoints()[2]-tr.getPoints()[1] };

    dPoint ac = tr.getPoints()[0]-center;

    dPoint n = cross(t[0], t[1]);
    double s = dot(n, ac);
    double r = fabs(dot(h, eAbs(n)));
    if (fabs(s) >= r)
        return false;

    const dPoint at[3] = { eAbs(t[0]), eAbs(t[1]), eAbs(t[2]) };

    dPoint bc = tr.getPoints()[1]-center;
    dPoint cc = tr.getPoints()[2]-center;

    // SAT test all cross-axes.
    // The following is t.getPoints()[0] fully unrolled loop of this code, stored here for reference:
    /*
    float d1, d2, a1, a2;
    const vec e[3] = { DIR_VEC(1, 0, 0), DIR_VEC(0, 1, 0), DIR_VEC(0, 0, 1) };
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
        {
            vec axis = Cross(e[i], t[j]);
            ProjectToAxis(axis, d1, d2);
            aabb.ProjectToAxis(axis, a1, a2);
            if (d2 <= a1 || d1 >= a2) return false;
        }
    */

    // eX <cross> t[0]
    double d1 = t[0][1] * ac[2] - t[0][2] * ac[1];
    double d2 = t[0][1] * cc[2] - t[0][2] * cc[1];
    double tc = (d1 + d2) * 0.5f;
    r = fabs(h[1] * at[0][2] + h[2] * at[0][1]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // eX <cross> t[1]
    d1 = t[1][1] * ac[2] - t[1][2] * ac[1];
    d2 = t[1][1] * bc[2] - t[1][2] * bc[1];
    tc = (d1 + d2) * 0.5f;
    r = fabs(h[1] * at[1][2] + h[2] * at[1][1]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // eX <cross> t[2]
    d1 = t[2][1] * ac[2] - t[2][2] * ac[1];
    d2 = t[2][1] * bc[2] - t[2][2] * bc[1];
    tc = (d1 + d2) * 0.5f;
    r = fabs(h[1] * at[2][2] + h[2] * at[2][1]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // eY <cross> t[0]
    d1 = t[0][2] * ac[0] - t[0][0] * ac[2];
    d2 = t[0][2] * cc[0] - t[0][0] * cc[2];
    tc = (d1 + d2) * 0.5f;
    r = fabs(h[0] * at[0][2] + h[2] * at[0][0]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // eY <cross> t[1]
    d1 = t[1][2] * ac[0] - t[1][0] * ac[2];
    d2 = t[1][2] * bc[0] - t[1][0] * bc[2];
    tc = (d1 + d2) * 0.5f;
    r = fabs(h[0] * at[1][2] + h[2] * at[1][0]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // eY <cross> t[2]
    d1 = t[2][2] * ac[0] - t[2][0] * ac[2];
    d2 = t[2][2] * bc[0] - t[2][0] * bc[2];
    tc = (d1 + d2) * 0.5f;
    r = fabs(h[0] * at[2][2] + h[2] * at[2][0]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // eZ <cross> t[0]
    d1 = t[0][0] * ac[1] - t[0][1] * ac[0];
    d2 = t[0][0] * cc[1] - t[0][1] * cc[0];
    tc = (d1 + d2) * 0.5f;
    r = fabs(h[1] * at[0][0] + h[0] * at[0][1]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // eZ <cross> t[1]
    d1 = t[1][0] * ac[1] - t[1][1] * ac[0];
    d2 = t[1][0] * bc[1] - t[1][1] * bc[0];
    tc = (d1 + d2) * 0.5f;
    r = fabs(h[1] * at[1][0] + h[0] * at[1][1]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // eZ <cross> t[2]
    d1 = t[2][0] * ac[1] - t[2][1] * ac[0];
    d2 = t[2][0] * bc[1] - t[2][1] * bc[0];
    tc = (d1 + d2) * 0.5f;
    r = fabs(h[1] * at[2][0] + h[0] * at[2][1]);
    if (r + fabs(tc - d1) < fabs(tc))
        return false;

    // No separating axis exists, the AABB and triangle intersect.
    return true;
}
