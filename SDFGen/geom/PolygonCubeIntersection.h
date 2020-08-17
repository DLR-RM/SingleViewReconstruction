/*
 * Code from GraphicsGems III (https://github.com/erich666/GraphicsGems/blob/master/gemsiii/triangleCube.c)
 * LICENSE
 * This code repository predates the concept of Open Source, and predates most licenses along such lines. As such, the official license truly is:
 * EULA: The Graphics Gems code is copyright-protected. In other words, you cannot claim the text of the code as your own and resell it. Using the code is permitted in any program, product, or library, non-commercial or commercial. Giving credit is not required, though is a nice gesture. The code comes as-is, and if there are any flaws or problems with any Gems code, nobody involved with Gems - authors, editors, publishers, or webmasters - are to be held responsible. Basically, don't be a jerk, and remember that anything free comes with no guarantee.
 */

#ifndef SDFGEN_POLYGONCUBEINTERSECTION_H
#define SDFGEN_POLYGONCUBEINTERSECTION_H

#include "Polygon.h"

bool t_c_intersection(const Polygon& tr, dPoint cubeMax, dPoint cubeMin);

#endif //SDFGEN_POLYGONCUBEINTERSECTION_H
