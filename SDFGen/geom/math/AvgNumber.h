//
// Created by Maximilian Denninger on 07.09.18.
//

#ifndef SDFGEN_AVGNUMBER_H
#define SDFGEN_AVGNUMBER_H

#include <array>
#include "Point.h"

class AvgNumber {

public:
	AvgNumber(double avg = 0.0): m_avg(avg), m_counter(0.0){}

	void addNr(double val){
		++m_counter;
		const double fac = 1.0 / m_counter;
		m_avg = fac * val + (1.0 - fac) * m_avg;
	}

	double avg() const { return m_avg; }

	unsigned long counter() const { return (unsigned long) m_counter; }

private:
	double m_avg;
	double m_counter;
};


class AvgPoint {

public:
	AvgPoint(double avg = 0.0): m_avgs({avg,avg,avg}), m_counter(0.0){}

	template <typename dataType>
	void addNr(const Point<dataType>& point){
		++m_counter;
		const double fac = 1.0 / m_counter;
		m_avgs[0] = fac * point[0] + (1.0 - fac) * m_avgs[0];
		m_avgs[1] = fac * point[1] + (1.0 - fac) * m_avgs[1];
		m_avgs[2] = fac * point[2] + (1.0 - fac) * m_avgs[2];
	}

	dPoint avg() const { return dPoint(m_avgs); }

private:
	std::array<double, 3> m_avgs;
	double m_counter;
};

#endif //SDFGEN_AVGNUMBER_H
