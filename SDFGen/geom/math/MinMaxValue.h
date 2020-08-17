//
// Created by denn_ma on 9/4/18.
//

#ifndef SDFGEN_MINMAXVALUE_H
#define SDFGEN_MINMAXVALUE_H


template<typename dataType>
class MinMaxValue {
public:

	MinMaxValue(dataType min, dataType max): m_min(min), m_max(max){
	};

	void add(dataType value){
		if(value < m_min){
			m_min = value;
		}
		if(value > m_max){
			m_max = value;
		}
	}

	void reset(dataType min, dataType max){
		m_min = min;
		m_max = max;
	}

	const dataType& min() const { return m_min; }

	const dataType& max() const { return m_max; }

private:
	dataType m_min;
	dataType m_max;

};

template<typename dataType>
std::ostream& operator<<(std::ostream& os, const MinMaxValue<dataType>& minMax){
	os << "min: " << minMax.min() << ", max: " << minMax.max();
	return os;
}
#endif //SDFGEN_MINMAXVALUE_H
