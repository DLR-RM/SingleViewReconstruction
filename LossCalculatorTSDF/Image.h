//
// Created by denn_ma on 2019-06-19.
//

#ifndef LOSSCALCULATOR_IMAGE_H
#define LOSSCALCULATOR_IMAGE_H


template <typename T>
class Image {
public:

	Image():m_sizeX(0), m_sizeY(0), m_amountOfChannels(0), m_innerSize(1){
		m_values.resize(m_innerSize);
	}


	template <typename ArrayType>
	void init(ArrayType* array, std::array<unsigned int, 3> shape){
		m_innerSize = shape[0] * shape[1] * shape[2];
		m_values.resize(m_innerSize);
		for(unsigned int i = 0; i < m_innerSize; ++i){
			m_values[i] = T(array[i]);
		}
		m_sizeX = shape[0];
		m_sizeY = shape[1];
		m_amountOfChannels = shape[2];
	}

	unsigned int getSizeX(){ return m_sizeX; }
	unsigned int getSizeY(){ return m_sizeY; }
	unsigned int getAmountOfChannels(){ return m_amountOfChannels; }

	T& getInner(unsigned int i) { return m_values[i]; }

private:
	std::vector<T> m_values;
	unsigned int m_sizeX, m_sizeY, m_amountOfChannels;
	unsigned int m_innerSize;
};

#endif //LOSSCALCULATOR_IMAGE_H
