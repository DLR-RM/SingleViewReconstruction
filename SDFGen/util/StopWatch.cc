//
// Created by denn_ma on 9/6/18.
//

#include "StopWatch.h"

StopWatch::StopWatch(){
	reset();
}

StopWatch::~StopWatch(){}

void StopWatch::reset(){
	m_m.lock();
	m_start = currentTime();
	m_m.unlock();
}

void StopWatch::elapsed_time(double &secs){
	m_m.lock();
	m_stop = currentTime();
	secs = m_stop - m_start;
	m_m.unlock();
}

void StopWatch::elapsed_mtime(double &msecs){
	elapsed_time(msecs);
	msecs *= 1000.;
}

double StopWatch::startTime() const{
	return m_start;
}

double StopWatch::stopTime() const{
	return m_stop;
}

double StopWatch::elapsed_time()	{
	double t;
	elapsed_time(t);
	return t;
}

double StopWatch::elapsed_mtime(){
	double t;
	elapsed_mtime(t);
	return t;
}
