//
// Created by denn_ma on 9/6/18.
//

#ifndef SDFGEN_STOPWATCH_H
#define SDFGEN_STOPWATCH_H

#include <mutex>

using Time = double;

class StopWatch {
public:
	StopWatch();
	~StopWatch();

	/** Resets the stopwatch. */
	void reset();
	/** Returns the start time in seconds*/
	double startTime() const;
	/** Returns the elapsed time in seconds */
	double stopTime() const;

	/** Elapsed time in seconds */
	void   elapsed_time(double &secs);
	/** Elapsed time in seconds */
	double elapsed_time();//inline

	/** Elapsed time in milliseconds */
	void   elapsed_mtime(double &msecs);
	/** Elapsed time in milliseconds */
	double elapsed_mtime();//inline

private:
	Time m_start;
	Time m_stop;
	std::mutex m_m;

};

static inline Time currentTime(){
	timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return double(t.tv_sec) + double(t.tv_nsec) / 1000000000.;
};


#endif //SDFGEN_STOPWATCH_H
