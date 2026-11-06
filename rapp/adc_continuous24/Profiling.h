#ifndef PROFILING_H
#define PROFILING_H

struct __attribute__((packed)) ProfilerTimes{
    unsigned long elapsed_time_DRDY;
    unsigned long max_time_DRDY = 0;
    unsigned long min_time_DRDY = 4294967295;
    unsigned long elapsed_time_read;
    unsigned long max_time_read = 0;
    unsigned long min_time_read = 4294967295;
    };

#endif