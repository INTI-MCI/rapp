#include "ArduinoUno_ADC_ADS1232.h"

void ArduinoUno_ADC_ADS1232::begin(void) {
    ads.begin(SCLK, DOUT_DRDY, PDWN, SPEED);
}

int32_t ArduinoUno_ADC_ADS1232::readADC() {
    int32_t value;
    ads.read(value);
    return value;
}

int32_t ArduinoUno_ADC_ADS1232::readADCwProfiler() {
    uint32_t start = micros();
    int32_t value;
    ads.read(value);
    profiler.elapsed_time_read = micros() - start;
    if (profiler.elapsed_time_read > profiler.max_time_read) profiler.max_time_read = profiler.elapsed_time_read;
    if (profiler.elapsed_time_read < profiler.min_time_read) profiler.min_time_read = profiler.elapsed_time_read;
    return value;
}

ProfilerTimes ArduinoUno_ADC_ADS1232::getProfiler() const {
    return profiler;
}

int ArduinoUno_ADC_ADS1232::getDOUT_DRDY(){
  return DOUT_DRDY;
}

int ArduinoUno_ADC_ADS1232::getSCLK(){
  return SCLK;
}

int ArduinoUno_ADC_ADS1232::getPDWN(){
  return PDWN;
}

int ArduinoUno_ADC_ADS1232::getSPEED(){
  return SPEED;
}