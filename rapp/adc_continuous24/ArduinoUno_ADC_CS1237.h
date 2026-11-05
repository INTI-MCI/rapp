#ifndef ArduinoUno_ADC_CS1237_h
#define ArduinoUno_ADC_CS1237_h
#define DEBUG 0 // Set to 0/1 to disable/enable debug messages
#define PROFILE_CS1237 0 // Set to 0/1 to disable/enable profiler
#define OFF_ON_SETTLING_TIME 1000
#define REGISTER_SETTLING_TIME 350

#if defined(ARDUINO) && (ARDUINO >= 100)
#include <Arduino.h>
#else
#include <WProgram.h>
#endif

#include "Profiling.h"

class ArduinoUno_ADC_CS1237 {
    public:
        ArduinoUno_ADC_CS1237(uint8_t CLOCK, uint8_t DATA) // Constructor. (SCLK pin, DATA pin)
            {SCLK = CLOCK; DOUT_DRDY = DATA;}
		
		void begin(void);

		long readADC();
        long readADCwProfiler();

        ProfilerTimes getProfiler() const;

		void setRegister(int registertowrite, int valuetowrite);
        void setDefaultRegister(void);
        void setFullRegister(byte register_to_write);
	    int getRegister();
        void read_and_printRegister();
        void printSerialByte(int value);

        int getDOUT_DRDY();
        int getSCLK();

    private:
        void clockCycle();
		void customDelay455ns();
        bool readBit();
        void writeBit(bool bit);
        
        ProfilerTimes profiler; // Profiler times in microseconds

        int DOUT_DRDY = 13; //DOUT/DRDY pin
		int SCLK = 19; //SCLK pin
        long ADCreading; //This variable stores the ADCreading - Can be omitted with all its relevant references!
		float pga_divider = 1.0; //default value = 1. This value stores the gain value
};

#endif