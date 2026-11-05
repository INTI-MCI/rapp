#ifndef ArduinoUno_ADC_ADS1232_h
#define ArduinoUno_ADC_ADS1232_h
#define DEBUG 0 // Set to 0/1 to disable/enable debug messages
#define PROFILE_ADS1232 0 // Set to 0/1 to disable/enable profiler
#define OFF_ON_SETTLING_TIME 1000
#define REGISTER_SETTLING_TIME 350

#if defined(ARDUINO) && (ARDUINO >= 100)
#include <Arduino.h>
#else
#include <WProgram.h>
#endif

#include "myADS1232.h"

#include "Profiling.h"

class ArduinoUno_ADC_ADS1232 {
    public:
        ArduinoUno_ADC_ADS1232(uint8_t CLOCK, uint8_t DATA, uint8_t PDWN, uint8_t SPEED) // Constructor. (SCLK pin, DATA pin)
            {pinSCLK = CLOCK; pinDOUT_DRDY = DATA; pinPDWN = PDWN; pinSPEED = SPEED;}
		
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
        int getPDWN();
        int getSPEED();

    private:
        ADS1232 ads; // Create an instance of the ADS1232 class
        
        ProfilerTimes profiler; // Profiler times in microseconds

        int pinDOUT_DRDY = 13; //DOUT/DRDY pin
		int pinSCLK = 19; //SCLK pin
		int pinPDWN = 10; //PDWN pin
		int pinSPEED = 12; //SPEED pin

        Speed _speed = SLOW; // Default speed is SLOW
};

#endif