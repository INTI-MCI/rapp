#ifndef ArduinoUno_ADC_CS1237_h
#define ArduinoUno_ADC_CS1237_h
#define DEBUG_CS1237 0 // Set to 0/1 to disable/enable debug messages
#define PROFILE_CS1237 0 // Set to 0/1 to disable/enable profiler
#define OFF_ON_SETTLING_TIME 1000
#define REGISTER_SETTLING_TIME 350

#if defined(ARDUINO) && (ARDUINO >= 100)
#include <Arduino.h>
#else
#include <WProgram.h>
#endif

class ArduinoUno_ADC_CS1237 {
    public:
        ArduinoUno_ADC_CS1237(uint8_t CLOCK, uint8_t DATA) // Constructor. (SCLK pin, DATA pin)
            {SCLK = CLOCK; DOUT_DRDY = DATA;}
		
		void begin(void);

        void clockCycle();
        bool readBit();
        void writeBit(bool bit);

		int32_t readADC();
        int32_t readADCwProfiler();

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
		void customDelay455ns();

		struct __attribute__((packed)) ProfilerTimes{
        unsigned long time1_DRDY; // Tiempos de espera a que cambie el pin DOUT_DRDY
        unsigned long time2_DRDY;
        unsigned long elapsed_time_DRDY;
        unsigned long max_time_DRDY = 0;
        unsigned long min_time_DRDY = 4294967295;
        unsigned long time1_read; // Tiempos de lectura de los 24 bits
        unsigned long time2_read;
        unsigned long elapsed_time_read;
        unsigned long max_time_read = 0;
        unsigned long min_time_read = 4294967295;
        unsigned long time1_bit; // Tiempos de lectura de cada bit
        unsigned long time2_bit;
        unsigned long elapsed_time_readBit[24];
		};
        
        ProfilerTimes profiler; // Profiler times in microseconds

        int DOUT_DRDY = 13; //DOUT/DRDY pin
		int SCLK = 19; //SCLK pin
        int32_t ADCreading; //This variable stores the ADCreading - Can be omitted with all its relevant references!
		float pga_divider = 1.0; //default value = 1. This value stores the gain value
};

#endif