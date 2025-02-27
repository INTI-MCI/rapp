#ifndef ArduinoUno_ADC_CS1237_h
#define ArduinoUno_ADC_CS1237_h

#if defined(ARDUINO) && (ARDUINO >= 100)
#include <Arduino.h>
#else
#include <WProgram.h>
#endif

class ArduinoUno_ADC_CS1237 {
    public:
        ArduinoUno_ADC_CS1237(uint8_t SCLK, uint8_t DATA) // Constructor. (SCLK pin, DATA pin)
            { SCLK = SCLK; DOUT_DRDY = DATA; }
		
		void begin(void);
		
    void clockCycle();
    bool readBit();
    void writeBit(bool bit);
		
		int32_t readADC();
		
		void setRegister(int registertowrite, int valuetowrite);
    void setDefaultRegister(void);
    void setFullRegister(byte register_to_write);
		int getRegister();
    void read_and_printRegister();
    void read_and_print(int value);
		
    private:
		int DOUT_DRDY = 19; //DOUT/DRDY pin
		int SCLK = 13; //SCLK pin
		int32_t ADCreading; //This variable stores the ADCreading - Can be omitted with all its relevant references!
		float pga_divider = 1.0; //default value = 1. This value stores the gain value
		
		void customDelay455ns();
};

#endif