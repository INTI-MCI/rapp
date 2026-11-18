#ifndef __OPEN_ADS1115__
#define __OPEN_ADS1115__

#include <Adafruit_ADS1X15.h>

class OpenADS1115 : public Adafruit_ADS1X15 {
    protected:
        uint8_t buffer[3];

    public:
        OpenADS1115() : Adafruit_ADS1X15() {};
    
        uint16_t open_readConfigRegister();
    
};

#endif