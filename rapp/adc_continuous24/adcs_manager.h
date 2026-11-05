#define SCLK_PIN_ADC0 6
#define DOUT_DRDY_PIN_ADC0 7
#define SCLK_PIN_ADC1 10
#define DOUT_DRDY_PIN_ADC1 11

#if (ADC_ACTIVO == ADC_ADS1232)
  #include "ArduinoUno_ADC_ADS1232.h"

  #define PDWN_ADC0 2
  #define PDWN_ADC1 8
  #define SPEED_ADC0 3
  #define SPEED_ADC1 9

  #define PROFILE PROFILE_ADS1232
  ArduinoUno_ADC_ADS1232 adc0(SCLK_PIN_ADC0, DOUT_DRDY_PIN_ADC0, PDWN_ADC0, SPEED_ADC0);
  ArduinoUno_ADC_ADS1232 adc1(SCLK_PIN_ADC1, DOUT_DRDY_PIN_ADC1, PDWN_ADC1, SPEED_ADC1);
  const byte register_to_write = 0b00000000; // 00, 00, 0bXX = SLOW or FAST, 00
  #define DELTA_VOLTS (2.49 / 8388608.0f)
#elif (ADC_ACTIVO == ADC_CS1237)
  #include "ArduinoUno_ADC_CS1237.h"

  #define PROFILE PROFILE_CS1237
  
  ArduinoUno_ADC_CS1237 adc0(SCLK_PIN_ADC0, DOUT_DRDY_PIN_ADC0);// Declare the object to work with the ArduinoUno_ADC_CS1237 library functions, specifying the pins (SCLK, DATA).
  ArduinoUno_ADC_CS1237 adc1(SCLK_PIN_ADC1, DOUT_DRDY_PIN_ADC1);
  const byte register_to_write = 0b01110000; // CH 0 input, PGA = 1, DRATE = 1280 Hz, VREF = DISABLED
  #define DELTA_VOLTS (1.24 / 8388608.0f)
#else
  #error "ADC_ACTIVO must be defined as either ADC_ADS1232 or ADC_CS1237"
#endif