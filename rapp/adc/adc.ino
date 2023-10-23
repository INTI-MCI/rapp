#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;

const float multiplier = 0.125F;
const unsigned short int BAUDRATE = 57600;

void setup(void) {
  Serial.begin(BAUDRATE);
  delay(200);

  // ads.setGain(GAIN_TWOTHIRDS);  // +/- 6.144V  1 bit = 0.1875mV (default)
  ads.setGain(GAIN_ONE);        // +/- 4.096V  1 bit = 0.125mV
  // ads.setGain(GAIN_TWO);        // +/- 2.048V  1 bit = 0.0625mV
  // ads.setGain(GAIN_FOUR);       // +/- 1.024V  1 bit = 0.03125mV
  // ads.setGain(GAIN_EIGHT);      // +/- 0.512V  1 bit = 0.015625mV
  // ads.setGain(GAIN_SIXTEEN);    // +/- 0.256V  1 bit = 0.0078125mV 

  ads.begin();
}

void loop(void) {
  short value0_bits = ads.readADC_SingleEnded(0);
  short value1_bits = ads.readADC_SingleEnded(1);

  float value0 = bits2volts(value0_bits);
  float value1 = bits2volts(value1_bits);

  Serial.print(value0);
  Serial.print(',');
  Serial.println(value1);
 
  delay(1000);
}


float bits2volts(short x) {
   return x * multiplier / 1000.0F;
}
