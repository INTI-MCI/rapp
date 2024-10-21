#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define RATE_ADS1115_860SPS (0x00E0)

Adafruit_ADS1115 ads;
OneWire oneWireObjeto(pinDatosDQ);
DallasTemperature sensorDS18B20(&oneWireObjeto);

const float multiplier = 0.125F;
const unsigned short int BAUDRATE = 57600;
const int pinDatosDQ = 4;   // Pin donde se conecta el bus 1-Wire

DeviceAddress sensorVaina = {0x28, 0xCF, 0x42, 0x76, 0xE0, 0x01, 0x3C, 0x70};

void setup(void) {
  Serial.begin(BAUDRATE);

  // ads.setGain(GAIN_TWOTHIRDS);  // +/- 6.144V  1 bit = 0.1875mV (default)
  ads.setGain(GAIN_ONE);        // +/- 4.096V  1 bit = 0.125mV
  // ads.setGain(GAIN_TWO);        // +/- 2.048V  1 bit = 0.0625mV
  // ads.setGain(GAIN_FOUR);       // +/- 1.024V  1 bit = 0.03125mV
  // ads.setGain(GAIN_EIGHT);      // +/- 0.512V  1 bit = 0.015625mV
  // ads.setGain(GAIN_SIXTEEN);    // +/- 0.256V  1 bit = 0.0078125mV 

  ads.setDataRate(RATE_ADS1115_860SPS);
  ads.begin();

  sensorDS18B20.begin();
  sensorDS18B20.setResolution(12);
}

void loop(void) {
  short value0_bits = ads.readADC_SingleEnded(0);
  short value1_bits = ads.readADC_SingleEnded(1);
  //value1_bits = ads.readADC_SingleEnded(1);
  sensorDS18B20.requestTemperatures();
  

  float value0 = bits2volts(value0_bits);
  float value1 = bits2volts(value1_bits);
  float temp = sensorDS18B20.getTempC(sensorVaina)

  Serial.print(value0, 6);
  Serial.print(',');
  Serial.print(value1, 6);
  Serial.print(',');
  Serial.println(temp, 6);

}


float bits2volts(short x) {
   return x * multiplier / 1000.0F;
}
