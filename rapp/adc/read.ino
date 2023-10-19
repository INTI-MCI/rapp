#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;
float value0;
float value1;

void setup(void) {
  Serial.begin(57600);
  delay(200);

// Cambia factor de escala: (Referencia: 4.096 V, Factor de escala: 0.125 mV)
  ads.setGain(GAIN_ONE); 
  ads.begin();
}

void loop(void) {
  short value0 = ads.readADC_SingleEnded(0);
  short value1 = ads.readADC_SingleEnded(1);

  Serial.print(value0);
  Serial.print(',');
  Serial.println(value1);
 
  delay(1000);
}
