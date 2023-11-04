#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;
const float multiplier = 0.125F;
const unsigned short int BAUDRATE = 57600;
const bool ADC_READING_MODE_CONTINUOUS = true;
const unsigned long CONTINUOUS_MODE_DELAY_MUS = 580; // This gives ~855 SPS.

void setup(void) {
    Serial.begin(BAUDRATE);

    // Change scale factor:
    // ads.setGain(GAIN_TWOTHIRDS);  // +/- 6.144V  1 bit = 0.1875mV (default)
    ads.setGain(GAIN_ONE);        // +/- 4.096V  1 bit = 0.125mV
    // ads.setGain(GAIN_TWO);        // +/- 2.048V  1 bit = 0.0625mV
    // ads.setGain(GAIN_FOUR);       // +/- 1.024V  1 bit = 0.03125mV
    // ads.setGain(GAIN_EIGHT);      // +/- 0.512V  1 bit = 0.015625mV
    // ads.setGain(GAIN_SIXTEEN);    // +/- 0.256V  1 bit = 0.0078125mV

    ads.setDataRate(RATE_ADS1115_860SPS);
    ads.begin(); 
}

void measure_sps() {
    float starttime;
    float endtime;
    starttime = millis();
    endtime = starttime;

    ads.startADCReading(MUX_BY_CHANNEL[0], ADC_READING_MODE_CONTINUOUS);

    long N = 0;
    while ((endtime - starttime) <= 1000)
    {
        short value0_bits = ads.getLastConversionResults();
        N = N + 1;
        endtime = millis();
        delayMicroseconds(CONTINUOUS_MODE_DELAY_MUS);
    }
    Serial.print("MUESTRAS POR SEGUNDO:");
    Serial.println(N);
}
void loop(void) {
    measure_sps();

    //  float value0 = bits2volts(value0_bits);
    //  float value1 = bits2volts(value1_bits);

    //  Serial.print(value0, 6);
    //  Serial.print(',');
    //  Serial.println(value1, 6);
}


float bits2volts(float x) {
  return x * multiplier / 1000.0F;
}
