#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;
const unsigned short int SERIAL_BAUDRATE = 57600;
const bool ADS_READING_MODE_CONTINUOUS = true;
const unsigned long ADS_CONTINUOUS_MODE_DELAY_MUS = 580; // This gives ~855 SPS.

void setup(void) {
    Serial.begin(SERIAL_BAUDRATE);

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
    // This function was useful to measure the samples per second
    float starttime;
    float endtime;
    starttime = millis();
    endtime = starttime;

    ads.startADCReading(MUX_BY_CHANNEL[0], ADS_READING_MODE_CONTINUOUS);

    long N = 0;
    while ((endtime - starttime) <= 1000)
    {
        short value0_bits = ads.getLastConversionResults();
        N = N + 1;
        endtime = millis();
        delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
    }
    Serial.print("SAMPLES PER SECOND: ");
    Serial.println(N);
}

void loop(void) {

    if (Serial.available() > 0) {  // Wait to recieve a signal.

        ads.startADCReading(MUX_BY_CHANNEL[0], ADS_READING_MODE_CONTINUOUS);
        short samples = Serial.parseInt();

        short i = 0;
        while (i < samples) {
            short value0_bits = ads.getLastConversionResults();
            float value0_volts = ads.computeVolts(value0_bits);
            
            Serial.println(value0_volts, 6);
            i = i + 1;
            delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
        };

        Serial.flush();  // waits for the output data to be transmitted before continue.
    }
}
