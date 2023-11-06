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

void read_n_samples_from_channel(short n_samples, byte channel, byte skip_values){
    ads.startADCReading(MUX_BY_CHANNEL[channel], ADS_READING_MODE_CONTINUOUS);
    delay(3);  // Time we need to wait so ADC goes out of suspension
    short i = 0;
    while (i < n_samples + skip_values) {
        short data = ads.getLastConversionResults();
        if (i > skip_values - 1)
            Serial.println(data, DEC);
        i = i + 1;
        delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
    };
}

void read_n_samples_from_channel_without_print(short n_samples, byte channel){
    ads.startADCReading(MUX_BY_CHANNEL[channel], ADS_READING_MODE_CONTINUOUS);

    short i = 0;
    while (i < n_samples) {
        short data = ads.getLastConversionResults();
        i = i + 1;
        delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
    };
}

void loop(void) {
    if (Serial.available() > 0) {  // Wait to recieve a signal.
        short n_samples = Serial.parseInt();
        
        read_n_samples_from_channel(n_samples, 0, 0);      
        read_n_samples_from_channel(n_samples, 1, 0);
    }
}
