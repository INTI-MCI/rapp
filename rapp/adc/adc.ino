#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;
const unsigned short int SERIAL_BAUDRATE = 57600;
const bool ADS_READING_MODE_CONTINUOUS = true;
const byte ADS_READING_DELAY = 3; // Time we need to wait so ADC goes out of suspension

// 600 gives ~812 SPS when using serial_write_short(), 580 gives ~855 SPS when using println().
const unsigned long ADS_CONTINUOUS_MODE_DELAY_MUS = 580;

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

void serial_write_short(short data){
    uint8_t buffer[2];
    buffer[0] = data >> 8;
    buffer[1] = data & 0xFF;
    Serial.write(buffer, 2);  
}

void read_n_samples_from_channel(short n_samples, byte channel){
    ads.startADCReading(MUX_BY_CHANNEL[channel], ADS_READING_MODE_CONTINUOUS);
    delay(ADS_READING_DELAY);

    short i = 0;
    while (i < n_samples) {
        short data = ads.getLastConversionResults();
        Serial.println(data, DEC);
 
        i = i + 1;
        delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
    };
}

short acquire(){
    short n_samples = Serial.parseInt();

    read_n_samples_from_channel(n_samples, 0);      
    read_n_samples_from_channel(n_samples, 1);

    return n_samples;
}

void measure_SPS(){
    float starttime = millis();
    short n_samples = acquire();
    float endtime = millis();

    float elapsedtime = (endtime - starttime) / 1000;
    short sps = (n_samples / elapsedtime);

    Serial.print("SAMPLES: ");
    Serial.println(n_samples, DEC);

    Serial.print("SECONDS: ");
    Serial.println(elapsedtime, DEC);

    Serial.print("SAMPLES PER SECOND: ");
    Serial.println(sps);
}

void loop(void) {
    if (Serial.available() > 0) {  // Wait to recieve a signal.

        measure_SPS();

        //acquire()
    }
}
