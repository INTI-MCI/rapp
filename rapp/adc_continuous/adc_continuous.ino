#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;
const unsigned short int SERIAL_BAUDRATE = 57600;
const bool ADS_READING_MODE_CONTINUOUS = true;

// ADS_READING_DELAY: Time we need to wait so ADC goes out of suspension.
// Otherwise we get repeated values at the beginning of each channel.
const byte ADS_READING_DELAY = 5;

// ADS_CONTINUOUS_MODE_DELAY_MUS: Time we need to wait between each read.
// 600 gives ~812 SPS when using serial_write_short(), 580 gives ~855 SPS when using println().
const unsigned long ADS_CONTINUOUS_MODE_DELAY_MUS = 600;

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

    // When we don't use terminator character, this helps to reduce the parseInt() delay.
    // Instead of setting this, for an optimal result is better to ALWAYS use terminator character.
    // Serial.setTimeout(10);
}

void serial_write_short(short data){
    uint8_t buffer[2];
    buffer[0] = data >> 8;
    buffer[1] = data & 0xFF;
    Serial.write(buffer, 2);  
}

void read_n_samples_from_channel(unsigned long n_samples, char channel){
    ads.startADCReading(MUX_BY_CHANNEL[channel], ADS_READING_MODE_CONTINUOUS);
    delay(ADS_READING_DELAY);

    unsigned long i = 0;
    while (i < n_samples) {
        short data = ads.getLastConversionResults();
        // Serial.println(data, DEC);  // Useful to test stuff from Serial Monitor.
        serial_write_short(data);

        i = i + 1;
        delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
    };
}

void read_n_samples(unsigned long n_samples, bool ch0, bool ch1){
    if (ch0) {
        read_n_samples_from_channel(n_samples, 0);
    }
    if (ch1) {
        read_n_samples_from_channel(n_samples, 1);
    }
}

unsigned long parse_and_read_n_samples() {
    bool ch0 = parse_bool();
    bool ch1 = parse_bool();
    unsigned long n_samples = parse_int();
    read_n_samples(n_samples, ch0, ch1);

    return n_samples;
}

unsigned long parse_int() {
    unsigned long value = Serial.parseInt();
    Serial.read(); // Remove next char (terminator) from buffer.
    return value;
}

bool parse_bool() {
    bool value = Serial.parseInt();
    Serial.read(); // Remove next char (terminator) from buffer.
    return value;
}

void measure_SPS() {
    float starttime = millis();
    unsigned long n_samples = parse_and_read_n_samples();
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

        // measure_SPS();

        parse_and_read_n_samples();

    }
}
