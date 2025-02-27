#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>

Adafruit_ADS1115 ads;
const unsigned short int SERIAL_BAUDRATE = 57600;
const bool ADS_READING_MODE_CONTINUOUS = true;
const int pinDatosDQ = 4;   // Pin donde se conecta el bus 1-Wire
OneWire oneWireObjeto(pinDatosDQ);
DallasTemperature sensorDS18B20(&oneWireObjeto);

// ADS_READING_DELAY: Time in miliseconds we need to wait so ADC goes out of suspension.
// Otherwise we get repeated values at the beginning of each channel.
const byte ADS_READING_DELAY = 5;

// ADS_CONTINUOUS_MODE_DELAY_MUS: Time we need to wait between each read.
// 600 gives ~812 SPS when using serial_write_short(), 580 gives ~855 SPS when using println().
const unsigned long ADS_CONTINUOUS_MODE_DELAY_MUS = 600;

DeviceAddress sensorVaina = {0x28, 0xCF, 0x42, 0x76, 0xE0, 0x01, 0x3C, 0x70};

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
    sensorDS18B20.begin();
    sensorDS18B20.setResolution(12);
    sensorDS18B20.setWaitForConversion(false);
}

void serial_write_short(short data){
    uint8_t buffer[2];
    buffer[0] = data >> 8;
    buffer[1] = data & 0xFF;
    Serial.write(buffer, 2);
}

union writable_float {
  float f;
  byte bytes[4];
};


float read_n_samples_from_channel(unsigned long n_samples, byte channel){
    ads.startADCReading(MUX_BY_CHANNEL[channel], ADS_READING_MODE_CONTINUOUS);
    delay(ADS_READING_DELAY);
    float starttime = millis();
    
    unsigned long i = 0;
    while (i < n_samples) {
        short data = ads.getLastConversionResults();
        // Serial.println(data, DEC);  // Useful to test stuff from Serial Monitor.
        serial_write_short(data);

        i = i + 1;
        delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
    };
    
    float endtime = millis();
    float elapsedtime = (endtime - starttime) / 1000;
    
    return elapsedtime;
}

void request_temp(void){
  sensorDS18B20.requestTemperatures();
}

float read_temp(void){
    float temp = sensorDS18B20.getTempC(sensorVaina);
    return temp;
}

void read_and_send_temp(void){
  writable_float temp;
  temp.f = read_temp();
  Serial.write(temp.bytes, 4);
}

float read_n_samples(unsigned long n_samples, bool ch0, bool ch1){
    float elapsedtime_ch0 = 0, elapsedtime_ch1 = 0, elapsedtime = 0;
    if (ch0) {
        elapsedtime_ch0 = read_n_samples_from_channel(n_samples, 0);
    }
    if (ch1) {
        elapsedtime_ch1 = read_n_samples_from_channel(n_samples, 1);
    }
    elapsedtime = elapsedtime_ch0 + elapsedtime_ch1;
    return elapsedtime;
}

unsigned long parse_and_read_n_samples(String command_args, float *out_elapsedtime, unsigned short *out_n_channels) {
    bool ch0 = parse_bool(command_args);
    bool ch1 = parse_bool(command_args);
    unsigned long n_samples = parse_int(command_args);
    *out_elapsedtime = read_n_samples(n_samples, ch0, ch1);
    *out_n_channels = (unsigned short) ch0 + (unsigned short) ch1;
    return n_samples;
}

bool parse_bool(String& command_args) {
  int ind = command_args.indexOf(";");
  String arg01 = command_args.substring(0, ind);
  bool value = arg01.startsWith("1");
  command_args = command_args.substring(ind + 1);
  return value;
}

unsigned long parse_int(String& command_args) {
  unsigned long value = command_args.substring(0, command_args.indexOf("\n")).toInt();
  return value;
}

void measure_SPS() {
    float elapsedtime;
    unsigned short n_channels;
    String command_args;
    unsigned long n_samples = parse_and_read_n_samples(command_args, &elapsedtime, &n_channels);
    
    short sps = (n_samples * n_channels / elapsedtime);

    Serial.print("\nSAMPLES: ");
    Serial.println(n_samples, DEC);

    Serial.print("SECONDS: ");
    Serial.println(elapsedtime, DEC);

    Serial.print("SAMPLES PER SECOND: ");
    Serial.println(sps);
}

void toggle_led(int n){  // Useful for debugging
  const int ledPin = 13;
  bool ledState = LOW;
  pinMode(ledPin, OUTPUT);
  for (int i = 0; i < 2*n; i++) {
    digitalWrite(ledPin, ledState);
    ledState = !ledState;
    delay(200);
  }
}

bool is_conversion_complete() {
  sensorDS18B20.isConversionComplete();
}

void process_serial_input() {
    if (Serial.available() > 0) {
        String input_command = Serial.readStringUntil('\n');
        String command_name = input_command.substring(0,input_command.indexOf(";"));
        if (command_name == "adc?") {
            int ind = input_command.indexOf(";") + 1;
            String command_args = input_command.substring(ind);
            float elapsedtime;
            unsigned short n_channels;
            parse_and_read_n_samples(command_args, &elapsedtime, &n_channels);
        } else if (command_name == "req-temp?") {
            request_temp();
        }else if (command_name == "temp?") {
            read_and_send_temp();
        } else if (command_name == "ready?") {
            Serial.println("yes"); 
            // If required, send no
        } else if (command_name == "complete?") {
            is_conversion_complete();
        }else {
            Serial.println("Comando no reconocido");
        }
    }
}

void loop(void) {
    if (Serial.available() > 0) {  // Wait to receive a signal.
        process_serial_input();
    }
}
