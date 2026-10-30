#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>

Adafruit_ADS1115 ads;
unsigned long* times;
// times[0] is total elapsed time of parse_and_read_n_samples
// times[1, 2] = [elapsed time ch0, elapsed time ch1]
// times[3, 4] = elapsed time serial write 16/32 bit in serial port for ch0 and ch1, ver si hace falta guardar esto!!
const unsigned short int SERIAL_BAUDRATE = 57600;
const bool ADS_READING_MODE_CONTINUOUS = true;
const int dataPin0 = 4;   // Pin where room temperature sensors 1-Wire bus is connected
OneWire oneWire0(dataPin0);
DallasTemperature sensorDS18B20_0(&oneWire0);
DeviceAddress roomTemp = {0x28, 0xCF, 0x42, 0x76, 0xE0, 0x01, 0x3C, 0x70};

const int dataPin1 = 5;   // Pin where quartz plate temperature sensors 1-Wire bus is connected
OneWire oneWire1(dataPin1);
DallasTemperature sensorDS18B20_1(&oneWire1);
//Falta averiguar la dirección del otro sensor!!

// ADS_READING_DELAY: Time in miliseconds we need to wait so ADC goes out of suspension.
// Otherwise we get repeated values at the beginning of each channel.
const byte ADS_READING_DELAY = 5;

// ADS_CONTINUOUS_MODE_DELAY_MUS: Time we need to wait between each read.
// 600 gives ~812 SPS when using serial_write_short(), 580 gives ~855 SPS when using println().
const unsigned long ADS_CONTINUOUS_MODE_DELAY_MUS = 600;

DeviceAddress plateTemp = {0x90, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};
//DeviceAddress plateTemp = {0x5C, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};
//DeviceAddress plateTemp = {0x6C, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};

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
    sensorDS18B20_0.begin();
    sensorDS18B20_0.setResolution(12);
    sensorDS18B20_0.setWaitForConversion(false);

    sensorDS18B20_1.begin();
    sensorDS18B20_1.setResolution(12);
    sensorDS18B20_1.setWaitForConversion(false);
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


void read_n_samples_from_channel(unsigned long n_samples, byte channel, bool measure_times) {
    unsigned long starttime;
    unsigned long endtime;
    unsigned long elapsedtime;

    unsigned long endtime2;
    unsigned long starttime2;
    unsigned long elapsedtime2;
    
    ads.startADCReading(MUX_BY_CHANNEL[channel], ADS_READING_MODE_CONTINUOUS);
    delay(ADS_READING_DELAY);
    if (measure_times){
      starttime = micros();
      times[3 + (unsigned short) channel] = 0; // Initialize the elapsed time for serial write for this channel
    }
    short data = 0;
    
    unsigned long i = 0;
    while (i < n_samples) {
        data = ads.getLastConversionResults();
        // Serial.println(data, DEC);  // Useful to test stuff from Serial Monitor.
        if (measure_times) starttime2 = micros(); // Revisar qué se quiere guardar y/o imprimir
        serial_write_short(data);
        if (measure_times) {
            endtime2 = micros();
            elapsedtime2 = endtime2 - starttime2;
            times[3 + (unsigned short) channel] += elapsedtime2;
        }

        i = i + 1;
        delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
    };
    
    if (measure_times) {
      endtime = micros();
      elapsedtime = endtime - starttime;
      times[1 + (unsigned short) channel] = elapsedtime;
    }

}

void request_temp(String channel) {
    bool ch = parse_bool(channel);
    if (ch) sensorDS18B20_1.requestTemperatures();
    else sensorDS18B20_0.requestTemperatures();
    
}

float read_temp(String channel) {
    bool ch = parse_bool(channel);
    float temp;
    if (ch) temp = sensorDS18B20_1.getTempCByIndex(0); //sensorDS18B20_1.getTempC(plateTemp);
    else temp = sensorDS18B20_0.getTempC(roomTemp); //sensorDS18B20_0.getTempCByIndex(0);
    return temp;
}

void read_and_send_temp(String channel) {
  writable_float temp;
  temp.f = read_temp(channel);
  Serial.write(temp.bytes, 4);
}

void read_n_samples(unsigned long n_samples, bool ch0, bool ch1, bool measure_times) {
    if (ch0) read_n_samples_from_channel(n_samples, 0, measure_times);
    if (ch1) read_n_samples_from_channel(n_samples, 1, measure_times);
}

unsigned long parse_and_read_n_samples(String command_args, unsigned short *out_n_channels, bool measure_times) {
    bool ch0 = parse_bool(command_args);
    bool ch1 = parse_bool(command_args);
    unsigned long n_samples = parse_int(command_args);
    read_n_samples(n_samples, ch0, ch1, measure_times);
    *out_n_channels = (unsigned short) ch0 + (unsigned short) ch1;
    if (measure_times) {
      times[0] = times[1] + times[2];
    }
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
  int ind = command_args.indexOf(";");
  unsigned long value = command_args.substring(0, ind).toInt();
  command_args = command_args.substring(ind + 1);
  return value;
}

void measure_SPS(String command_args) {
    unsigned short n_channels;
    bool measure_times = 1;

    unsigned long n_samples = parse_and_read_n_samples(command_args, &n_channels, measure_times);
    
    float sps = (n_samples * n_channels / times[0]);

    Serial.print("\nSAMPLES: ");
    Serial.println(n_samples, DEC);

    Serial.print("SECONDS: ");
    Serial.println(times[0], DEC);

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

bool is_conversion_complete(String channel) {
  bool ch = parse_bool(channel);
  bool answer;
  if (ch) answer = sensorDS18B20_1.isConversionComplete(); //sensorDS18B20_1.getTempC(plateTemp);
  else answer = sensorDS18B20_0.isConversionComplete(); //sensorDS18B20_0.getTempCByIndex(0);
  Serial.write(answer);
}

String getArgs(String in_command) {
    int ind = in_command.indexOf(";") + 1;
    String command_arguments = in_command.substring(ind);
    return command_arguments;
}

void send_measured_times() {
  Serial.write((uint8_t*)times, sizeof(unsigned long) * 5); // Send times[0-4]
}

void process_serial_input() {
    if (Serial.available() > 0) {
        String input_command = Serial.readStringUntil('\n');
        String command_name = input_command.substring(0,input_command.indexOf(";"));
        if (command_name == "adc") { // Command: "adc;ch0;ch1;nsamples;"
            String command_args = getArgs(input_command);
            unsigned short n_channels;
            bool measure_times = 0;
            times = new unsigned long[N_TIMES_ADC];

            parse_and_read_n_samples(command_args, &n_channels, measure_times);
        } else if (command_name == "req-temp") { // Command: "req-temp;channel;"
            String command_args = getArgs(input_command);
            request_temp(command_args);
        }else if (command_name == "temp") { // Command: "temp;channel;"
            String command_args = getArgs(input_command);
            read_and_send_temp(command_args);
        } else if (command_name == "ready?") {
            Serial.println("yes"); 
            // If required, send no
        } else if (command_name == "complete?") { // Command: "complete?;channel;"
            String command_args = getArgs(input_command);
            is_conversion_complete(command_args);
        }
        else if (command_name == "sps?") {
            String command_args = getArgs(input_command);
            measure_SPS(command_args);
        }
        else if (command_name == "measured_times?") {
            send_measured_times();
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
