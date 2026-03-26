#include "ArduinoUno_ADC_CS1237.h"
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define N_TIMES_ADC 10

unsigned long* times;
// times[0] is total elapsed time of parse_and_read_n_samples
const unsigned short int SERIAL_BAUDRATE = 57600;

ArduinoUno_ADC_CS1237 adc0(13,19);// Declare the object to work with the ArduinoUno_ADC_CS1237 library functions, specifying the pins (SCLK, DATA).
ArduinoUno_ADC_CS1237 adc1(11, 9);

const byte register_to_write = 0b01010000; // CH 0 input, PGA = 1, DRATE = 640 Hz, VREF = DISABLED

const int dataPin0 = 4;   // Pin where room temperature sensors 1-Wire bus is connected
OneWire oneWire0(dataPin0);
DallasTemperature sensorDS18B20_0(&oneWire0);
DeviceAddress roomTemp = {0x28, 0xCF, 0x42, 0x76, 0xE0, 0x01, 0x3C, 0x70};

const int dataPin1 = 5;   // Pin where quartz plate temperature sensors 1-Wire bus is connected
OneWire oneWire1(dataPin1);
DallasTemperature sensorDS18B20_1(&oneWire1);
//Falta averiguar la dirección del otro sensor!!

DeviceAddress plateTemp = {0x90, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};
//DeviceAddress plateTemp = {0x5C, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};
//DeviceAddress plateTemp = {0x6C, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};

void setup(void) {
    Serial.begin(SERIAL_BAUDRATE);

    if (DEBUG_CS1237) {
    Serial.println();
    serial_log("Comienzo de adc_continuous24::setup.");
    }

    adc0.begin();
    adc1.begin();

    if (DEBUG_CS1237) {
    serial_log("adc0.begin completado.");
    Serial.println("Pin de DOUT_DRDY: " + String(adc0.getDOUT_DRDY()));
    Serial.println("Pin de SCLK: " + String(adc0.getSCLK()));
    Serial.print("Antes de setear, ");
    adc0.read_and_printRegister();
    }

    adc0.setFullRegister(register_to_write);
    adc1.setFullRegister(register_to_write);

    if (DEBUG_CS1237) {
    serial_log("Registro actualizado.");
    adc0.read_and_printRegister();
    }

    sensorDS18B20_0.begin();
    sensorDS18B20_0.setResolution(12);
    sensorDS18B20_0.setWaitForConversion(false);

    sensorDS18B20_1.begin();
    sensorDS18B20_1.setResolution(12);
    sensorDS18B20_1.setWaitForConversion(false);
}

void serial_log(String msg) {
    Serial.println(String(millis()) + " ms: " + msg);
}

void serial_write_short(short data) {
    uint8_t buffer[2];
    buffer[0] = data >> 8;
    buffer[1] = data & 0xFF;
    Serial.write(buffer, 2);
}

void serial_write_24bit(int32_t data) {
    uint8_t buffer[3];
    buffer[0] = data >> 16;
    buffer[1] = (data >> 8) & 0xFF;
    buffer[2] = data & 0xFF;
    if (DEBUG_CS1237) {
    Serial.print(buffer[0], BIN);
    Serial.print(buffer[1], BIN);
    Serial.println(buffer[2], BIN);
    Serial.println();
    }
    else Serial.write(buffer, 3);
}

void serial_write_32bit(int32_t data) {
    uint8_t buffer[4];
    buffer[0] = data >> 24 & 0xFF;
    buffer[1] = (data >> 16) & 0xFF;
    buffer[2] = (data >> 8) & 0xFF;
    buffer[3] = data & 0xFF;

    if (DEBUG_CS1237) {
    serial_print_byte_bin(buffer[0]);
    Serial.print(' ');
    serial_print_byte_bin(buffer[1]);
    Serial.print(' ');
    serial_print_byte_bin(buffer[2]);
    Serial.print(' ');
    serial_print_byte_bin(buffer[3]);
    //int32_t data_signed = data;
    char float_str[10];
    char data_signed_str[60];
    dtostrf((float) data/83886.08, 5, 2, float_str);
    sprintf(data_signed_str, " ( %ld , %s %%)", data, float_str);
    Serial.print(data_signed_str);
    }
    else Serial.write(buffer, 4);
}

void serial_print_byte_bin(uint8_t regvalues) {
  for (int i = 7; i >= 0; i--) {
    Serial.print((regvalues >> i) & 1); //Print/shift the register bits until the whole byte is printed
  }
}

union writable_float {
  float f;
  byte bytes[4];
};


void read_n_samples_from_channel(unsigned long n_samples, bool channel, bool measure_times) {
    unsigned long starttime;
    unsigned long endtime;
    unsigned long elapsedtime;

    unsigned long starttime2;
    unsigned long endtime2;
    unsigned long elapsedtime2;

    measure_times |= PROFILE_CS1237;

    if (measure_times) starttime = micros();
    int32_t data = 0;

    unsigned long i = 0;
    while (i < n_samples) {
        if (measure_times) {
          if (channel) data = adc1.readADCwProfiler();
          else data = adc0.readADCwProfiler();
        }
        else {
          if (channel) data = adc1.readADC();
          else data = adc0.readADC();
        }
        if (DEBUG_CS1237 & measure_times) starttime2 = micros(); // Revisar qué se quiere guardar y/o imprimir
        serial_write_32bit(data);
        if (DEBUG_CS1237 & measure_times) {
          endtime2 = micros();
          elapsedtime2 = endtime2 - starttime2; // Tal vez no tenga sentido ver esto cuando veamos tiempos guardados en times?
          Serial.print("Elapsed time serial_write_32bit in microseconds: ");
          Serial.println(elapsedtime2);
        }
        i = i + 1;
    }

    if (measure_times) {
      endtime = micros();
      elapsedtime = endtime - starttime;
      times[1 + (unsigned short) channel] = elapsedtime;
    }

    if (measure_times) {
      Serial.print("Elapsed time per sample ch");
      Serial.print((unsigned short) channel);
      Serial.print(": ");
      Serial.println(elapsedtime/n_samples);
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
  if (DEBUG_CS1237) Serial.println(temp.f);
  else Serial.write(temp.bytes, 4);
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
    if (PROFILE_CS1237 or measure_times) {
      times[0] = times[1] + times[2];
    }
    if (measure_times) {
      Serial.print("Elapsed time per sample (avg) in microseconds: ");
      Serial.println(times[0] / (n_samples * *out_n_channels));
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

byte parse_byte(String& command_args) {
  int ind = command_args.indexOf(";");
  String arg01 = command_args.substring(0, ind);
  uint8_t start_at = arg01.startsWith("0b")?2:0;
  byte value = arg01[start_at] == '1';
  for (int i=0;i<8 && (start_at + i) < arg01.length();i++){
    value = value << 1;
    value |= arg01[start_at + i] == '1';
  }
  command_args = command_args.substring(ind + 1);
  return value;
}

unsigned long parse_int(String& command_args) {
  int ind = command_args.indexOf(";");
  unsigned long value = command_args.substring(0, ind).toInt();
  command_args = command_args.substring(ind + 1);
  return value;
}

unsigned long parse_read_and_print_n_samples_dt(String command_args) {
  unsigned long n_samples = parse_int(command_args);
  unsigned long dt = parse_int(command_args);
  unsigned long initial_time = micros();
  for (int i = 0; i < n_samples; i++){
    int32_t data0 = adc0.readADC();
    int32_t data1 = adc1.readADC();
    Serial.print( (String) i+" : " );
    serial_write_32bit(data0);
    // Serial.print(bits_to_volts(data0));
    Serial.print(" | ");
    serial_write_32bit(data1);
    Serial.println();
    delay(dt);
  }
  unsigned long final_time = micros();
  unsigned long elapsed_time = final_time - initial_time;
  return elapsed_time;
}

float bits_to_volts(int32_t value){
  return value * (1240/(2^23)) / 1000;
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

void toggle_led(int n) {  // Useful for debugging
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

void process_serial_input() {
    if (Serial.available() > 0) {
        String input_command = Serial.readStringUntil('\n');
        String command_name = input_command.substring(0,input_command.indexOf(";"));
        if (command_name == "adc") { // Command: "adc;ch0;ch1;nsamples;"
            String command_args = getArgs(input_command);
            unsigned long elapsedtime;
            unsigned short n_channels;
            bool measure_times = 0;
            times = new unsigned long[N_TIMES_ADC];

            parse_and_read_n_samples(command_args, &n_channels, measure_times);
        }
        else if (command_name == "adc_n_dt") { // Command: "adc_n_dt;nsamples;dt;", dt in miliseconds
            String command_args = getArgs(input_command);
            unsigned long initial_time = micros();
            unsigned long elapsed_time = parse_read_and_print_n_samples_dt(command_args);
            unsigned long final_time = micros();
            unsigned long elapsed_time_n = final_time - initial_time;
            unsigned long n_samples = parse_int(command_args);
            unsigned long dt = parse_int(command_args);
            unsigned long time_n = (elapsed_time_n-(dt*n_samples)) / (2*1000);
            // Serial.print("Elapsed time of n samples in miliseconds: ");
            // Serial.println(time_n);
            // Serial.print("Elapsed time of each sample (avg) in miliseconds: ");
            // Serial.println(time_n/n_samples);
        }
        else if (command_name == "adc_register?") {
            for (uint8_t i = 0; i < 5; i++) {
              Serial.print(String(i) + " : ");
              adc0.read_and_printRegister();
              delay(1000);
            }
        }
        else if (command_name == "adc_register") { // Command: "adc_register;xxxxxxxx;"
            String command_args = getArgs(input_command);
            byte register_to_write_ = parse_byte(command_args);
            adc0.setFullRegister(register_to_write_);
            adc1.setFullRegister(register_to_write_);
        }
        else if (command_name == "req-temp") { // Command: "req-temp;channel;"
            String command_args = getArgs(input_command);
            request_temp(command_args);
        }
        else if (command_name == "temp") { // Command: "temp;channel;"
            String command_args = getArgs(input_command);
            read_and_send_temp(command_args);
        }
        else if (command_name == "ready?") {
            Serial.println("yes");
            // If required, send no
        }
        else if (command_name == "complete?") { // Command: "complete?;channel;"
            String command_args = getArgs(input_command);
            is_conversion_complete(command_args);
        }
        else if (command_name == "sps?") {
            String command_args = getArgs(input_command);
            measure_SPS(command_args);
        }
        else {Serial.println("Comando no reconocido");}
    }
}

void loop(void) {
    if (Serial.available() > 0) {  // Wait to receive a signal.
        process_serial_input();
    }
}
