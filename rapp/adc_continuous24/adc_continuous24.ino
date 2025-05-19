#include "ArduinoUno_ADC_CS1237.h"
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

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
};

union writable_float {
  float f;
  byte bytes[4];
};


float read_n_samples_from_channel(unsigned long n_samples, bool channel) {
    float starttime = millis();
    int32_t data = 0;

    unsigned long i = 0;
    while (i < n_samples) {
        if (channel) data = adc1.readADC();
        else data = adc0.readADC();
        serial_write_32bit(data);
        if (DEBUG_CS1237) Serial.println();
        i = i + 1;
    };

    float endtime = millis();
    float elapsedtime = (endtime - starttime) / 1000;

    return elapsedtime;
}

void request_temp(String channel) {
    bool ch = parse_bool(channel);
    if (ch) sensorDS18B20_1.requestTemperatures();
    else sensorDS18B20_0.requestTemperatures();
    
}

float read_temp(String channel) {
    bool ch = parse_bool(channel);
    float temp;
    if (ch) temp = sensorDS18B20_1.getTempC(plateTemp);
    else temp = sensorDS18B20_0.getTempC(roomTemp);
    return temp;
}

void read_and_send_temp(String channel) {
  writable_float temp;
  temp.f = read_temp(channel);
  if (DEBUG_CS1237) Serial.println(temp.f);
  else Serial.write(temp.bytes, 4);
}

float read_n_samples(unsigned long n_samples, bool ch0, bool ch1) {
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

void parse_read_and_print_n_samples_dt(String command_args) {
  unsigned long n_samples = parse_int(command_args);
  unsigned long dt = parse_int(command_args);
  for (int i = 0; i < n_samples; i++){
    int32_t data0 = adc0.readADC();
    int32_t data1 = adc1.readADC();
    Serial.print( (String) i+" : " );
    serial_write_32bit(data0);
    Serial.print(" | ");
    serial_write_32bit(data1);
    Serial.println();
    delay(dt);
  }
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

bool is_conversion_complete() {
  sensorDS18B20_0.isConversionComplete();
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
            float elapsedtime;
            unsigned short n_channels;
            parse_and_read_n_samples(command_args, &elapsedtime, &n_channels);
        }
        else if (command_name == "adc_n_dt") { // Command: "adc_n_dt;nsamples;dt;"
            String command_args = getArgs(input_command);
            parse_read_and_print_n_samples_dt(command_args);
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
        else if (command_name == "complete?") {
            is_conversion_complete();
        }
        else {Serial.println("Comando no reconocido");}
    }
}

void loop(void) {
    if (Serial.available() > 0) {  // Wait to receive a signal.
        process_serial_input();
    }
}
