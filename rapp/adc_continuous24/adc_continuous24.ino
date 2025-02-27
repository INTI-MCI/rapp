#include "ArduinoUno_ADC_CS1237.h"
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

ArduinoUno_ADC_CS1237 adc(13,19);// Declare the object to work with the ArduinoUno_ADC_CS1237 library functions, specifying the pins (SCLK, DATA). You can specify any Arduino pins.

const unsigned short int SERIAL_BAUDRATE = 57600;

// ADS_READING_DELAY: Time in miliseconds we need to wait so ADC goes out of suspension.
// Otherwise we get repeated values at the beginning of each channel.
const byte ADS_READING_DELAY = 5; //?? en el ejemplo hace un delay(3000) despues de Serial.begin() y del while (!Serial) {} porque despues manda un print

// ADS_CONTINUOUS_MODE_DELAY_MUS: Time we need to wait between each read.
// 600 gives ~812 SPS when using serial_write_short(), 580 gives ~855 SPS when using println().
const unsigned long ADS_CONTINUOUS_MODE_DELAY_MUS = 600; //?? Sería el customDelay455ns()?

const int pinDatosDQ = 4;   // Pin where bus 1-Wire is connected
OneWire oneWireObjeto(pinDatosDQ);
DallasTemperature sensorDS18B20(&oneWireObjeto);

DeviceAddress sensorVaina = {0x28, 0xCF, 0x42, 0x76, 0xE0, 0x01, 0x3C, 0x70};

const byte register_to_write = 0b01110000;

void setup(void) {
    Serial.begin(SERIAL_BAUDRATE);

    adc.begin();
	  //adc.setDefaultRegister(); // CH 0 input, PGA = 1, DRATE = 1280 Hz, VREF = DISABLED
    Serial.print("Antes de setear, ");
    adc.read_and_printRegister();

    adc.setFullRegister(register_to_write);

    adc.read_and_printRegister();

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

void serial_write_24bit(int32_t data){
    uint8_t buffer[3];
    buffer[0] = data >> 16;
    buffer[1] = (data >> 8) & 0xFF;
    buffer[2] = data & 0xFF;
    Serial.print(buffer[0], BIN);
    Serial.print(buffer[1], BIN);
    Serial.println(buffer[2], BIN);
    //Serial.println();
    //Serial.write(buffer, 3);
}

void serial_write_32bit(int32_t data){
    uint8_t buffer[4];
    buffer[0] = data >> 24 & 0xFF;
    buffer[1] = (data >> 16) & 0xFF;
    buffer[2] = (data >> 8) & 0xFF;
    buffer[3] = data & 0xFF;
    serial_print_byte_bin(buffer[0]);
    Serial.print(' ');
    serial_print_byte_bin(buffer[1]);
    Serial.print(' ');
    serial_print_byte_bin(buffer[2]);
    Serial.print(' ');
    serial_print_byte_bin(buffer[3]);
    //int32_t data_signed = data;
    char data_signed_str[30];
    sprintf(data_signed_str, " ( %ld )", data);
    Serial.print(data_signed_str);
    Serial.println();
    Serial.println();
    //Serial.write(buffer, 4);
}

void serial_print_byte_bin(uint8_t regvalues){
  for (int i = 7; i >= 0; i--) 
  {
    Serial.print((regvalues >> i) & 1); //Print/shift the register bits until the whole byte is printed
  }
};

union writable_float {
  float f;
  byte bytes[4];
};


float read_n_samples_from_channel(unsigned long n_samples, byte channel){
    //ads.startADCReading(MUX_BY_CHANNEL[channel], ADS_READING_MODE_CONTINUOUS);
    //delay(ADS_READING_DELAY); //(puede ser que no haga falta?)
    float starttime = millis();
    
    unsigned long i = 0;
    while (i < n_samples) {
        //short data = ads.getLastConversionResults();
        int32_t data = adc.readADC();
		// Serial.println(data, DEC);  // Useful to test stuff from Serial Monitor.
        serial_write_32bit(data);

        i = i + 1;
        //delayMicroseconds(ADS_CONTINUOUS_MODE_DELAY_MUS);
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
  int ind = command_args.indexOf(";");
  unsigned long value = command_args.substring(0, ind).toInt();
  command_args = command_args.substring(ind + 1);
  return value;
}

void parse_read_and_print_n_samples_dt(String command_args){
  unsigned long n_samples = parse_int(command_args);
  unsigned long dt = parse_int(command_args);
  for (int i = 0; i < n_samples; i++){
    int32_t data = adc.readADC();
    Serial.print( (String) i+" : " );
    serial_write_32bit(data);
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
        } else if (command_name == "adc_n_dt?"){
            int ind = input_command.indexOf(";") + 1;
            String command_args = input_command.substring(ind);
            parse_read_and_print_n_samples_dt(command_args);
        } else if (command_name == "adc_register?"){
            for (uint8_t i = 0; i < 10; i++) {
              adc.read_and_printRegister();
              delay(1000);
            }
        } else if (command_name == "req-temp?") {
            request_temp();
        } else if (command_name == "temp?") {
            read_and_send_temp();
        } else if (command_name == "ready?") {
            Serial.println("yes"); 
            // If required, send no
        } else if (command_name == "complete?") {
            is_conversion_complete();
        } else {
            Serial.println("Comando no reconocido");
        }
    }
}

void loop(void) {
    if (Serial.available() > 0) {  // Wait to receive a signal.
        process_serial_input();
    }
}
