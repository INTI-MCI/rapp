#define ADC_ADS1232 1
#define ADC_CS1237 2

#define ADC_ACTIVO ADC_ADS1232
#include "adcs_manager.h"
#include "Profiling.h"
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define N_TIMES_ADC 10

#define PIN_TEMP_SENSOR_0 4
#define PIN_TEMP_SENSOR_1 5

unsigned long* times;
// times[0] is total elapsed time of parse_and_read_n_samples
// times[1, 2] = [elapsed time ch0, elapsed time ch1]
// times[3, 4] = elapsed time serial write 32 bit in serial port for ch0 and ch1, ver si hace falta guardar esto!!
const unsigned short int SERIAL_BAUDRATE = 57600;

const int dataPin0 = PIN_TEMP_SENSOR_0;   // Pin where room temperature sensors 1-Wire bus is connected
OneWire oneWire0(dataPin0);
DallasTemperature sensorDS18B20_0(&oneWire0);
DeviceAddress roomTemp = {0x28, 0xCF, 0x42, 0x76, 0xE0, 0x01, 0x3C, 0x70};

const int dataPin1 = PIN_TEMP_SENSOR_1;   // Pin where quartz plate temperature sensors 1-Wire bus is connected
OneWire oneWire1(dataPin1);
DallasTemperature sensorDS18B20_1(&oneWire1);
//Falta averiguar la dirección del otro sensor!!

DeviceAddress plateTemp = {0x90, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};
//DeviceAddress plateTemp = {0x5C, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};
//DeviceAddress plateTemp = {0x6C, 0x01, 0x55, 0x05, 0x7F, 0xA5, 0xA5, 0x66};

void setup(void) {
    Serial.begin(SERIAL_BAUDRATE);

    if (DEBUG) {
    Serial.println();
    serial_log("Comienzo de adc_continuous24::setup.");
    }

    adc0.begin();
    adc1.begin();

    if (DEBUG) {
      serial_log("adc0.begin completado.");
      Serial.println("Pin de DOUT_DRDY: " + String(adc0.getDOUT_DRDY()));
      Serial.println("Pin de SCLK: " + String(adc0.getSCLK()));
      Serial.print("Antes de setear, ");
      adc0.read_and_printRegister();
    }

    adc0.setFullRegister(register_to_write);
    adc1.setFullRegister(register_to_write);

    if (DEBUG) {
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

void serial_write_24bit(long data) {
    uint8_t buffer[3];
    buffer[0] = data >> 16;
    buffer[1] = (data >> 8) & 0xFF;
    buffer[2] = data & 0xFF;
    if (DEBUG) {
    Serial.print(buffer[0], BIN);
    Serial.print(buffer[1], BIN);
    Serial.println(buffer[2], BIN);
    Serial.println();
    }
    else Serial.write(buffer, 3);
}

void serial_write_32bit_error(ERROR_t err){
    uint8_t buffer[4];
    buffer[0] = 0x80; // Indicate an error with a specific pattern
    buffer[1] = 0x00;
    buffer[2] = 0x00;
    buffer[3] = 0xFF - (uint8_t)err; // Store the error code in the last byte
    Serial.write(buffer, 4);
}

void serial_write_32bit(long data) {
    uint8_t buffer[4];
    buffer[0] = data >> 24 & 0xFF;
    buffer[1] = (data >> 16) & 0xFF;
    buffer[2] = (data >> 8) & 0xFF;
    buffer[3] = data & 0xFF;

    if (DEBUG) {
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

    unsigned long endtime2;
    unsigned long starttime2;
    unsigned long elapsedtime2;

    measure_times |= PROFILE;
    ERROR_t err;

    if (measure_times){
      starttime = micros();
      times[3 + (unsigned short) channel] = 0; // Initialize the elapsed time for serial write for this channel
    }
    long data = 0;

    unsigned long i = 0;
    while (i < n_samples) {
        if (PROFILE) {
          if (channel) err = adc1.readADCwProfiler(data);
          else err = adc0.readADCwProfiler(data);
        }
        else {
          if (channel) err = adc1.readADC(data);
          else err = adc0.readADC(data);
        }
        if (err != NoERROR) serial_write_32bit_error(err);
        if (DEBUG or measure_times) starttime2 = micros(); // Revisar qué se quiere guardar y/o imprimir
        serial_write_32bit(data);
        if (DEBUG or measure_times) {
          endtime2 = micros();
          elapsedtime2 = endtime2 - starttime2;
          times[3 + (unsigned short) channel] += elapsedtime2;
        }
        if (DEBUG) {
          Serial.print("Elapsed time serial_write_32bit in microseconds: ");
          Serial.println(elapsedtime2);
        }
        i = i + 1;
    }

    if (measure_times or DEBUG) {
      endtime = micros();
      elapsedtime = endtime - starttime;
      times[1 + (unsigned short) channel] = elapsedtime;
    }

    if (DEBUG) {
      Serial.print("Elapsed time per sample ch");
      Serial.print((unsigned short) channel);
      Serial.print(": ");
      Serial.println(elapsedtime/n_samples);
    }
}

void request_temp(String channel) {
    bool ch, err;
    err = parse_bool(channel, ch);
    if (err) {
      Serial.println("Error parsing channel for request_temp");
      return;
    } else {
      Serial.println("ACK");
    }
    if (ch) sensorDS18B20_1.requestTemperatures();
    else sensorDS18B20_0.requestTemperatures();
}

float read_temp(String channel) {
    bool ch, err;
    err = parse_bool(channel, ch);
    if (err) {
      Serial.println("Error parsing channel for read_temp");
      return NAN;
    }
    float temp;
    if (ch) temp = sensorDS18B20_1.getTempCByIndex(0); //sensorDS18B20_1.getTempC(plateTemp);
    else temp = sensorDS18B20_0.getTempC(roomTemp); //sensorDS18B20_0.getTempCByIndex(0);
    return temp;
}

void read_and_send_temp(String channel) {
  writable_float temp;
  temp.f = read_temp(channel);
  if (isnan(temp.f)) return; else Serial.println("ACK"); 
  if (DEBUG) Serial.println(temp.f);
  else Serial.write(temp.bytes, 4);
}

void read_n_samples(unsigned long n_samples, bool ch0, bool ch1, bool measure_times) {
    if (ch0) read_n_samples_from_channel(n_samples, 0, measure_times);
    if (ch1) read_n_samples_from_channel(n_samples, 1, measure_times);
}

unsigned long parse_and_read_n_samples(String command_args, unsigned short *out_n_channels, bool measure_times) {
    bool ch0, ch1, err_ch0, err_ch1, err_n_samples;
    unsigned long n_samples;
    err_ch0 = parse_bool(command_args, ch0);
    err_ch1 = parse_bool(command_args, ch1);
    if (err_ch0 or err_ch1) {
      Serial.println("Error parsing channels for parse_and_read_n_samples");
      return 0;
    } 
    err_n_samples = parse_int(command_args, n_samples);
    if (err_n_samples) {
      Serial.println("Error parsing n_samples for parse_and_read_n_samples");
      return 0;
    } else {
      Serial.println("ACK");
    }

    read_n_samples(n_samples, ch0, ch1, measure_times);
    *out_n_channels = (unsigned short) ch0 + (unsigned short) ch1;
    if (PROFILE or measure_times) {
      times[0] = times[1] + times[2];
    }
    if (DEBUG) {
      Serial.print("Elapsed time per sample (avg) in microseconds: ");
      Serial.println(times[0] / (n_samples * *out_n_channels));
    }
    return n_samples;
}

bool parse_bool(String& command_args, bool& value) {
  int ind = command_args.indexOf(";");
  if (ind == -1) return true;
  String arg01 = command_args.substring(0, ind);
  value = !arg01.startsWith("0");
  command_args = command_args.substring(ind + 1);
  return false;
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

bool parse_int(String& command_args, unsigned long& value) {
  int ind = command_args.indexOf(";");
  if (ind == -1) return true;
  value = command_args.substring(0, ind).toInt();
  command_args = command_args.substring(ind + 1);
  return false;
}

void parse_read_and_print_n_samples_dt(String command_args) {
  // This function reads n samples every dt miliseconds and prints them in serial monitor
  unsigned long n_samples, dt_ms, time_start, total_time = 0;
  bool err_n_samples, err_dt_ms;
  ERROR_t err_adc0, err_adc1;
  long data0, data1;
  long *data_ch0, *data_ch1;

  err_n_samples = parse_int(command_args, n_samples);
  if (err_n_samples) {
    Serial.println("Error parsing n_samples for parse_read_and_print_n_samples_dt");
    return;
  }
  err_dt_ms = parse_int(command_args, dt_ms);
  if (err_dt_ms) {
    Serial.println("Error parsing dt_ms for parse_read_and_print_n_samples_dt");
    return;
  } else {
    Serial.println("ACK");
  }

  data_ch0 = (long*)malloc(n_samples * sizeof(long));
  data_ch1 = (long*)malloc(n_samples * sizeof(long));

  for (int i = 0; i < n_samples; i++){
    time_start = millis();
    err_adc0 = adc0.readADC(data0);
    data_ch0[i] = data0;
    delay(dt_ms);
    total_time += millis() - time_start;
  }
  for (int i = 0; i < n_samples; i++){
    time_start = millis();
    err_adc1 = adc1.readADC(data1);
    data_ch1[i] = data1;
    delay(dt_ms);
    total_time += millis() - time_start;
  }

  Serial.print("Average time per sample: ");
  Serial.print((float)total_time/n_samples/2.0);
  Serial.println(" ms");

  for (int i = 0; i < n_samples; i++){
    data0 = data_ch0[i];
    data1 = data_ch1[i];
    Serial.print( (String) i+" : ERROR : " );
    Serial.print((int)err_adc0);
    Serial.print(", ");
    // serial_write_32bit(data0);
    serialWriteBinary((uint8_t*)&data0, sizeof(data0));
    Serial.print(" d");
    Serial.print(data0);
    Serial.print(", ");
    Serial.print(bits_to_volts(data0));
    Serial.print(" V | ERROR : ");
    Serial.print((int)err_adc1);
    Serial.print(", ");
    // serial_write_32bit(data1);
    serialWriteBinary((uint8_t*)&data1, sizeof(data1));
    Serial.print(" d");
    Serial.print(data1);
    Serial.print(", ");
    Serial.print(bits_to_volts(data1));
    Serial.print(" V");
    Serial.println();
  }
  free(data_ch0);
  free(data_ch1);
}

float bits_to_volts(long value){
  return (float)value * DELTA_VOLTS;
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
  bool err, ch;
  err = parse_bool(channel, ch);
  if (err) Serial.println("Error parsing channel for is_conversion_complete");
  else Serial.println("ACK");
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

void send_profiled_times() {
  ProfilerTimes profiler0 = adc0.getProfiler();
  ProfilerTimes profiler1 = adc1.getProfiler();

  Serial.write((uint8_t*)&profiler0, sizeof(profiler0));
  Serial.write((uint8_t*)&profiler1, sizeof(profiler1));
}

void serialWriteHex(const uint8_t *buffer, size_t length) {
    for (size_t i = 0; i < length; i++) {
        Serial.print("0x");
        
        // Si el valor es menor a 16 (0x10), imprimimos un '0' a la izquierda 
        // para mantener siempre el formato de 2 dígitos (ej: 0x0F en vez de 0xF)
        if (buffer[i] < 16) {
            Serial.print("0");
        }
        
        // Imprimimos el valor en formato hexadecimal en mayúsculas (o minúsculas si prefieres)
        Serial.print(buffer[i], HEX);
        
        // Añadimos un espacio entre cada byte (opcional)
        if (i < length - 1) {
            Serial.print(" ");
        }
    }
}

void serialWriteBinary(const uint8_t *buffer, size_t length) {
    for (size_t i = 0; i < length; i++) {
        Serial.print("0b");
        
        // Recorremos los 8 bits de cada byte desde el más significativo (7) al menos (0)
        for (int bit = 7; bit >= 0; bit--) {
            // Evaluamos si el bit actual está en 1 o 0 usando una máscara y desplazamiento
            if ((buffer[i] >> bit) & 1) {
                Serial.print("1");
            } else {
                Serial.print("0");
            }
        }
        
        // Añadimos un espacio entre cada byte para que sea legible
        if (i < length - 1) {
            Serial.print(" ");
        }
    }
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

        }
        else if (command_name == "adc_n_dt") { // Command: "adc_n_dt;nsamples;dt;", dt in miliseconds
            String command_args = getArgs(input_command);
            parse_read_and_print_n_samples_dt(command_args);
        }
        else if (command_name == "adc_register?") {
            Serial.println("ACK");
            for (uint8_t i = 0; i < 5; i++) {
              Serial.print(String(i) + " : ");
              adc0.read_and_printRegister();
              delay(1000);
            }
        }
        else if (command_name == "adc_register") { // Command: "adc_register;xxxxxxxx;"
            Serial.println("ACK");
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
        else if (command_name == "measured_times?") {
            Serial.println("ACK");
            send_measured_times();
        }
        else if (command_name == "profiled_times?") {
            Serial.println("ACK");
            send_profiled_times();
        }
        else if (command_name == "self_calibrate") {
            Serial.println("ACK");
            ERROR_t err0 = adc0.calibrate();
            ERROR_t err1 = adc1.calibrate();
            Serial.write((uint8_t*)&err0, sizeof(err0));
            Serial.write((uint8_t*)&err1, sizeof(err1));
        }
        else if (command_name == "self_calibrate_monitor") {
            Serial.println("ACK");
            ERROR_t err0 = adc0.calibrate();
            ERROR_t err1 = adc1.calibrate();
            serialWriteHex((uint8_t*)&err0, sizeof(err0));
            Serial.print(" ");
            serialWriteHex((uint8_t*)&err1, sizeof(err1));
        }
        else if (command_name == "reset_adc") {
            Serial.println("ACK");
            adc0.reset_ADC();
            adc0.setFullRegister(register_to_write);
            adc1.reset_ADC();
            adc1.setFullRegister(register_to_write);
        }
        else {Serial.println("Comando no reconocido");}
    }
}

void loop(void) {
    if (Serial.available() > 0) {  // Wait to receive a signal.
        process_serial_input();
    }
}
