#include "ArduinoUno_ADC_ADS1232.h"

void ArduinoUno_ADC_ADS1232::begin(void) {
    ads.begin(pinDOUT_DRDY, pinSCLK, pinPDWN, pinSPEED, _speed);
}

long ArduinoUno_ADC_ADS1232::readADC() {
    long value = 0;
    ERROR_t err = ads.read(value);
    if (err != NoERROR) {
        value = -255 - long(err); // Return a negative value to indicate an error, with the error code encoded
    }
    return value;
}

long ArduinoUno_ADC_ADS1232::readADCwProfiler() {
    uint32_t start = micros();
    long value = 0;
    ERROR_t err = ads.read(value);
    profiler.elapsed_time_read = micros() - start;
    if (profiler.elapsed_time_read > profiler.max_time_read) profiler.max_time_read = profiler.elapsed_time_read;
    if (profiler.elapsed_time_read < profiler.min_time_read) profiler.min_time_read = profiler.elapsed_time_read;
    return value;
}

ProfilerTimes ArduinoUno_ADC_ADS1232::getProfiler() const {
    return profiler;
}

int ArduinoUno_ADC_ADS1232::getDOUT_DRDY(){
  return pinDOUT_DRDY;
}

int ArduinoUno_ADC_ADS1232::getSCLK(){
  return pinSCLK;
}

int ArduinoUno_ADC_ADS1232::getPDWN(){
  return pinPDWN;
}

int ArduinoUno_ADC_ADS1232::getSPEED(){
  return pinSPEED;
}

void ArduinoUno_ADC_ADS1232::setRegister(int registertowrite, int valuetowrite){
    if (registertowrite == 2){
        Speed new_speed = Speed(valuetowrite);
        if (new_speed != _speed){
            _speed = new_speed;
            ads.setSpeed(_speed);
        }
    }
}

int ArduinoUno_ADC_ADS1232::getRegister(){
    return int(_speed << 2);
}

void ArduinoUno_ADC_ADS1232::setDefaultRegister(void){
    if (_speed != SLOW){
        _speed = SLOW;
        ads.setSpeed(_speed);
    }
}

void ArduinoUno_ADC_ADS1232::setFullRegister(byte register_to_write){
    Speed new_speed = Speed((register_to_write >> 2) & 0b00000001);
    if (new_speed != _speed){
        _speed = new_speed;
        ads.setSpeed(_speed);
    }
}

void ArduinoUno_ADC_ADS1232::read_and_printRegister() {
  int regvalues = getRegister(); //Read the register value
  Serial.print("Register ");
  printSerialByte(regvalues);
}

void ArduinoUno_ADC_ADS1232::printSerialByte(int value) {
    Serial.print("Value: 0b"); //Print the register value in a convenient 0bxxxxxxxx format

    for (int i = 15; i >= 8; i--) {
      Serial.print((value >> i) & 1); //Print/shift the register bits until the whole byte is printed
    }
    Serial.print(" "); // Print a newline character

    for (int i = 7; i >= 0; i--) {
      Serial.print((value >> i) & 1); //Print/shift the register bits until the whole byte is printed
    }
    Serial.println(); // Print a newline character
}