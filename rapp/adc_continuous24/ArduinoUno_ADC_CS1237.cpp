#include "ArduinoUno_ADC_CS1237.h"

void ArduinoUno_ADC_CS1237::begin(void) {
//Configure SCL, SDA pins:
    pinMode(DOUT_DRDY, INPUT); //DOUT_DRDY - Input
    //digitalWrite(DOUT_DRDY, LOW); //Pull it LOW

    pinMode(SCLK, OUTPUT); //SCLK - OUTPUT
    digitalWrite(SCLK, LOW); //Pull it LOW

//Make sure the chip is awake
    if (DEBUG_CS1237) Serial.println("Clock en 0, leyendo DOUT esperando un 1");
    while (digitalRead(DOUT_DRDY) == 0) {} //Wait while DOUT_DRDY is low
    if (DEBUG_CS1237) Serial.println("1 recibido en DOUT, esperando un 0");
    while (digitalRead(DOUT_DRDY) == 1) {} //Wait while DOUT_DRDY is high

    if (DEBUG_CS1237) Serial.println("0 recibido, comienza delay");
    delay(OFF_ON_SETTLING_TIME);
    if (DEBUG_CS1237) Serial.println("Fin del delay");
}

void ArduinoUno_ADC_CS1237::clockCycle() {
    digitalWrite(SCLK, HIGH);
    customDelay455ns(); //t5
    digitalWrite(SCLK, LOW);
    customDelay455ns(); //t5
}

void ArduinoUno_ADC_CS1237::writeBit(bool bit) {
    digitalWrite(SCLK, HIGH);
    digitalWrite(DOUT_DRDY, bit); //Write the bit value into DOUT_DRDY pin
    customDelay455ns(); //t6
    digitalWrite(SCLK, LOW);
    customDelay455ns(); //t5
}

bool ArduinoUno_ADC_CS1237::readBit() {
    digitalWrite(SCLK, HIGH);
    customDelay455ns(); //t6
    bool single_bit = digitalRead(DOUT_DRDY); //Read the value from DOUT_DRDY
    digitalWrite(SCLK, LOW);
    customDelay455ns(); //t5
    return single_bit;
}

int32_t ArduinoUno_ADC_CS1237::readADC() {
//Data acquisition function - Returns int32 variable
    int previousValue = digitalRead(DOUT_DRDY);
    int newValue = digitalRead(DOUT_DRDY);
    //Wait for the DOUT_DRDY to fall LOW:
    while (previousValue - newValue != 1) {
        previousValue = newValue;
        newValue = digitalRead(DOUT_DRDY);
    }

    int32_t result = 0; //24-bit output data is stored in this variable

    delayMicroseconds(0); //t4

    //Read the 24-bits:
    for (int i = 0; i < 24; i++) {
        result <<= 1;
        result |= readBit();
        //i = 0; MSB @ bit 23
        //i = 1; MSB-1 @ bit 22
        //... i = 23; LSB @ bit 0 (not shifted, just OR'd together with the result)
    }

    //Shift bit 25-26-27 as well:
    for (uint8_t i = 0; i < 3; i++) clockCycle();

    //Check if the data is signed:
    if(result & 0x00800000) result |= 0xFF800000;

    return result;
}

int32_t ArduinoUno_ADC_CS1237::readADCwProfiler() {
//Data acquisition function - Returns int32 variable
    unsigned long time1;
    unsigned long time2;
    unsigned long elapsed_time_while;
    unsigned long max_time_while = 0;
    unsigned long min_time_while = 4294967295;
    if (PROFILE_CS1237 or DEBUG_CS1237) time1 = micros(); // Queda que se imprimen los tiempos si está en modo debug!
    int previousValue = digitalRead(DOUT_DRDY);
    int newValue = digitalRead(DOUT_DRDY);
    //Wait for the DOUT_DRDY to fall LOW:
    while (previousValue - newValue != 1) {
        previousValue = newValue;
        newValue = digitalRead(DOUT_DRDY);
    }
    if (PROFILE_CS1237 or DEBUG_CS1237) {
      time2 = micros();
      elapsed_time_while = time2 - time1;
      if (elapsed_time_while > max_time_while) max_time_while = elapsed_time_while;
      if (elapsed_time_while < min_time_while) min_time_while = elapsed_time_while;
    }
    // if (DEBUG_CS1237) {
      //Serial.print("Elapsed time while loop in microseconds: ");
      //Serial.println(elapsed_time_while);
      //}

    int32_t result = 0; //24-bit output data is stored in this variable

    delayMicroseconds(0); //t4
    unsigned long starttime;
    unsigned long endtime;
    unsigned long elapsed_time_read;
    unsigned long max_time_read = 0;
    unsigned long min_time_read = 4294967295;

    if (PROFILE_CS1237 or DEBUG_CS1237) starttime = micros();
    //Read the 24-bits:
    for (int i = 0; i < 24; i++) {
        result <<= 1;
        result |= readBit();
        //i = 0; MSB @ bit 23
        //i = 1; MSB-1 @ bit 22
        //... i = 23; LSB @ bit 0 (not shifted, just OR'd together with the result)
    }

    //Shift bit 25-26-27 as well:
    for (uint8_t i = 0; i < 3; i++) clockCycle();

    //Check if the data is signed:
    if(result & 0x00800000) result |= 0xFF800000;

    if (PROFILE_CS1237 or DEBUG_CS1237) {
      endtime = micros();
      elapsed_time_read = endtime - starttime;
      if (elapsed_time_read > max_time_read) max_time_read = elapsed_time_read;
      if (elapsed_time_read < min_time_read) min_time_read = elapsed_time_read;
    }
    // if (DEBUG_CS1237) {
      //Serial.print("Elapsed time read in microseconds: ");
      //Serial.println(elapsed_time_read);
    //}

    return result;
}

void ArduinoUno_ADC_CS1237::setRegister(int registertowrite, int valuetowrite) {
//Write a value to a specific register
    //"Arbitrary" register numbers
    //0 - Channel
    //1 - PGA
    //2 - Speed
    //3 - REF

    //Config register structure
    //bit 0-1 : Channel. 00 - A (Default), 01 - Reserved, 10 - Temperature, 11 - Internal short (maybe for offset calibration?)
    //bit 2-3 : PGA. 00 - 1, 01 - 2, 10 - 64, 11 - 128 (Default)
    //bit 4-5 : Speed. 00 - 10 Hz (Default), 01 - 40 Hz, 10 - 640 Hz, 11 - 1280 Hz
    //bit 6   : Reference. 0 - Enabled (Default), 1 - Disabled
    //bit 7   : Reserved, don't touch. If needed, set to 0
    //----------------------------------------------------------
    
    byte register_value = 0b00000000;

    int byteMask = 0b00000000; //Masking byte for writing only 1 register at a time

    switch (registertowrite)
    {
        case 0: //channel
        byteMask = 0b11111100; //when using & operator, we keep all, except channel bits
        register_value = register_value & byteMask; //Update the register_value with mask. This deletes the first two
        
            switch (valuetowrite)
            {
            case 0: // A // W0 0
                register_value = register_value | 0b00000000; //Basically keep everything as-is
                Serial.println("Channel = 0");
                break;
            case 1: // Reserved // W0 1
                //don't implement it!
                Serial.println("Channel = Reserved, invalid!");
                break;
            case 2: //Temperature // W0 2
                register_value = register_value | 0b00000010;
                Serial.println("Channel = Temp");
                break;
            case 3: //Internal short //W0 3
                register_value = register_value | 0b00000011;
                Serial.println("Channel = Short");
                break;
            }
            Serial.print("Después de setear ch, ");
            printSerialByte(register_value);
        break;
        //-------------------------------------------------------------------------------------------------------------

        case 1: //PGA
            byteMask = 0b11110011; //when using & operator, we keep all, except channel bits
            register_value = register_value & byteMask; //Update the register_value with mask. This deletes the first two

            switch (valuetowrite)
            {
            case 0: // PGA 1 //W1 0
                register_value = register_value | 0b00000000; //Basically keep everything as-is
                pga_divider = 1;
                Serial.println("PGA = 1");
                break;
            case 1: // PGA 2 //W1 1
                register_value = register_value | 0b00000100;
                pga_divider = 2;
                Serial.println("PGA = 2");
                break;
            case 2: //PGA 64 //W1 2
                register_value = register_value | 0b00001000;
                pga_divider = 64;
                Serial.println("PGA = 64");
                break;
            case 3: //PGA 128 //W1 3
                register_value = register_value | 0b00001100;
                pga_divider = 128;
                Serial.println("PGA = 128");
                break;
            }
            Serial.print("Después de setear PGA, ");
            printSerialByte(register_value);
            break;
            //-------------------------------------------------------------------------------------------------------------

        case 2: //DRATE
            byteMask = 0b11001111; //when using & operator, we keep all, except channel bits
            register_value = register_value & byteMask; //Update the register_value with mask. This deletes the first two

            switch (valuetowrite)
            {
            case 0: // 10 Hz //W2 0
                register_value = register_value | 0b00000000; //Basically keep everything as-is
                Serial.println("DRATE = 10 Hz");
                break;
            case 1: // 40 Hz //W2 1
                register_value = register_value | 0b00010000;
                Serial.println("DRATE = 40 Hz");
                break;
            case 2: //640 Hz //W2 2
                register_value = register_value | 0b00100000;
                Serial.println("DRATE = 640 Hz");
                break;
            case 3: //1280 Hz //W2 3
                register_value = register_value | 0b00110000;
                Serial.println("DRATE = 1280 Hz");
                break;
            }
            Serial.print("Después de setear sps, ");
            printSerialByte(register_value);
            break;
            //-------------------------------------------------------------------------------------------------------------
        case 3: //VREF
            if (valuetowrite == 0) //W3 0
            {
                bitWrite(register_value, 6, 0); //Enable
                Serial.println("VREF ON");
            }
            else if (valuetowrite == 1) //W3 1
            {
                bitWrite(register_value, 6, 1); //Disable
                Serial.println("VREF OFF");
            }
            else {}//Other values wont trigger anything
            Serial.print("Después de setear VREF, ");
            printSerialByte(register_value);
            break;
    }

    //Shift out 27 (24+3) bits:
    ADCreading = readADC(); //32-bit variable that stores the whole ADC reading

    pinMode(DOUT_DRDY, OUTPUT); //After the 27th SCLK pulse, set DOUT_DRDY to OUTPUT

    for (uint8_t i = 0; i < 2; i++) clockCycle(); //Emit 2 pulses (28-29)

    //SCLK 30-36, sending READ word:
    for (uint8_t i = 0; i < 7; i++) {
        writeBit(((0x65 >> (6 - i)) & 0b00000001) ? HIGH : LOW); //0x65 - WRITE
    }

    clockCycle(); //Send the 37th SCLK pulse

    if (DEBUG_CS1237) {
    Serial.print("Registro que se va a escribir: ");
    printSerialByte(register_value);
    }

    //38-45 SCLK pulses:
    for (uint8_t i = 0; i < 8; i++) {
        writeBit(((register_value >> (7 - i)) & 0b00000001) ? HIGH : LOW); //Write the register values
    }

    // Send 1 clock pulse, to set the pins of the ADCs to output and pull high
    clockCycle();

    // At the 46th SCLK, switch DOUT_DRDY to output and pull up DOUT_DRDY:
    pinMode(DOUT_DRDY, INPUT_PULLUP);
    delay(REGISTER_SETTLING_TIME);
}

void ArduinoUno_ADC_CS1237::setDefaultRegister(void) {
//Set all registers to a default value (my arbitrarily chosen default values) using setRegister
    setRegister(0, 0); //CH 0 input
    setRegister(1, 0); //PGA = 1
    setRegister(2, 2); //DRATE = 640 Hz
    setRegister(3, 1); //VREF = DISABLED
}

void ArduinoUno_ADC_CS1237::setFullRegister(byte register_to_write) {
// Set all registers to a given value
    //Config register structure
    //bit 0-1 : Channel. 00 - A (Default), 01 - Reserved, 10 - Temperature, 11 - Internal short (maybe for offset calibration?)
    //bit 2-3 : PGA. 00 - 1, 01 - 2, 10 - 64, 11 - 128 (Default)
    //bit 4-5 : Speed. 00 - 10 Hz (Default), 01 - 40 Hz, 10 - 640 Hz, 11 - 1280 Hz
    //bit 6   : Reference. 0 - Enabled (Default), 1 - Disabled
    //bit 7   : Reserved, don't touch. If needed, set to 0
    //----------------------------------------------------------

    if (DEBUG_CS1237) Serial.println("Registro que se va a escribir: "+ String(register_to_write, BIN));

    //Shift out 27 (24+3) bits:
    ADCreading = readADC(); //32-bit variable that stores the whole ADC reading

    pinMode(DOUT_DRDY, OUTPUT); //After the 27th SCLK pulse, set DOUT_DRDY to OUTPUT
    digitalWrite(DOUT_DRDY, 1); //Force DOUT_DRDY high

    for (uint8_t i = 0; i < 2; i++) clockCycle(); //Emit 2 pulses (28-29)

    //SCLK 30-36, sending READ word:
    for (uint8_t i = 0; i < 7; i++) {
      writeBit(((0x65 >> (6 - i)) & 0b00000001) ? HIGH : LOW); //0x65 - WRITE
    }

    clockCycle(); //Send the 37th SCLK pulse

    //38-45 SCLK pulses:
    for (uint8_t i = 0; i < 8; i++) {
        writeBit(((register_to_write >> (7 - i)) & 0b00000001) ? HIGH : LOW); //Write the register values
    }

    // send 1 clock pulse, to set the Pins of the ADCs to output and pull high
    clockCycle();

    // At the 46th SCLK, switch DOUT_DRDY to output and pull up DOUT_DRDY
    pinMode(DOUT_DRDY, INPUT_PULLUP);
}

int ArduinoUno_ADC_CS1237::getRegister() {
//Reads all registers. The actual values will be fetched with other, simple functions
    int registerValue; //Variable that stores the config register value

    //Shift out 27 (24+3) bits:
    ADCreading = readADC(); //32-bit variable that stores the whole ADC reading
    
    pinMode(DOUT_DRDY, OUTPUT); //After the 27th SCLK pulse, set DOUT_DRDY to OUTPUT
    digitalWrite(DOUT_DRDY, 1);

    for (uint8_t i = 0; i < 2; i++) clockCycle(); //Emit 2 pulses (28-29)

    //SCLK 30-36, sending READ word:
    for (uint8_t i = 0; i < 7; i++) {
      writeBit(((0x56 >> (6 - i)) & 0b00000001) ? HIGH : LOW); //0x56 - READ
    }

    pinMode(DOUT_DRDY, INPUT_PULLUP); //We read, so DOUT_DRDY becomes INPUT

    clockCycle(); //Send the 37th SCLK pulse
    
    //After the 37th SCLK pulse switch the direction of DOUT_DRDY. (This was moved before 37th SCLK pulse)

    registerValue = 0; //Because we are reading

    //38-45 SCLK pulses:
    for (uint8_t i = 0; i < 8; i++) {
      registerValue <<= 1;
      registerValue |= readBit();
    }

    // send 1 clock pulse, to set the Pins of the ADCs to output and pull high
    clockCycle();

    // At the 46th SCLK, switch DOUT_DRDY to output and pull up DOUT_DRDY.
    pinMode(DOUT_DRDY, INPUT_PULLUP); //Ready to receive the DOUT_DRDY to perform a new acquisition

    return registerValue;
}

void ArduinoUno_ADC_CS1237::read_and_printRegister() {
  int regvalues = getRegister(); //Read the register value
  Serial.print("Register ");
  printSerialByte(regvalues);
}

void ArduinoUno_ADC_CS1237::printSerialByte(int value) {
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

int ArduinoUno_ADC_CS1237::getDOUT_DRDY(){
  return DOUT_DRDY;
}

int ArduinoUno_ADC_CS1237::getSCLK(){
  return SCLK;
}

//-------------------------------------------------------------------------------------------------------------
//This is valid for Arduino Uno, make sure you adjust it for your own MCU based on its clock speed.  
#define DELAY_455_NS asm volatile ("nop\n\t" "nop\n\t" "nop\n\t")
//In Arduino Uno: 1 / 16 MHz = 62.5 ns (1 cycle time of the MCU)
//3 NOP is 3 x 62.5 ns = 187.5 ns

void ArduinoUno_ADC_CS1237::customDelay455ns() 
{
  // Adjust the number of cycles based on the calculated value
  // This may need to be fine-tuned based on the actual execution time
  for (int i = 0; i < 2; ++i) // 455 ns/ 187.5 ns = 2.4. Since 455 is a minimum req, I increased to 10. There's no max value...
  {
    DELAY_455_NS;
  }
  //So actually, this delay is more like 1875 ns. But it seems to work well. It is stable.
  //However, always refer to the clock cycle of your chosen MCU!!!
}
//--------------------------------------------------------------------------------------------------