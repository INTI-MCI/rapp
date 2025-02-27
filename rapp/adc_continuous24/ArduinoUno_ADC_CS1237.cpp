#include "ArduinoUno_ADC_CS1237.h"													//

#define OFF_ON_SETTLING_TIME 1000 //
#define REGISTER_SETTLING_TIME 350 //

void ArduinoUno_ADC_CS1237::begin(void){											//	Parameter: none
	//	Configure SCL, SDA pins:												//
		pinMode(DOUT_DRDY, INPUT); //DRDY - Input
		//digitalWrite(DOUT_DRDY, LOW); //Pull it LOW

		pinMode(SCLK, OUTPUT); //SCLK - OUTPUT
		digitalWrite(SCLK, LOW); //Pull it LOW										//

	//Make sure the chip is awake
		while (digitalRead(DOUT_DRDY) == 0) {} //Wait while DRDY is low
		while (digitalRead(DOUT_DRDY) == 1) {} //Wait while DRDY is high
		delay(OFF_ON_SETTLING_TIME);
}

void ArduinoUno_ADC_CS1237::clockCycle()
{
  digitalWrite(SCLK, HIGH);
  customDelay455ns(); //t6
  digitalWrite(SCLK, LOW);
  customDelay455ns(); //t6
}

void ArduinoUno_ADC_CS1237::writeBit(bool bit)
{
  digitalWrite(SCLK, HIGH);
  customDelay455ns();
  digitalWrite(DOUT_DRDY, bit); //Write the register values        
  digitalWrite(SCLK, LOW);
  customDelay455ns();
}

bool ArduinoUno_ADC_CS1237::readBit()
{
  digitalWrite(SCLK, HIGH);
  customDelay455ns(); //t6
  bool single_bit = digitalRead(DOUT_DRDY); //Acá estamos suponiendo que para leer un bit alcanza el tiempo de espera del delay455
  digitalWrite(SCLK, LOW);
  customDelay455ns(); //t6
  return single_bit;
}

void ArduinoUno_ADC_CS1237::setDefaultRegister(void) {
	//Set all registers to a default value (my arbitrarily chosen default values)
		setRegister(0, 0); //CH 0 input
		setRegister(1, 0); //PGA = 1
		setRegister(2, 3); //DRATE = 1280 Hz
		setRegister(3, 1); //VREF = DISABLED

}

int32_t ArduinoUno_ADC_CS1237::readADC() //Data acquisition function - Returns a long variable
{
    while (digitalRead(DOUT_DRDY) == 1) //Wait for the DRDY/DOUT to fall LOW
    {
        //Wait until dout/drdy becomes 0
    }
    //DRDY was low, so we can proceed further

    int32_t result = 0; //24-bit output data is stored in this variable

    delayMicroseconds(1); //t4 (could be zero, actually)

    for (int i = 0; i < 24; i++) //Read the 24-bits
    {
      result <<= 1; 
      result |= readBit();
      //i = 0; MSB @ bit 23
      //i = 1; MSB-1 @ bit 22
      //... i = 23; LSB @ bit 0 (not shifted, just OR'd together with the result)
    }

    //Shift bit 25-26-27 as well.
    for (uint8_t i = 0; i < 3; i++) clockCycle();

	  if( result & 0x00800000 ){ result |= 0xFF800000; }

    return result;
}

void ArduinoUno_ADC_CS1237::setRegister(int registertowrite, int valuetowrite)
{
    //"Arbitrary" register numbers
    //0 - Channel
    //1 - PGA
    //2 - Speed
    //3 - REF

    //Config register structure
    //bit 0-1 : Channel. 00 - A, 01 - reserved, 10 - Temperature, 11 - internal short (maybe for offset calibration?)
    //bit 2-3 : PGA. 00 - 1, 01 - 2, 10 - 64, 11 - 128
    //bit 4-5 : speed. 00 - 10 Hz, 01 - 40 Hz, 10 - 640 Hz, 11 - 1280 Hz
    //bit 6   : Reference. Default is enabled which is 0.
    //bit 7   : reserved, don't touch
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
                //dont implement it!
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
            read_and_print(register_value);
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
            read_and_print(register_value);
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
            read_and_print(register_value);
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
            read_and_print(register_value);
            break;
    }

    //Shift out 27 (24+3) bits
    ADCreading = readADC(); //32-bit variable that stores the whole ADC reading

    pinMode(DOUT_DRDY, OUTPUT); //After the 27th SCLK pulse, set DOUT to OUTPUT

    for (uint8_t i = 0; i < 2; i++) clockCycle(); //Emit 2 pulses (28-29)

    for (uint8_t i = 0; i < 7; i++) //SCLK 30-36, sending READ word
    {
        writeBit(((0x65 >> (6 - i)) & 0b00000001) ? HIGH : LOW); //0x65 - WRITE
    }

    clockCycle(); //Send the 37th SCLK pulse

    Serial.print("Registro que se va a escribir: ");
    read_and_print(register_value);

    for (uint8_t i = 0; i < 8; i++) //38-45 SCLK pulses
    {
        writeBit(((register_value >> (7 - i)) & 0b00000001) ? HIGH : LOW); //Write the register values        
        //Serial.print("Bit que se escribió: ");
        //read_and_print(((register_value >> (7 - i)) & 0b00000001));
    }

    // send 1 clock pulse, to set the Pins of the ADCs to output and pull high
    clockCycle();

    // At the 46th SCLK, switch DRDY / DOUT to output and pull up DRDY / DOUT. 
    pinMode(DOUT_DRDY, INPUT_PULLUP);
    delay(REGISTER_SETTLING_TIME);
}

void ArduinoUno_ADC_CS1237::setFullRegister(byte register_to_write)
{
    //"Arbitrary" register numbers
    //0 - Channel
    //1 - PGA
    //2 - Speed
    //3 - REF

    //Config register structure
    //bit 0-1 : Channel. 00 - A, 01 - reserved, 10 - Temperature, 11 - internal short (maybe for offset calibration?)
    //bit 2-3 : PGA. 00 - 1, 01 - 2, 10 - 64, 11 - 128
    //bit 4-5 : speed. 00 - 10 Hz, 01 - 40 Hz, 10 - 640 Hz, 11 - 1280 Hz
    //bit 6   : Reference. Default is enabled which is 0.
    //bit 7   : reserved, don't touch
    //----------------------------------------------------------

    //Shift out 27 (24+3) bits
    ADCreading = readADC(); //32-bit variable that stores the whole ADC reading

    pinMode(DOUT_DRDY, OUTPUT); //After the 27th SCLK pulse, set DOUT to OUTPUT

    for (uint8_t i = 0; i < 2; i++) clockCycle(); //Emit 2 pulses (28-29)

    for (uint8_t i = 0; i < 7; i++) //SCLK 30-36, sending READ word
    {
      writeBit(((0x65 >> (6 - i)) & 0b00000001) ? HIGH : LOW); //0x65 - WRITE
    }

    clockCycle(); //Send the 37th SCLK pulse

    Serial.print("Registro que se va a escribir: ");
    read_and_print(register_to_write);

    for (uint8_t i = 0; i < 8; i++) //38-45 SCLK pulses
    {
        writeBit(((register_to_write >> (7 - i)) & 0b00000001) ? HIGH : LOW); //Write the register values        
        Serial.print("Bit que se escribió: ");
        read_and_print(((register_to_write >> (7 - i)) & 0b00000001) ? HIGH : LOW);
    }

    // send 1 clock pulse, to set the Pins of the ADCs to output and pull high
    clockCycle();

    // At the 46th SCLK, switch DRDY / DOUT to output and pull up DRDY / DOUT. 
    pinMode(DOUT_DRDY, INPUT_PULLUP);
}

int ArduinoUno_ADC_CS1237::getRegister() //Reads all registers. The actual values will be fetched with other, simple functions
{  
    int registerValue; //Variable that stores the config register value

    //Shift out 27 (24+3) bits
    ADCreading = readADC(); //32-bit variable that stores the whole ADC reading
    
    pinMode(DOUT_DRDY, OUTPUT); //After the 27th SCLK pulse, set DOUT to OUTPUT

    for (uint8_t i = 0; i < 2; i++) clockCycle(); //Emit 2 pulses (28-29)

    for (uint8_t i = 0; i < 7; i++) //SCLK 30-36, sending READ word
    {
      writeBit(((0x56 >> (6 - i)) & 0b00000001) ? HIGH : LOW); //0x56 - READ
    }

    clockCycle(); //Send the 37th SCLK pulse
    
    //After the 37th SCLK pulse switch the direction of DOUT. 
    pinMode(DOUT_DRDY, INPUT_PULLUP); //we read, so dout becomes INPUT

    registerValue = 0; //Because we are reading

    for (uint8_t i = 0; i < 8; i++) //38-45 SCLK pulses
    {
      registerValue <<= 1;
      registerValue |= readBit();

      //read_and_print(registerValue);
    }

    // send 1 clock pulse, to set the Pins of the ADCs to output and pull high
    clockCycle();

    // At the 46th SCLK, switch DRDY / DOUT to output and pull up DRDY / DOUT. 
    pinMode(DOUT_DRDY, INPUT_PULLUP); //Ready to receive the DRDY to perform a new acquisition

    return registerValue;
}

void ArduinoUno_ADC_CS1237::read_and_printRegister(){
  int regvalues = getRegister(); //Read the register value
  Serial.print("Register ");
  read_and_print(regvalues);
}

void ArduinoUno_ADC_CS1237::read_and_print(int value){
    Serial.print("Value: 0b"); //Print the register value in a convenient 0bxxxxxxxx format

    for (int i = 15; i >= 8; i--) 
    {
      Serial.print((value >> i) & 1); //Print/shift the register bits until the whole byte is printed
    }
    Serial.print(" "); // Print a newline character
    
    for (int i = 7; i >= 0; i--) 
    {
      Serial.print((value >> i) & 1); //Print/shift the register bits until the whole byte is printed
    }
    Serial.println(); // Print a newline character
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
  for (int i = 0; i < 20; ++i) // 455 ns/ 187.5 ns = 2.4. Since 455 is a minimum req, I increased to 10. There's no max value...
  {
    DELAY_455_NS;
  }
  //So actually, this delay is more like 1875 ns. But it seems to work well. It is stable.
  //However, always refer to the clock cycle of your chosen MCU!!!
}
//--------------------------------------------------------------------------------------------------