#include <OneWire.h>
#include <DallasTemperature.h>
#include <stdio.h>

// Pin donde se conecta el bus 1-Wire
const int pinDatosDQ = 4;

// Instancia a las clases OneWire y DallasTemperature
OneWire oneWireObjeto(pinDatosDQ);
DallasTemperature sensorDS18B20(&oneWireObjeto);

DeviceAddress sensorVaina = {0x28, 0xCF, 0x42, 0x76, 0xE0, 0x01, 0x3C, 0x70};

uint8_t resolucionGlobal =  sensorDS18B20.getResolution(sensorVaina);

void setup() {
    // Iniciamos la comunicación serie
    Serial.begin(57600);
    // Iniciamos el bus 1-Wire
    sensorDS18B20.begin();
    sensorDS18B20.setResolution(12);
    Serial.println(" ");
    Serial.print("Resolución sensor: ");
    Serial.print(resolucionGlobal);
    Serial.println(" bits");

    sensorDS18B20.requestTemperatures();
}


void loop() {
    Serial.print("Temperatura sensor 0: ");
    Serial.print(sensorDS18B20.getTempC(sensorVaina));
    Serial.println(" C");
}