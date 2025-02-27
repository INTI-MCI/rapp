#include <OneWire.h>
#include <DallasTemperature.h>
#include <stdio.h>
#include <time.h>

#define ITERACIONES 100


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

    double tiempos_request[ITERACIONES], tiempos_requestAdd[ITERACIONES], tiempos_getTempC[ITERACIONES], tiempos_getTemp[ITERACIONES];

    // Cronometrar funcion1
    for (int i = 0; i < ITERACIONES; i++) {
        unsigned long inicio_req = millis();
        sensorDS18B20.requestTemperatures();
        unsigned long fin_req = millis();
        tiempos_request[i] = (double)(fin_req - inicio_req); // en milisegundos

        unsigned long inicio_reqAdd = millis();
        sensorDS18B20.requestTemperaturesByAddress(sensorVaina);
        unsigned long fin_reqAdd = millis();
        tiempos_requestAdd[i] = (double)(fin_reqAdd - inicio_reqAdd); // en milisegundos

        unsigned long inicio_getC = millis();
        sensorDS18B20.getTempC(sensorVaina);
        unsigned long fin_getC = millis();
        tiempos_getTempC[i] = (double)(fin_getC - inicio_getC);

        unsigned long inicio_get = millis();
        sensorDS18B20.getTemp(sensorVaina);
        unsigned long fin_get = millis();
        tiempos_getTemp[i] = (double)(fin_get - inicio_get);
    }

    // Calcular promedio y máximo para request y getTemp
    double suma_request = 0;
    double max_request = tiempos_request[0];
    double suma_requestAdd = 0;
    double max_requestAdd = tiempos_requestAdd[0];
    double suma_getTempC = 0;
    double max_getTempC = tiempos_getTempC[0];
    double suma_getTemp = 0;
    double max_getTemp = tiempos_getTemp[0];

    for (int i = 0; i < ITERACIONES; i++) {
        suma_request += tiempos_request[i];
        suma_requestAdd += tiempos_requestAdd[i];
        suma_getTempC += tiempos_getTempC[i];
        suma_getTemp += tiempos_getTemp[i];
        /*
        if (tiempos_request[i] > max_request) {
            max_request = tiempos_request[i];
        }
        if (tiempos_requestAdd[i] > max_requestAdd) {
            max_requestAdd = tiempos_requestAdd[i];
        }
        */
        if (tiempos_getTempC[i] > max_getTempC) {
            max_getTempC = tiempos_getTempC[i];
        }
        if (tiempos_getTemp[i] > max_getTemp) {
            max_getTemp = tiempos_getTemp[i];
        }

    }
    /*
    double promedio_request = suma_request / ITERACIONES;
    double promedio_requestAdd = suma_requestAdd / ITERACIONES;
    */
    double promedio_getTempC = suma_getTempC / ITERACIONES;
    double promedio_getTemp = suma_getTemp / ITERACIONES;

    /*
    Serial.print("Promedio request: ");
    Serial.print(promedio_request);
    Serial.println(" milisegundos");
    Serial.print("Máximo request: ");
    Serial.print(max_request);
    Serial.println(" milisegundos");

    Serial.print("Promedio requestAdd: ");
    Serial.print(promedio_requestAdd);
    Serial.println(" milisegundos");
    Serial.print("Máximo requestAdd: ");
    Serial.print(max_requestAdd);
    Serial.println(" milisegundos");
    */

    Serial.print("Promedio getTempC: ");
    Serial.print(promedio_getTempC);
    Serial.println(" milisegundos");
    Serial.print("Máximo getTempC: ");
    Serial.print(max_getTempC);
    Serial.println(" milisegundos");

    Serial.print("Promedio getTemp: ");
    Serial.print(promedio_getTemp);
    Serial.println(" milisegundos");
    Serial.print("Máximo getTemp: ");
    Serial.print(max_getTemp);
    Serial.println(" milisegundos");


}


void loop() {

    /*
    // Mandamos comandos para toma de temperatura a los sensores
    Serial.println("Mandando comandos a los sensores");
    for(int i; i=0; i++){

      sensorDS18B20.requestTemperatures();

    }
    sensorDS18B20.requestTemperatures();

    // Leemos y mostramos los datos de los sensores DS18B20
    Serial.print("Temperatura sensor 0: ");
    Serial.print(sensorDS18B20.getTempC(sensorVaina));
    Serial.println(" C");

    delay(1000);
    */
}