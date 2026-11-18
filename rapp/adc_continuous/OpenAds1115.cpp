#include "OpenAds1115.h"

uint16_t OpenADS1115::open_readConfigRegister() {
  buffer[0] = ADS1X15_REG_POINTER_CONFIG;
  buffer[1] = 0; // Placeholder for the first byte of the config register
  buffer[2] = 0; // Placeholder for the second byte of the config register
  m_i2c_dev->write(buffer, 1);
  m_i2c_dev->read(buffer, 2);
  return ((buffer[0] << 8) | buffer[1]);
}