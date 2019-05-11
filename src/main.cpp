#include <Arduino.h>

#define PASSTHROUGH 0 //Set to 1 just to test camera stream displayed on LCD, and check the UI

#ifdef __cplusplus
extern "C"
{
#endif
#if !PASSTHROUGH
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"
#endif
#ifdef __cplusplus
}
#endif

#if !PASSTHROUGH
//Include model in memory
INCBIN(model, "logoclassifier.kmodel");
#endif

// Should be the same as image size during training
#define IMAGE_WIDTH               224
#define IMAGE_HEIGHT              224
#define VALID_CLASSIFY_THRESHOLD  0.72f //Change this depends on how confidence you are :)

#include <Sipeed_OV2640.h>
#include <Sipeed_ST7789.h>
#include "MobileNet.h"
#include "Maix_KPU.h"

SPIClass spi_(SPI0); // MUST be SPI0 for Maix series on board LCD
Sipeed_ST7789 lcd(320, 240, spi_);

Sipeed_OV2640 camera(IMAGE_WIDTH, IMAGE_HEIGHT, PIXFORMAT_RGB565);
KPUClass KPU;
MobileNet mbnet(KPU, lcd, camera);

//Specific for this demo
uint8_t detectedCount = 0;
const char *MSG_HAPPY = "Happy 12th Birthday!";

void setup()
{
    Serial.begin(115200);
    while (!Serial) {
        ; // wait for serial port to connect. Needed for native USB port only
    }
    
    mbnet.setScreenRotation(3);
    
#if PASSTHROUGH
    int ret = mbnet.begin();
#else
    int ret = mbnet.beginWithModelData(model_data, VALID_CLASSIFY_THRESHOLD);
#endif
    if(ret != 0)
    {
        printf("Mobile net init is failed with err: %d\n", ret);
        while(1);
    }
}

void loop()
{
    if(mbnet.detect() != 0)
    {
#if !PASSTHROUGH
      Serial.println("Object classification is failed");
      return;
#endif
    }
    mbnet.show();


    // Just for the demo on my video
#if !PASSTHROUGH
    //Specific for this demo
    if (mbnet.lastPredictionLabelIndex == 0) {
      detectedCount++;
    }
    else if (mbnet.lastPredictionLabelIndex == -1){
      detectedCount = 0;
    }
    else {
      detectedCount = 0;
    }
    // printf("cnt: %d\n", detectedCount);

    if (detectedCount > 10) {
      if (lcd.getRotation() == 3 || lcd.getRotation() == 1) {
          uint16_t textX = (uint16_t)((lcd.width() - (strlen(MSG_HAPPY)*1.0f*LABEL_TEXT_SIZE*6))/2);
          lcd.setCursor(textX, lcd.height() - 1*(8*LABEL_TEXT_SIZE + 8));
          lcd.print(MSG_HAPPY);
      }
      else {
          lcd.setCursor(camera.width(), 4);
          lcd.print("HAPPY");
          lcd.setCursor(camera.width(), 4 + 1*(8*LABEL_TEXT_SIZE + 8));
          lcd.print("12TH");
          lcd.setCursor(camera.width(), 4 + 2*(8*LABEL_TEXT_SIZE + 8));
          lcd.print("BIRTHDAY");
      }
    }
#endif
}
