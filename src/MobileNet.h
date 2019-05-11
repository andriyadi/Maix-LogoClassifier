/**
 * @file MobileNet.cpp
 * @brief Detect object type
 * Adapted from MBNet_1000.cpp by Neucrack@sipeed.com
 */

#ifndef __MOBILENET_H
#define __MOBILENET_H

#include "Sipeed_OV2640.h"
#include "Sipeed_ST7789.h"
#include <SD.h>
#include <Maix_KPU.h>

#define KMODEL_SIZE         (4220 * 1024) //not used for now
#define USING_STATISTICS    0   //to use stats algo, for better result filtering
#define STATISTICS_NUM      3
#define LABEL_TEXT_SIZE     2   //2x font size

typedef struct
{
    const char *name;
    float sum;
    bool updated;
    float prob;
    uint16_t origIndex;
} statistics_t;

class MobileNet {
public:
    MobileNet(KPUClass &kpu, Sipeed_ST7789 &lcd, Sipeed_OV2640 &camera);
    ~MobileNet();
    int begin();
    int beginWithModelName(const char *kmodel_name, float threshold = 0.5f);
    int beginWithModelData(uint8_t *model_data, float threshold = 0.5f);

    // Set to 0 or 2 for landscape, 1 or 3 for portrait on LCD. Beware, portrait means wrong camera rotation. 
    // It seems not possible to rotate the camera image programmatically. 
    void setScreenRotation(uint8_t rotation) {
        _rotation = rotation;
    }
    int detect();
    void show();

    const char **labels;
    int16_t lastPredictionLabelIndex = -1;

private:
    float _threshold = 0.5f;
    uint8_t _rotation = 3;

    KPUClass &_kpu;
    Sipeed_ST7789 &_lcd;
    Sipeed_OV2640 &_camera;
    uint8_t *_modelData;
    size_t _labelCount;
    statistics_t _statistics[STATISTICS_NUM];
    float *_kpuResult;
    uint16_t _labelIndices[1000];

    int _modelLoaded = -1;

    size_t printCenterOnLCD(Sipeed_ST7789 &lcd_, const char *msg, uint8_t textSize = LABEL_TEXT_SIZE);

    void label_indices_init();
    void label_get(uint16_t index, float *prob, const char **name);
    void label_sort();

#if USING_STATISTICS
    void label_stats(uint8_t *firstIdxOut, uint8_t *secondIdxOut);
#endif

};

#endif //__MOBILENET_H
