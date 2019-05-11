/**
 * @file MobileNet.cpp
 * @brief Detect object type
 * Adapted from MBNet_1000.cpp by Neucrack@sipeed.com
 */

#include "MobileNet.h"
#include "names.h"
#include "stdlib.h"
#include "errno.h"

const char *MSG_LOADING = "Loading...";
const char *MSG_NO_MEM = "Memory not enough...";
const char *MSG_NO_SD = "No SD card!";
const char *MSG_UNKNOWN = "Unknown";
const char *MSG_INSTRUCTION = "Show me...";


#define BG_COLOR COLOR_BLACK

MobileNet::MobileNet(KPUClass &kpu, Sipeed_ST7789 &lcd, Sipeed_OV2640 &camera)
    : _kpu(kpu), _lcd(lcd), _camera(camera),
      _modelData(nullptr), _labelCount(0), _kpuResult(nullptr)
{
    labels = mbnet_label_name;
    memset(_statistics, 0, sizeof(_statistics));
}

MobileNet::~MobileNet()
{
    if (_modelData)
        free(_modelData);
}

int MobileNet::begin() 
{
    if (!_camera.begin())
        return -1;
    if (!_lcd.begin(15000000, BG_COLOR))
        return -2;
    _camera.run(true);

    _lcd.setTextSize(LABEL_TEXT_SIZE);
    _lcd.setTextColor(COLOR_WHITE);
    _lcd.setRotation(_rotation);

    printCenterOnLCD(_lcd, MSG_LOADING, LABEL_TEXT_SIZE);
    delay(500); //So, I can see some message, not really needed

    return 0;
}

int MobileNet::beginWithModelData(uint8_t *model_data, float threshold) 
{
    _threshold = threshold;
    int ret = begin();
    if (ret != 0) {
        printf("FAILED to init, err: %d\n", ret);
        return ret;
    }

    _modelLoaded = _kpu.begin(model_data);
    if (_modelLoaded != 0)
    {
        printf("FAILED to load model, err: %d\n", _modelLoaded);
        return -6;
    }

    return 0;
}

int MobileNet::beginWithModelName(const char *kmodel_name, float threshold)
{
    _threshold = threshold;
    int ret = begin();
    if (ret != 0) {
        return ret;
    }

    if (!SD.begin())
    {
        printCenterOnLCD(_lcd, MSG_NO_SD, LABEL_TEXT_SIZE);
        return -3;
    }

    File myFile = SD.open(kmodel_name);
    if (!myFile)
        return -4;
    uint32_t fSize = myFile.size();
    
    _modelData = (uint8_t *)malloc(fSize);
    if (!_modelData)
    {
        printCenterOnLCD(_lcd, MSG_NO_MEM, LABEL_TEXT_SIZE);
        return ENOMEM;
    }
    long retSize = myFile.read(_modelData, fSize);
    myFile.close();
    if (retSize != fSize)
    {
        free(_modelData);
        _modelData = nullptr;
        return -5;
    }

    _modelLoaded = _kpu.begin(_modelData);
    if (_modelLoaded != 0)
    {
        printf("FAILED to load model, err: %d\n", _modelLoaded);
        free(_modelData);
        _modelData = nullptr;
        return -6;
    }
    return 0;
}

size_t MobileNet::printCenterOnLCD(Sipeed_ST7789 &lcd_, const char *msg, uint8_t textSize) 
{
    lcd_.setCursor((lcd_.width() - (6 * textSize * strlen(msg))) / 2, (lcd_.height() - (8*textSize)) / 2);
    return lcd_.print(msg);
}

int MobileNet::detect()
{
    uint8_t *img = _camera.snapshot();
    if (img == nullptr || img == 0) {
        return -1;
    }
    
    if (_modelLoaded != KPU_ERROR_NONE) {
        return _modelLoaded;
    }

    uint8_t *img888 = _camera.getRGB888();
    if (_kpu.forward(img888) != 0)
    {
        return -2;
    }

    while (!_kpu.isForwardOk())
        ;

    if (_kpu.getResult((uint8_t **)&_kpuResult, &_labelCount) != 0)
    {
        return -3;
    }

    return 0;
}

void MobileNet::show()
{
    char predLabel[64];// = "Unknown";
    uint16_t *img;
    
    int16_t validLabelIdx = -1;
    lastPredictionLabelIndex = -1;

#if USING_STATISTICS
    uint8_t firstIdx = 0, secondIdx = 0;
#endif

    if (_modelLoaded == KPU_ERROR_NONE) {
        
        _labelCount /= sizeof(float);

        label_indices_init();
        label_sort();

        // for(int x = 0; x < _labelCount; x++) {
        //     printf("%d -> %.2f, ", _labelIndices[x], _kpuResult[x]);
        // }
        // printf("\n");//=======\n");

        float firstProb;
        const char *firstName;
        
#if USING_STATISTICS
        label_stats(&firstIdx, &secondIdx);

        firstProb = _statistics[firstIdx].prob;
        firstName = _statistics[firstIdx].name;
        // label_get(firstIdx, &firstProb, &firstName);

        printf("%s -> %d, %d | %.2f\n", firstName, firstIdx, secondIdx, firstProb);
#else
        label_get(0, &firstProb, &firstName);
        // printf("%s -> %.2f\n", firstName, firstProb);
#endif

        if (firstProb < _threshold) {
            sprintf(predLabel, "%s", MSG_INSTRUCTION);
        }
        else {

#if USING_STATISTICS
            // label_stats(_statistics, &firstIdx, &secondIdx);
            // label_get(_kpuResult, labels, _labelIndices, firstIdx, &firstProb, &firstName);
            // firstIdx = _labelIndices[0];
            // secondIdx = _labelIndices[1];

            validLabelIdx = _statistics[firstIdx].origIndex;
#else
            validLabelIdx = 0;
#endif
            lastPredictionLabelIndex = _labelIndices[validLabelIdx];
            sprintf(predLabel, "%s (%.2f%s)", firstName, (firstProb*100), "%");
        }
    }
    else {
        sprintf(predLabel, "%s", MSG_INSTRUCTION);
        // sprintf(predLabel, "%s (%.2f%s)", "Unknown", (1.0*100), "%");
    }

    img = _camera.getRGB565();

    uint16_t imgX = 0, imgY = 0;
    if (_rotation == 3 || _rotation == 1) {
        imgX = (_lcd.width() - _camera.width())/2;
        imgY = 0;

        _lcd.drawImage(imgX, imgY, _camera.width(), _camera.height(), img);
        _lcd.fillRect(0, imgY + _camera.height(), _lcd.width(), (_lcd.height() - _camera.height()), BG_COLOR);
    }
    else {
        imgY = (_lcd.height() - _camera.height());
        imgX = 0;

        _lcd.drawImage(imgX, imgY, _camera.width(), _camera.height(), img);
        _lcd.fillRect(0, 0, _camera.width(), imgY, BG_COLOR);
        _lcd.fillRect(_camera.width(), 0, (_lcd.width() - _camera.width()), _lcd.height(), BG_COLOR);
    }

    if (_rotation == 3 || _rotation == 1) {
        uint16_t textX = (uint16_t)((_lcd.width() - (strlen(predLabel)*1.0f*LABEL_TEXT_SIZE*6))/2);
        // printf("x %d, x1 %d\n", textX, (uint16_t)(strlen(predLabel)*1.0f*LABEL_TEXT_SIZE*8));
        _lcd.setCursor(textX, imgY + _camera.height() + 10);
    }
    else {   
        _lcd.setCursor(4, 0);
    }

    //Print first prediction or unknown
    _lcd.print(predLabel);

    //Print second prediction, if any valid prediction
    if (validLabelIdx > -1) {        

        float scndProb;
        const char *scndName;

#if USING_STATISTICS
        scndProb = _statistics[secondIdx].prob;
        scndName = _statistics[secondIdx].name;
        // label_get(secondIdx, &scndProb, &scndName);
#else
        label_get(1, &scndProb, &scndName);
        // float scndProb = _kpuResult[1];
        // const char *scndName = labels[_labelIndices[1]];
#endif

        if (_rotation == 3 || _rotation == 1) {          
            sprintf(predLabel, "%s (%.2f%s)", scndName, (scndProb*100), "%");  
            uint16_t textX = (uint16_t)((_lcd.width() - (strlen(predLabel)*1.0f*LABEL_TEXT_SIZE*6))/2);
            _lcd.setCursor(textX, imgY + _camera.height() + 10 + 8*LABEL_TEXT_SIZE + 8);
        }
        else {
            _lcd.setCursor(4, 1*(8*LABEL_TEXT_SIZE + 8));
            sprintf(predLabel, "%s (%.2f%s)", scndName, (scndProb*100), "%"); 
        }
        _lcd.print(predLabel);
    }
    
#if USING_STATISTICS
    // printf("valid idx: %d\n", validLabelIdx);
    // for(int x = 0; x < STATISTICS_NUM; x++) {
    //     printf("%s %.2f, %.2f; ", _statistics[x].name, _statistics[x].sum, _statistics[x].prob);
    // }
    // printf("\n");//=======\n");
#endif
}

#if USING_STATISTICS
void MobileNet::label_stats(uint8_t *firstIdxOut, uint8_t *secondIdxOut) {

    uint8_t i, j;   
    float prob;
    const char *name;

    // Initialize
    for (j = 0; j < STATISTICS_NUM; ++j) {
        _statistics[j].updated = false;
    }

    for (i = 0; i < STATISTICS_NUM; i++)
    {
        //Query prob, label for an index
        label_get(i, &prob, &name);
        for (j = 0; j < STATISTICS_NUM; ++j)
        {
            if (_statistics[j].name == NULL) //Initial
            {
                _statistics[j].name = name;
                _statistics[j].sum = prob;
                _statistics[j].prob = prob;
                _statistics[j].origIndex = i;
                _statistics[j].updated = true;
                break;
            }
            else if (_statistics[j].name == name) //If occured again, add
            {
                _statistics[j].sum += prob;
                _statistics[j].prob = prob;
                _statistics[j].origIndex = i;
                _statistics[j].updated = true;
                break;
            }
            else
            {
            }
        }
        if (j == STATISTICS_NUM)
        {
            float min = _statistics[0].sum;
            j = 0;
            for (i = 1; i < STATISTICS_NUM; ++i)
            {
                if (_statistics[i].name)
                {
                    if (_statistics[i].sum <= min)
                    {
                        min = _statistics[i].sum;
                        j = i;
                    }
                }
            }
            _statistics[j].name = name;
            _statistics[j].sum = prob;
            _statistics[j].updated = true;
        }
    }
    float firstMax = _statistics[0].sum;
    float secondMax = 0;
    
    for (i = 0; i < STATISTICS_NUM; ++i)
    {
        if (_statistics[i].name)
        {
            if (_statistics[i].sum > firstMax)
            {
                firstMax = _statistics[i].sum;
                *firstIdxOut = i;
            }
            else if (_statistics[i].sum > secondMax && _statistics[i].sum < firstMax)
            {
                *secondIdxOut = i;
            }
        }

        if (!_statistics[i].updated)
        {
            float tmp = _statistics[i].sum - _statistics[i].sum * 2 / STATISTICS_NUM;
            if (tmp < 0)
                tmp = 0;
            _statistics[i].sum = tmp;
        }
    }
}
#endif

void MobileNet::label_indices_init()
{
    int i;
    for (i = 0; i < _labelCount; i++) {
        _labelIndices[i] = i;
    }
}

void MobileNet::label_sort()
{
    int i,j;
    float tmp_prob;
    uint16_t tmp_index;
    for(j = 0; j < _labelCount; j++) {
        for(i = 0; i < _labelCount-1-j; i++) {
            if (_kpuResult[i] < _kpuResult[i+1])
            {
                tmp_prob = _kpuResult[i];
                _kpuResult[i] = _kpuResult[i+1];
                _kpuResult[i+1] = tmp_prob;
                
                tmp_index = _labelIndices[i];
                _labelIndices[i] = _labelIndices[i+1];
                _labelIndices[i+1] = tmp_index;
            }
        }
    }
}

void MobileNet::label_get(uint16_t index, float *prob, const char **label)
{
    *prob = _kpuResult[index];
    *label = labels[_labelIndices[index]];
}
