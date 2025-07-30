#ifndef READ_AI_ACTION_JSON_H
#define READ_AI_ACTION_JSON_H

#include <json/json.h>
#include <fstream>
#include <iostream>
#include <string>
#include <map>

struct SliceConfig {
    int bandwidth;
    int priority;
    std::string qos;
};

class AIActionReader {
public:
    bool ReadJson(const std::string& filename);
    std::map<std::string, SliceConfig> GetSlices() const;

private:
    std::map<std::string, SliceConfig> slices;
};

#endif

