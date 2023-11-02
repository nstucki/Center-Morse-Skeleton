#pragma once

#include "config.h"

#include <vector>
#include <unordered_map>

using namespace std;

void readImage(const string& filename, const fileFormat& format, vector<double>& image, vector<index_t>& shape);