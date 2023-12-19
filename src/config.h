#pragma once

#include <cstdint>

#define USE_CUBE_MAP

typedef uint32_t index_t;
typedef double value_t;

#define INFTY numeric_limits<value_t>::infinity()

enum fileFormat { DIPHA, PERSEUS, NUMPY };