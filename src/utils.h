#pragma once

#include "config.h"
#include "data_structures.h"

#include <vector>
#include <queue>

using namespace std;



void readImage(const string& filename, const fileFormat& format, vector<double>& image, vector<index_t>& shape);



void removeFromPQ(const Cube& cube, priority_queue<Cube, vector<Cube>, ReverseOrder>& PQ);



void printPQ(priority_queue<Cube, vector<Cube>, ReverseOrder>& PQ);