#pragma once

#include "config.h"
#include "data_structures.h"
#include "nlohmann/json.hpp"

#include <vector>
#include <queue>
#include <map>

using namespace std;
using json = nlohmann::json;



void readImage(const string& filename, const fileFormat& format, vector<double>& image, vector<index_t>& shape);



void removeFromPQ(const Cube& cube, priority_queue<Cube, vector<Cube>, ReverseOrder>& PQ);



void printPQ(priority_queue<Cube, vector<Cube>, ReverseOrder>& PQ);



class DataFrame {
public:
    void addRow(const string& name, const vector<size_t>& row);
    void saveToJson(const string& filename) const;

private:
    map<string, vector<size_t>> data;
};