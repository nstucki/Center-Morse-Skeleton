#include "utils.h"
#include "npy.hpp"

#include <cfloat>
#include <cassert>

void readImage(const string& filename, const fileFormat& format, vector<double>& image, vector<index_t>& shape) {
    switch (format) {
        case DIPHA: {
            ifstream fin(filename, ios::in | ios::binary );
            int64_t d;
            fin.read((char *) &d, sizeof(int64_t));
            assert(d == 8067171840);
            fin.read((char *) &d, sizeof(int64_t));
            assert(d == 1);
            fin.read((char *) &d, sizeof(int64_t));
            fin.read((char *) &d, sizeof(int64_t));
            uint8_t dim = d;
            assert(dim < 4);
            uint64_t n;
            fin.read((char *) &d, sizeof(int64_t));
            shape.push_back(d);
            n = d;
            if (dim > 1) {
                fin.read((char *) &d, sizeof(int64_t));
                shape.push_back(d);
                n *= d;
            }
            if (dim > 2) {
                fin.read((char *)&d, sizeof(int64_t));
                shape.push_back(d);
                n *= d;
            }
            double value;
            image.reserve(n);
            while (!fin.eof()){
                fin.read((char *)&value, sizeof(double));
                image.push_back(value);
            }
            fin.close();
            return;
        }

        case PERSEUS: {
            ifstream reading_file; 
            reading_file.open(filename.c_str(), ios::in); 
            string reading_line_buffer; 
            getline(reading_file, reading_line_buffer); 
            uint8_t dim = atoi(reading_line_buffer.c_str());
            uint64_t n;
            getline(reading_file, reading_line_buffer);
            shape.push_back(atoi(reading_line_buffer.c_str()));
            n = shape[0];
            if (dim > 1) {
                getline(reading_file, reading_line_buffer); 
                shape.push_back(atoi(reading_line_buffer.c_str()));
                n *= shape[1];
            }
            if (dim > 2) {
                getline(reading_file, reading_line_buffer);
                shape.push_back(atoi(reading_line_buffer.c_str()));
                n *= shape[2];
            }
            image.reserve(n);
            double value;
            while(!reading_file.eof()) {
                getline(reading_file, reading_line_buffer);
                value = atof(reading_line_buffer.c_str());
                if (value != -1) { image.push_back(value); }
                else { image.push_back(DBL_MAX); }
            }
            reading_file.close();
            return;
		}

        case NUMPY: {
            vector<unsigned long> _shape;
            try { npy::LoadArrayFromNumpy(filename.c_str(), _shape, image); } 
            catch (...) {
                cerr << "The data type of an numpy array should be numpy.float64." << endl;
                exit(-2);
            }
            uint8_t dim = shape.size();
            for (uint32_t i : _shape) { shape.push_back(i); }
            return;
        }
    }
}



void removeFromPQ(const Cube& cube, priority_queue<Cube, vector<Cube>, ReverseOrder>& PQ) {
    vector<Cube> temp;
    while (!PQ.empty()) {
        if (PQ.top() != cube) {
            temp.push_back(PQ.top());
        }
        PQ.pop();
    }

    make_heap(temp.begin(), temp.end());
    PQ = priority_queue<Cube, vector<Cube>, ReverseOrder>(temp.begin(), temp.end());
} 



void printPQ(priority_queue<Cube, vector<Cube>, ReverseOrder>& PQ) {
    vector<Cube> temp;
    while (!PQ.empty()) {
        PQ.top().print(); cout << endl;
        temp.push_back(PQ.top());
        PQ.pop();
    }

    make_heap(temp.begin(), temp.end());
    PQ = priority_queue<Cube, vector<Cube>, ReverseOrder>(temp.begin(), temp.end());
}



void DataFrame::addRow(const string& name, const vector<vector<size_t>>& row) { data.emplace(name, row); }


void DataFrame::saveToJson(const string& filename, const value_t& threshold) const {
    json result;

    for (auto it = data.begin(); it != data.end(); ++it) {
        json column;
        for (size_t i = 0; i < (it->second).size(); ++i) {
            for (size_t j = 0; j < (it->second)[i].size(); ++j) {
                if (i == 0) { column["total in dim " + to_string(j)] = (it->second)[i][j]; }
                else if (i == 1) { 
                    column["< " + to_string(threshold) + " in dim " + to_string(j)] = (it->second)[i][j];
                }
                else {
                    column[">= " + to_string(threshold) + " in dim " + to_string(j)] = (it->second)[i][j];
                }
            }
        }
        result[it->first] = column;
    }

    ofstream outputFile(filename);
    outputFile << result.dump(2);
    outputFile.close();
    }