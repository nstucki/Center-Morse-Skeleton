#include "utils.h"
#include "data_structures.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;



void print_usage_and_exit(int exit_code) {
    cout << endl;
    cerr << "Usage: "
         << "CancelPairs: "
         << "[options] [input_directory]" << endl
         << endl
         << "Options:" << endl
         << endl
         << "  --help, -h                      print this screen" << endl
         << "  --epsilon, -e                   minimum distance of pixel values for perturbation" << endl
         << "  --patch_x, -px                  patch size of x-axis for processing lower stars" << endl
         << "  --patch_y, -py                  patch size of y-axis for processing lower stars" << endl
         << "  --patch_z, -pz                  patch size of z-axis for processing lower stars" << endl
         << "  --threshold, -t                 cancel pairs up to threshold" << endl
         << "  --order_dim, -od                order of dimensions for canceling below threshold" << endl
         << "  --order_value, -ov              order of values for canceling below threshold" << endl
         << "  --print, -p                     print result to console" << endl
         << "  --save, -s                      save result to file" << endl            
         << endl;
	exit(exit_code);
}



int main(int argc, char** argv) {
    string directory = "";
    string saveName = "result.json";
	fileFormat format;
    value_t epsilon = 0;
    index_t patchX = 10;
    index_t patchY = 10;
    index_t patchZ = 10;
    value_t threshold = INFTY;
    string orderDim = ">";
    string orderValue = ">";
    bool perturb = false;
    bool print = false;
	bool save = false;
    DataFrame df; 

    for (int i = 1; i < argc; ++i) {
		const string arg(argv[i]);
		if (arg == "--help" || arg == "-h") { print_usage_and_exit(0); }
        else if (arg == "--epsilon" || arg == "-e") { 
            perturb = true;
            epsilon = stod(argv[++i]);
        }
        else if (arg == "--patch_x" || arg == "-px") { patchX = stod(argv[++i]); }
        else if (arg == "--patch_y" || arg == "-py") { patchY = stod(argv[++i]); }
        else if (arg == "--patch_z" || arg == "-pz") { patchZ = stod(argv[++i]); }
        else if (arg == "--threshold" || arg == "-t") { threshold = stod(argv[++i]); }
        else if (arg == "--order_dim" || arg == "-od") { orderDim = argv[++i]; }
        else if (arg == "--order_value" || arg == "-ov") { orderValue = argv[++i]; }
        else if (arg == "--print" || arg == "-p") { print = true; }
        else if (arg == "--save" || arg == "-s") {
            save = true;
            saveName = argv[++i];
        } 
        else { directory = argv[i]; } 
	}

    if (print) {
        if (perturb) { cout << "Perturbation: " << epsilon << endl; }
        cout << "Threshold: " << threshold << endl;
        cout << "Canceling order:" << endl;
        cout << "dimension: " << orderDim << endl;
        cout << "value: " << orderValue << endl;
        if (save) { cout << "Saving results to " << saveName << "-[options].json" << endl; }
    }

    for (const auto& entry : fs::directory_iterator(directory)) {

        string fileName = entry.path().filename().string();
        if (print) { cout << "-----------------------------------------------------------------------------------" << endl; }
        if (print) { cout << "Processing " << entry.path().string() << ":" << endl; }

        if (fileName.empty()) { continue; }
        if (fileName.find(".txt")!= string::npos) { format = PERSEUS; }
        else if (fileName.find(".npy")!= string::npos) { format = NUMPY; } 
        else if (fileName.find(".complex")!= string::npos) { format = DIPHA; }
        else { cerr << "File format is not supported!" << endl; continue; }

        string filePath = entry.path().string();
        ifstream fileStream(filePath);
        if (!fileName.empty() && fileStream.fail()) {
	        cerr << "Couldn't open file!" << endl;
	        exit(-1);
	    }

        vector<value_t> input;
        vector<index_t> shape;
        readImage(filePath, format, input, shape);

        MorseComplex mc(std::move(input), std::move(shape));

        if (perturb) {
            if (print) { cout << "Perturbing image ..." << endl; }
            mc.perturbImage(epsilon);
        }

        if (print) { cout << "Processing Lower Stars ..." << endl; }
        mc.processLowerStars(patchX, patchY, patchZ);

        //if (print) { cout << "Checking gradient vectorfield ... "; }
        //mc.checkV();

        if (print) {
            cout << endl << "Critical cells:" << endl;
            mc.printNumberOfCriticalCells(threshold); cout << endl << endl;
        }

        if (save) {
            if (print) { cout << "Saving result ..." << endl; }
            vector<vector<size_t>> numCriticalCells = mc.getNumberOfCriticalCells(threshold);
            df.addRow(fileName + " - before canceling", numCriticalCells);
        }

        mc.cancelPairsBelow(threshold, orderDim, orderValue, print);
        mc.cancelPairsAbove(threshold, orderDim, orderValue, print);

        //if (print) { cout << endl << "Checking gradient vectorfield ... "; }
        //mc.checkV();

        if (save) {
            if (print) { cout << "Saving result ..." << endl; }
            vector<vector<size_t>> numCriticalCells = mc.getNumberOfCriticalCells(threshold);
            df.addRow(fileName + " - after canceling", numCriticalCells);
        }
    }

    if (save) {
        if (print) {
            cout << "-----------------------------------------------------------------------------------" << endl;
            cout << "Saving results to " << directory + "/" + saveName << endl;
        }
        df.saveToJson(directory + "/" + saveName + "-order_dim_" + orderDim + "-order_value_" + orderValue + ".json", threshold);
    }
}