#include "utils.h"
#include "data_structures.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>
#include <filesystem>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;



void print_usage_and_exit(int exit_code) {
    cout << endl;
    cerr << "Usage: "
         << "Betti Matching "
         << "[options] [input_filename]" << endl
         << endl
         << "Options:" << endl
         << endl
         << "  --help, -h                      print this screen" << endl
         << "  --threshold, -t                 cancel pairs up to threshold" << endl
         << "  --epsilon, -e                   minimum distance of pixel values for perturbation" << endl
         << "  --print, -p                     print result to console" << endl 
         << "  --plot, -pl                     plot result to console" << endl              
         << endl;
	exit(exit_code);
}



int main(int argc, char** argv) {
#ifdef RUNTIME
    cout << endl << "reading config & images ... ";
    auto start = high_resolution_clock::now();
#endif

    string filename = "";
    string directory = "";
	fileFormat format;
    value_t threshold = INFTY;
    value_t epsilon = INFTY;
    bool print = false;
    bool plot = false;
	bool saveResult = false;

    for (int i = 1; i < argc; ++i) {
		const string arg(argv[i]);
		if (arg == "--help" || arg == "-h") { print_usage_and_exit(0); }
        else if (arg == "--threshold" || arg == "-t") { threshold = stod(argv[++i]); }
        else if (arg == "--epsilon" || arg == "-e") { epsilon = stod(argv[++i]); }
        else if (arg == "--print" || arg == "-p") { print = true; }
        else if (arg == "--plot" || arg == "-pl") { plot = true; } 
        else { filename = argv[i]; } 
	}

    // if (filename.empty()) { print_usage_and_exit(-1); } 
    // if (filename.find(".txt")!= string::npos) { format = PERSEUS; } 
    // else if (filename.find(".npy")!= string::npos) { format = NUMPY; } 
    // else if (filename.find(".complex")!= string::npos) { format = DIPHA; } 
    // else {
	// 	cerr << "unknown input file format! (the filename extension should be .txt/.npy/.complex): " << filename << endl;
	// 	exit(-1);
	// }
    // ifstream fileStream(filename);
	// if (!filename.empty() && fileStream.fail()) {
	// 	cerr << "couldn't open file " << filename << endl;
	// 	exit(-1);
	// }

    // vector<value_t> input;
    // vector<index_t> shape;
    // readImage(filename, format, input, shape);

    // MorseComplex mc(std::move(input), std::move(shape));

    // mc.perturbImage();

    // auto start = high_resolution_clock::now();

    // //mc.processLowerStars();
    // //mc.processLowerStars2();
    // mc.processLowerStarsParallel(8, 9, 10);

    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<milliseconds>(stop - start);
    // cout << duration.count() << " ms" << endl;

    // mc.checkV();

    // // cout << mc.getPerturbation() << endl;

    // mc.printC(threshold); cout << endl;

    // mc.cancelPairs(threshold, "<", ">", "<", "<", print);

    // mc.checkV();

    // if (print) {
    //     cout << endl << "Morse boundary:" << endl;
    //     vector<vector<Cube>> C = mc.getCriticalCells();
    //     for (int dim = 0; dim < 4; ++dim) {
    //         for (const Cube& c : C[dim]) {
    //             cout << "c: "; c.print(); cout << endl;
    //             mc.printMorseBoundary(c);
    //             cout << endl;
    //         }
    //     }
    // }

    directory = filename;
    for (const auto& entry : fs::directory_iterator(directory)) {
        string fileName = entry.path().filename().string();
        cout << "-----------------------------------------------------------------------" << endl;
        cout << "processing " << fileName << endl;

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

        // MorseComplex mc(std::move(input), std::move(shape));
        MorseComplex mc(input, shape);
        mc.perturbImage(epsilon);
        mc.processLowerStars();
        cout << "Checking gradient vectorfield ... ";
        mc.checkV();

        MorseComplex mc2(input, shape);
        mc2.perturbImage(epsilon);
        mc2.processLowerStarsParallel(8, 9, 10);
        cout << "Checking gradient vectorfield ... ";
        mc2.checkV();

        vector<vector<size_t>> num = mc.getNumberOfCriticalCells(1);
        vector<vector<size_t>> num2 = mc2.getNumberOfCriticalCells(1);

        for (uint8_t k = 0; k < 3; ++k) {
            for (uint8_t i = 0; i < 4; ++i) {
                cout << num[k][i] << " <-> " << num2[k][i] << endl;
            }
        }
        if (num != num2) { cerr << "something went wrong!" << endl;}
        

        // cout << "Checking boundaries and coboundaries ... ";
        // mc.checkBoundaryAndCoboundary();
    }
}