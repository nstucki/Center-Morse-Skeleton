#include "utils.h"
#include "data_structures.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>

using namespace std;
using namespace std::chrono;



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
         << "  --mindist, -m                   minimum distance of pixel values" << endl
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
	fileFormat format;
    value_t threshold = 1;
    value_t minDistance = INFTY;
    bool print = false;
    bool plot = false;
	bool saveResult = false;

    for (int i = 1; i < argc; ++i) {
		const string arg(argv[i]);
		if (arg == "--help" || arg == "-h") { print_usage_and_exit(0); }
        else if (arg == "--threshold" || arg == "-t") { threshold = stod(argv[++i]); }
        else if (arg == "--mindist" || arg == "-m") { minDistance = stod(argv[++i]); }
        else if (arg == "--print" || arg == "-p") { print = true; }
        else if (arg == "--plot" || arg == "-pl") { plot = true; } 
        else { filename = argv[i]; } 
	}

    if (filename.empty()) { print_usage_and_exit(-1); } 
    if (filename.find(".txt")!= string::npos) { format = PERSEUS; } 
    else if (filename.find(".npy")!= string::npos) { format = NUMPY; } 
    else if (filename.find(".complex")!= string::npos) { format = DIPHA; } 
    else {
		cerr << "unknown input file format! (the filename extension should be .txt/.npy/.complex): " << filename << endl;
		exit(-1);
	}
    ifstream fileStream(filename);
	if (!filename.empty() && fileStream.fail()) {
		cerr << "couldn't open file " << filename << endl;
		exit(-1);
	}

    vector<value_t> input;
    vector<index_t> shape;
    readImage(filename, format, input, shape);

#ifdef RUNTIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "of shape (" << shape[0];
    for (uint8_t i = 1; i < shape.size(); i++) { cout << "," << shape[i]; }
    cout << ") ... " << duration.count() << " ms" << endl << endl;
#endif

    MorseComplex mc(std::move(input), std::move(shape));

    if (plot) { cout << "Image:" << endl; mc.plotImage(); cout << endl; }

    mc.perturbImage(minDistance);

    if (plot) { cout << "Perturbed image:" << endl; mc.plotImage(); cout << endl; }

    mc.processLowerStars();
    mc.checkV();

    if (print) { mc.printC(threshold); cout << endl; }

    if (print) { cout << "canceling < " << threshold << " ..." << endl; }
    mc.cancelPairsBelow(threshold, print);

    if (print) { cout << "canceling >= " << threshold << " ..." << endl; }
    mc.cancelPairsAbove(threshold, print);

    mc.checkV();

    if (print) {
        cout << endl << "Morse boundary:" << endl;
        vector<vector<Cube>> C = mc.getCriticalCells();
        for (int dim = 0; dim < 4; ++dim) {
            for (const Cube& c : C[dim]) {
                cout << "c: "; c.print(); cout << endl;
                mc.printMorseBoundary(c);
                cout << endl;
            }
        }
    }
}