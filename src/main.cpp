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
         << "  --print, -p                     print result to console" << endl              
         << endl;
	exit(exit_code);
}



int main(int argc, char** argv) {
#ifdef RUNTIME
    cout << endl << "reading config & images ... ";
    auto start = high_resolution_clock::now();
#endif

    string filename = "";
	string matchedFilename = "matched.csv";
	string unmatched0Filename = "unmatched_0.csv";
	string unmatched1Filename = "unmatched_1.csv";
	fileFormat format;
    bool print = false;
	bool saveResult = false;

    for (int i = 1; i < argc; ++i) {
		const string arg(argv[i]);
		if (arg == "--help" || arg == "-h") { print_usage_and_exit(0); }
        else if (arg == "--print" || arg == "-p") { print = true; } 
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
    if (print) { mc.printImage(); cout << endl; }

    mc.perturbImage();
    if (print) { mc.printImage(); cout << endl; }

    mc.processLowerStars();
    mc.checkGradientVectorfield();
    if (print) {
        cout << "critical cubes: " << endl;
        for (uint8_t dim = 0; dim < 4; ++dim) {
            for (const Cube& c : mc.C[dim]) {
                c.print(); cout << endl;
            }
        }
        cout << endl;
        mc.printGradientVectorfieldImage(); cout << endl;
    }
    
    mc.extractMorseComplex();
    if (print) { mc.printFaces(); cout << endl; }

    Cube s = mc.C[2][1];
    vector<tuple<Cube,Cube,Cube>> flow;
    mc.traverseFlow(s, flow);
    if (print) {
        mc.printGradientVectorfieldDim(s.dim); cout << endl; 
        cout << "critical cube s: "; s.print(); cout << endl;
        cout << "flow: " << endl; 
        mc.printFlow(s); cout << endl;
    }

    vector<pair<Cube, uint8_t>> boundary = mc.getMorseBoundary(s);
    if (print) {
        cout  << "boundary: " << endl;
        for (const pair<Cube, uint8_t>& b : boundary) {
            get<0>(b).print(); cout << " " << unsigned(get<1>(b)) << endl;
        }
        cout << endl;
    }

    Cube t = mc.C[1][0];
    flow.clear();
    mc.traverseCoFlow(t, flow);

    if (print) {
        cout << "critical cube t: "; t.print(); cout << endl;
        cout << "coflow: " << endl; mc.printCoFlow(t); cout << endl;
    }
    

    vector<tuple<Cube,Cube,Cube>> connections;
    mc.getConnections(s, t, connections);
    if (print) { 
        cout << "connections from s to t: " << endl; 
        mc.printConnections(s, t); cout << endl;
    }
}