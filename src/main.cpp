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
	string matchedFilename = "matched.csv";
	string unmatched0Filename = "unmatched_0.csv";
	string unmatched1Filename = "unmatched_1.csv";
	fileFormat format;
    bool print = false;
    bool plot = false;
	bool saveResult = false;

    for (int i = 1; i < argc; ++i) {
		const string arg(argv[i]);
		if (arg == "--help" || arg == "-h") { print_usage_and_exit(0); }
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
    if (plot) { 
        cout << "Image:" << endl;
        mc.plotImage(); cout << endl;
    }

    mc.perturbImage();
    if (plot) { 
        cout << "Perturbed image:" << endl;
        mc.plotImage(); cout << endl;
    }

    mc.processLowerStars();
    mc.checkV();
    if (print) { 
        mc.printC(); cout << endl;
    }

    value_t epsilon = 0.9999;

    //mc.plotV(1);

    mc.extractMorseSkeleton(epsilon);
    if (plot) { 
        cout << "Morse Skeleton:" << endl;
        mc.plotMorseSkeleton(); cout << endl;
        mc.plotMorseSkeletonPixels(); cout << endl;
    }
    
    //mc.extractMorseComplex();
    //if (print) { mc.printFaces(); cout << endl; }

    // Cube s = mc.C[3][0];
    // vector<tuple<Cube,Cube,Cube>> flow;
    // mc.traverseFlow(s, flow);
    // if (print) { cout << "critical cube s: "; s.print(); cout << endl; }
    // if (plot) { 
    //     cout << "flow: " << endl; 
    //     mc.plotFlow(s); cout << endl;
    // }

    // vector<pair<Cube, uint8_t>> boundary = mc.getMorseBoundary(s);
    // if (print) {
    //     cout  << "boundary: " << endl;
    //     for (const pair<Cube, uint8_t>& b : boundary) {
    //         get<0>(b).print(); cout << " " << unsigned(get<1>(b)) << endl;
    //     }
    //     cout << endl;
    // }

    // Cube t;
    // for (const pair<Cube, uint8_t>& b : boundary) { if (get<1>(b) == 1) { t = get<0>(b); break; } }
    // flow.clear();
    // mc.traverseCoFlow(t, flow);
    // if (print) { cout << "critical cube t: "; t.print(); cout << endl; }
    // if (plot) { 
    //     cout << "coflow: " << endl; 
    //     mc.plotCoFlow(t); cout << endl;
    // } else { cout << endl; }
    

    // vector<tuple<Cube,Cube,Cube>> connections;
    // mc.getConnections(s, t, connections);
    // if (print) {
    //     cout << "connections from s to t:" << endl;
    //     for (const tuple<Cube, Cube, Cube>& c : connections) {
    //         get<0>(c).print(); get<1>(c).print(); get<2>(c).print(); cout << endl;
    //     }
    // }
    // if (plot) {
    //     cout << "connections from s to t: " << endl; 
    //     mc.plotConnections(s, t); cout << endl;
    // }

    // mc.cancelPair(s, t);
    // mc.checkV();
    // if (print) { mc.printC(); }

    value_t delta = 20;
    mc.cancelPairs(delta);
    mc.checkV();
    if (print) { mc.printC(); cout << endl; }
}