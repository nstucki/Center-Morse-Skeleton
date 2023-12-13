#pragma once

#include "config.h"

#include <vector>
#include <set>
#include <unordered_map>
#include <mutex>

using namespace std;



class MorseComplex;



class Cube {
public:
	Cube();
	Cube(const value_t& birth, const index_t& x, const index_t& y, const index_t& z, 
			const uint8_t& type, const uint8_t& dim);
	Cube(const MorseComplex& mc, const index_t& x, const index_t& y, const index_t& z, 
			const uint8_t& type, const uint8_t& dim);
	Cube(const Cube& other);
	Cube& operator=(const Cube& rhs);
	bool operator==(const Cube& rhs) const;
	bool operator!=(const Cube& rhs) const;
	bool operator<(const Cube& rhs) const;
	vector<vector<index_t>> getVertices() const;
	bool isFaceOf(const Cube& other) const;
	void print() const;
	value_t birth;
	index_t x;
	index_t y;
	index_t z;
	uint8_t type;
	uint8_t dim;
	struct Hash {
		size_t operator()(const Cube& cube) const {
			size_t h1 = hash<value_t>{}(cube.birth);
			size_t h2 = hash<index_t>{}(cube.x);
			size_t h3 = hash<index_t>{}(cube.y);
			size_t h4 = hash<index_t>{}(cube.z);
			size_t h5 = hash<uint8_t>{}(cube.type);
			size_t h6 = hash<uint8_t>{}(cube.dim);

			const size_t seed = 0;
			const size_t prime1 = 17;
			const size_t prime2 = 31;
			const size_t prime3 = 37;
			const size_t prime4 = 43;
			const size_t prime5 = 53;
			const size_t prime6 = 61;
			
			size_t hash = seed;
			hash ^= (h1 + prime1) + (h2 + prime2) + (h3 + prime3) + (h4 + prime4) + (h5 + prime5) + (h6 + prime6) 
					+ seed + (seed << 6) + (seed >> 2);
			return hash;
		}
	};
};



struct ReverseOrder {
    bool operator()(const Cube& lhs, const Cube& rhs) const {
        return rhs < lhs;
    }
};



class MorseComplex {
public:
	MorseComplex(const vector<value_t>&& image, const vector<index_t>&& shape);
	MorseComplex(const vector<value_t>& image, const vector<index_t>& shape);
	MorseComplex(MorseComplex&& other);
	~MorseComplex();
	value_t getValue(const index_t& x, const index_t& y, const index_t& z) const;
	value_t getBirth(const index_t& x, const index_t& y, const index_t& z, 
						const uint8_t& type, const uint8_t& dim) const;
	
	void perturbImage(const value_t& epsilon=INFTY);
	void perturbImageMinimal();
	void processLowerStars();
	void processLowerStarsParallel(const index_t& xPartition = 1, const index_t& yPartition = 1, const index_t& zPartition = 1);
	void cancelPairs(const value_t& threshold, string orderDimBelow, string orderValueBelow,
						string orderDimAbove, string orderValueAbove, bool print);
	void prepareMorseSkeletonBelow(const value_t& threshold, bool print);
	void prepareMorseSkeletonAbove(const value_t& threshold, bool print);
	void extractMorseSkeletonBelow(const value_t& threshold);
	void extractMorseSkeletonAbove(const value_t& threshold);
	vector<vector<index_t>> getMorseSkeletonVoxelsBelow() const;
	vector<vector<index_t>> getMorseSkeletonVoxelsAbove() const;
	value_t getPerturbation() const;
	vector<vector<Cube>> getCriticalCells() const;
	vector<vector<vector<vector<index_t>>>> getCriticalVoxels() const;
	vector<vector<size_t>> getNumberOfCriticalCells(const value_t& threshold=INFTY) const;
	void checkV() const;
	void checkBoundaryAndCoboundary() const;
	void printC(const value_t& threshold=INFTY) const;
	void printV() const;
	void printMorseBoundary(const Cube& c) const;
	void printFaces();
	void plotV() const;
	void plotV(uint8_t dim) const;
	void plotFlow(const Cube& s) const;
	void plotCoFlow(const Cube& s) const;
	void plotConnections(const Cube& s, const Cube& t) const;
	void plotMorseSkeleton() const;
	void plotMorseSkeletonPixels() const;
	void plotImage() const;
	const vector<index_t> shape;

private:
	value_t*** allocateMemory() const;
	void getGridFromVector(const vector<value_t>& vector);
	void setValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value);
	void addValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value);
	Cube getCube(const index_t& x, const index_t& y, const index_t& z) const;
	vector<Cube> getFaces(const Cube& cube);
	value_t getMinimumDistance();
	vector<Cube> getLowerStar(const index_t& x, const index_t& y, const index_t& z) const;
	size_t numUnpairedFaces(const Cube& cube, const vector<Cube>& L) const;
	size_t numUnpairedFacesParallel(const Cube& cube, const vector<Cube>& L) const;
	Cube getUnpairedFace(const Cube& cube, const vector<Cube>& L) const;
	Cube getUnpairedFaceParallel(const Cube& cube, const vector<Cube>& L) const;
	void insertToC(const Cube& cube);
	void insertToV(const Cube& cube0, const Cube& cube1);
	void processLowerStarsBetween(const index_t& x_min, const index_t& x_max, const index_t& y_min, const index_t& y_max,
									const index_t& z_min, const index_t& z_max);
	vector<pair<Cube, uint8_t>> getMorseBoundary(const Cube& s) const;
	vector<pair<Cube, uint8_t>> getMorseCoboundary(const Cube& s) const;
	void traverseFlow(const Cube& s, vector<tuple<Cube, Cube, Cube>>& flow, bool coordinated=true) const;
	void traverseCoflow(const Cube& s, vector<tuple<Cube, Cube, Cube>>& flow, bool coordinated=true) const;
	void getConnections(const Cube&s, const Cube& t, vector<tuple<Cube, Cube, Cube>>& connections) const;
	void cancelPair(const Cube&s, const Cube& t);
	void cancelPairsDimDecreasingValueDecreasingBelow(const value_t& threshold=INFTY, bool print=true);
	void cancelPairsDimDecreasingValueIncreasingBelow(const value_t& threshold=INFTY, bool print=true);
	void cancelPairsDimIncreasingValueDecreasingBelow(const value_t& threshold=INFTY, bool print=true);
	void cancelPairsDimIncreasingValueIncreasingBelow(const value_t& threshold=INFTY, bool print=true);
	void cancelPairsDimDecreasingValueDecreasingAbove(const value_t& threshold=INFTY, bool print=true);
	void cancelPairsDimDecreasingValueIncreasingAbove(const value_t& threshold=INFTY, bool print=true);
	void cancelPairsDimIncreasingValueDecreasingAbove(const value_t& threshold=INFTY, bool print=true);
	void cancelPairsDimIncreasingValueIncreasingAbove(const value_t& threshold=INFTY, bool print=true);
	void cancelClosePairsBelow(const value_t& threshold, bool print=true);
	void extractMorseComplex();
	void processLowerStarsBackup();
	value_t*** grid;
	bool perturbed;
	bool processedLowerStars;
	value_t perturbation;
	vector<vector<Cube>> C;
	mutable std::mutex mutexC;
	unordered_map<Cube, Cube, Cube::Hash> V;
	unordered_map<Cube, Cube, Cube::Hash> coV;
	mutable std::mutex mutexV;
	unordered_map<Cube, vector<Cube>, Cube::Hash> faces;
	set<Cube> morseSkeletonBelow;
	set<Cube> morseSkeletonAbove;
	set<vector<index_t>> morseSkeletonVoxelsBelow;
	set<vector<index_t>> morseSkeletonVoxelsAbove;
};