#pragma once

#include "config.h"

#include <vector>
#include <set>
#include <unordered_map>
#include <mutex>
#include <iostream>

using namespace std;



class MorseComplex;



class Cube {
public:
	Cube();
	Cube(const value_t& birth, const index_t& x, const index_t& y, const index_t& z, 
			const int8_t& type, const int8_t& dim);
	Cube(const MorseComplex& mc, const index_t& x, const index_t& y, const index_t& z, 
			const int8_t& type, const int8_t& dim);
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
	int8_t type;
	int8_t dim;
	struct Hash {
		size_t operator()(const Cube& cube) const {
			size_t h1 = hash<value_t>{}(cube.birth);
			size_t h2 = hash<index_t>{}(cube.x);
			size_t h3 = hash<index_t>{}(cube.y);
			size_t h4 = hash<index_t>{}(cube.z);
			size_t h5 = hash<int8_t>{}(cube.type);
			size_t h6 = hash<int8_t>{}(cube.dim);

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



template<typename T>
class CubeMap {
public:
	CubeMap(const vector<index_t>& _shape)  : shape(_shape), mYZ(shape[1]*shape[2]), nYZ(mYZ*3), nZ(shape[2]*3), map(4) {
		size_t m = shape[0]*shape[1]*shape[2];
		map[0] = vector<T>(m);
		map[1] = vector<T>(m*3);
		map[2] = vector<T>(m*3);
		map[3] = vector<T>(m);
	};

	CubeMap(const CubeMap& other) : shape(other.shape), mYZ(other.mYZ), nYZ(other.nYZ), nZ(other.nZ), map(other.map) {};

	void emplace(const Cube& key, const T& value) {
		map[key.dim][hashCube(key)] = value;
	};

	void increment(const Cube& key) {
		++map[key.dim][hashCube(key)];
	}

	void addValue(const Cube& key, T value) {
		map[key.dim][hashCube(key)] += value;
	}

	T getValue(const Cube& key) const {
		return map[key.dim][hashCube(key)];
	};

	void erase(const Cube& key) {
		map[key.dim][hashCube(key)] = T();
	};
	

private:
	size_t hashCube(const Cube& cube) const {
		switch(cube.dim) {
			case 0:
			return cube.x*mYZ + cube.y*shape[2] + cube.z;

			case 1:
			return cube.x*nYZ + cube.y*nZ + cube.z*3 + cube.type;

			case 2:
			return cube.x*nYZ + cube.y*nZ + cube.z*3 + cube.type;

			case 3:
			return cube.x*mYZ + cube.y*shape[2] + cube.z;
		}
		cerr << "Did not catch cube!" << endl;

		return 0;
	};
	vector<index_t> shape;
	size_t mYZ;
	size_t nYZ;
	size_t nZ;
	vector<vector<T>> map;
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
	void perturbImage(const value_t& epsilon=0);
	void perturbImageMinimal();
	void processLowerStars(const index_t& xPatch=1, const index_t& yPatch=1, const index_t& zPatch=1);
	void cancelPairsBelow(const value_t& threshold=INFTY, string orderDim=">", string orderValue=">", bool print=true);
	void cancelPairsAbove(const value_t& threshold=INFTY, string orderDim=">", string orderValue=">", bool print=true);
	void cancelLowPersistencePairsBelow(const value_t& threshold=INFTY, const value_t& epsilon=0, bool print=true);
	void cancelLowPersistencePairsInDimBelow(const uint8_t& dim, const value_t& threshold=INFTY, const value_t& epsilon=0, bool print=true);
	void cancelClosePairsBelow(const value_t& threshold=INFTY, const value_t& epsilon=0, bool print=true);
	void cancelBoundaryPairsBelow(const value_t& threshold=INFTY, const value_t& delta=0, bool print=true);
	void prepareMorseSkeletonBelow(const value_t& threshold=INFTY, const value_t& epsilon=0, const value_t& delta=-1, bool print=true);
	void prepareMorseSkeletonTestBelow(const value_t& threshold=INFTY, const value_t& epsilon=0, const value_t& delta=-1, bool print=true);
	void prepareMorseSkeletonAbove(const value_t& threshold=INFTY, const value_t& tolerance=0, bool print=true);
	void extractMorseSkeletonBelow(const value_t& threshold=INFTY);
	void extractMorseSkeletonInDimBelow(const uint8_t& dim, const value_t& threshold=INFTY);
	void prepareAndExtractMorseSkeletonBelow(const value_t& threshold=INFTY, const value_t& epsilon=0, const vector<uint8_t>& dimensions={1,2,3});
	void extractMorseSkeletonAbove(const value_t& threshold=INFTY);
	vector<pair<Cube, uint8_t>> getMorseBoundary(const Cube& s) const;
	vector<pair<Cube, uint8_t>> getMorseCoboundary(const Cube& s) const;
	vector<vector<index_t>> getMorseSkeletonVoxelsBelow() const;
	vector<vector<index_t>> getMorseSkeletonVoxelsAbove() const;
	value_t getPerturbation() const;
	vector<vector<Cube>> getCriticalCells() const;
	vector<vector<size_t>> getNumberOfCriticalCells(const value_t& threshold=INFTY) const;
	void checkV() const;
	void checkBoundaryAndCoboundary() const;
	void printNumberOfCriticalCells(const value_t& threshold=INFTY) const;

private:
	value_t*** allocateMemory() const;
	void getGridFromVector(const vector<value_t>& vector);
	void setValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value);
	void addValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value);
	size_t hashVoxel(const vector<index_t>& voxelCoordinates) const;
	Cube getCube(const index_t& x, const index_t& y, const index_t& z) const;
	vector<index_t> getParentVoxel(const Cube& cube) const;
	value_t getMinimumDistance();
	vector<Cube> getLowerStar(const index_t& x, const index_t& y, const index_t& z) const;
	vector<Cube> getLowerStarWithoutPerturbation(const index_t& x, const index_t& y, const index_t& z) const;
	void getLowerStarsWithoutPerturbation(vector<vector<Cube>>& lowerStars) const;
	void getLowerStarsWithoutPerturbationBetween(vector<vector<Cube>>& lowerStars, 
													const index_t& xMin, const index_t& xMax,
													const index_t& yMin, const index_t& yMax,
													const index_t& zMin, const index_t& zMax) const;
	size_t numUnpairedFaces(const Cube& cube, const vector<Cube>& L) const;
	Cube getUnpairedFace(const Cube& cube, const vector<Cube>& L) const;
	void insertToC(const Cube& cube);
	void insertToV(const Cube& cube0, const Cube& cube1);
	void processLowerStarsWithPerturbation();
	void processLowerStarsWithoutPerturbation();
	void processLowerStarsBetween(const index_t& xMin, const index_t& xMax, const index_t& yMin, const index_t& yMax,
									const index_t& zMin, const index_t& zMax);
	void processLowerStarsWithoutPerturbationBetween(const index_t& xMin, const index_t& xMax, const index_t& yMin,
														const index_t& yMax, const index_t& zMin, const index_t& zMax);
	void processLowerStarsWithPerturbationParallel(const index_t& xPatch = 1, const index_t& yPatch = 1, const index_t& zPatch = 1);
	void processLowerStarsWithoutPerturbationParallel(const index_t& xPatch = 1, const index_t& yPatch = 1, 
														const index_t& zPatch = 1);
	void traverseFlow(const Cube& s, vector<tuple<Cube, Cube, Cube>>& flow, bool coordinated=true) const;
	void traverseCoflow(const Cube& s, vector<tuple<Cube, Cube, Cube>>& flow, bool coordinated=true) const;
	void getConnections(const Cube&s, const Cube& t, vector<tuple<Cube, Cube, Cube>>& connections) const;
	void cancelPair(const Cube&s, const Cube& t);
	const vector<index_t> shape;
	index_t mYZ = shape[1]*shape[2];
	value_t*** grid;
	bool perturbed;
	bool processedLowerStars;
	value_t perturbation;
	vector<vector<Cube>> C;
#ifdef USE_CUBE_MAP
	CubeMap<Cube> V;
	CubeMap<Cube> coV;
#else
	unordered_map<Cube, Cube, Cube::Hash> V;
	unordered_map<Cube, Cube, Cube::Hash> coV;
#endif
	mutable std::mutex mutexC;
	mutable std::mutex mutexV;
	set<Cube> morseSkeletonBelow;
	set<Cube> morseSkeletonAbove;
	set<vector<index_t>> morseSkeletonVoxelsBelow;
	set<vector<index_t>> morseSkeletonVoxelsAbove;
};