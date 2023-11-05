#pragma once

#include "config.h"

#include <vector>

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
	~MorseComplex();
	value_t getValue(const index_t& x, const index_t& y, const index_t& z) const;
	value_t getBirth(const index_t& x, const index_t& y, const index_t& z, 
						const uint8_t& type, const uint8_t& dim) const;
	Cube getCube(const index_t& x, const index_t& y, const index_t& z) const;
	vector<Cube> getFaces(const Cube& cube);
	void perturbImage();
	void processLowerStars();
	void extractMorseComplex();
	void checkGradientVectorfield() const;
	void printGradientVectorfield() const;
	void printGradientVectorfieldImage() const;
	void printGradientVectorfieldDim(uint8_t dim) const;
	void printFaces();
	void printImage() const;

	private:
	value_t*** allocateMemory() const;
	void getGridFromVector(const vector<value_t>& vector);
	void setValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value);
	void addValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value);
	value_t findMinimumDistance();
	vector<Cube> getLowerStar(const index_t& x, const index_t& y, const index_t& z) const;
	size_t numUnpairedFaces(const Cube& cube, const vector<Cube>& L);
	Cube unpairedFace(const Cube& cube, const vector<Cube>& L);
	value_t*** grid;
	const vector<index_t> shape;
	vector<vector<Cube>> C;
	unordered_map<Cube, Cube, Cube::Hash> V;
	unordered_map<Cube, Cube, Cube::Hash> Vdual;
	unordered_map<Cube, vector<Cube>, Cube::Hash> faces;
	bool perturbed;
};