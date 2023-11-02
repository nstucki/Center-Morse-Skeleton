#include "data_structures.h"

#include <iostream>
#include <algorithm>
#include <iomanip>

using namespace std;



Cube::Cube() : birth(0), x(0), y(0), z(0), type(0), dim(0) {}


Cube::Cube(value_t _birth, index_t _x, index_t _y, index_t _z, uint8_t _type, uint8_t _dim) :
	birth(_birth), x(_x), y(_y), z(_z), type(_type), dim(_dim) {}


bool Cube::operator==(const Cube& rhs) const{ 
	return (x == rhs.x && y == rhs.y && z == rhs.z && type == rhs.type && dim == rhs.dim);
}

bool Cube::operator<(const Cube& other) const {
	if (birth != other.birth) { return birth < other.birth; }
	else if (dim != other.dim) { return dim < other.dim; }
	else if (x != other.x) { return x < other.x; }
	else if (y != other.y) { return y < other.y; }
	else if (z != other.z) { return z < other.z; }
	return type < other.type;
}


void Cube::print() const {
	cout << "(" << birth << "," << x << "," << y << "," << z << "," << unsigned(type) << ","
		<< unsigned(dim) << ")";
}



CubicalGridComplex::CubicalGridComplex(const vector<value_t>&& image, const vector<index_t>&& _shape) : 
	shape(_shape), perturbed(false) { getGridFromVector(image); }


CubicalGridComplex::~CubicalGridComplex() {
	if (grid != nullptr) {
		for (index_t i = 0; i < shape[0]+2; ++i) {
			for (index_t j = 0; j < shape[1]+2; ++j) { delete[] grid[i][j]; }
			delete[] grid[i];
		}
		delete[] grid;
	}
}


value_t CubicalGridComplex::getValue(const index_t& x, const index_t&  y, const index_t& z) const { 
	return grid[x+1][y+1][z+1];
}


value_t CubicalGridComplex::getBirth(const index_t& x, const index_t& y, const index_t& z, 
										const uint8_t& type, const uint8_t& dim) const {
	switch (dim) {
		case 0:
			return getValue(x, y, z);

		case 1:
			switch (type) {
				case 0:
					return max(getValue(x, y, z), getValue(x+1, y, z));

				case 1:
					return max(getValue(x, y, z), getValue(x, y+1, z));

				case 2:
					return max(getValue(x, y, z), getValue(x, y, z+1));
			}

		case 2:
			switch (type) {
				case 0:
					return max({getValue(x, y, z), getValue(x, y+1, z), 
								getValue(x, y, z+1), getValue(x, y+1, z+1)});

				case 1:
					return max({getValue(x, y, z), getValue(x+1, y, z), 
								getValue(x, y, z+1), getValue(x+1, y, z+1)});

				case 2:
					return max({getValue(x, y, z), getValue(x+1, y, z), 
								getValue(x, y+1, z) , getValue(x+1, y+1, z)});
			}
			
		case 3:
			return max({getValue(x, y, z), getValue(x+1, y+1, z),  getValue(x+1, y, z+1), getValue(x, y+1, z+1),
						getValue(x+1, y, z), getValue(x, y+1, z), getValue(x, y, z+1), getValue(x+1, y+1, z+1)});
	}

	cerr << "birth not found!" << endl;
	return INFTY;
}


void CubicalGridComplex::perturbImage() {
	perturbed = true;
	value_t minDistance = findMinimumDistance();

	if (minDistance != 0) {
		value_t denom = 3*shape[0]*shape[1]*shape[2];
		for (size_t x = 0; x < shape[0]; ++x) {
			for (size_t y = 0; y < shape[1]; ++y) {
				for (size_t z = 0; z < shape[2]; ++z) {
					addValue(x, y, z, minDistance*(x+shape[0]*y+shape[0]*shape[1]*z)/denom);          
				}
			}
		}
	}

}


void CubicalGridComplex::printImage() const {
    value_t value;
    for (size_t y = 0; y < shape[1]; ++y) {
		for (size_t x = 0; x < shape[0]; ++x) {
            for (size_t z = 0; z < shape[2]; ++z) {
                value = getValue(x, y, z);
                if (value < 10) { cout << ' ' << fixed << setprecision(3) << value << ' '; }
				else { cout << fixed << setprecision(3) << value << ' '; }            
            }
            cout << "  ";
        }
        cout << endl;
    }
}


value_t*** CubicalGridComplex::allocateMemory() const {
	value_t*** g = new value_t**[shape[0]+2];
    for (index_t i = 0; i < shape[0]+2; ++i) {
        g[i] = new value_t*[shape[1]+2];
        for (index_t j = 0; j < shape[1]+2; ++j) { g[i][j] = new value_t[shape[2]+2]; }
    }
	if (g == NULL) { cerr << "Out of memory!" << endl; }
	return g;
}


void CubicalGridComplex::getGridFromVector(const vector<value_t>& vec) {
	size_t counter = 0;
	grid = allocateMemory();
	for (index_t x = 0; x < shape[0]+2; ++x) {
		for (index_t y = 0; y < shape[1]+2; ++y) {
			for (index_t z = 0; z < shape[2]+2; ++z) {
				if (x == 0 || x == shape[0]+1 || y == 0 || y == shape[1]+1 || z == 0 || z == shape[2]+1) {
					grid[x][y][z] = INFTY;
				}
				else { grid[x][y][z] = vec[counter++]; }
			}
		}
	}
}


void CubicalGridComplex::setValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value) {
	grid[x+1][y+1][z+1] = value;
}


void CubicalGridComplex::addValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value) {
	grid[x+1][y+1][z+1] += value;
}


value_t CubicalGridComplex::findMinimumDistance() {
    value_t minDistance = std::numeric_limits<value_t>::max();
	bool needPerturbation = false;
    for (size_t x1 = 0; x1 < shape[0]; ++x1) {
        for (size_t y1 = 0; y1 < shape[1]; ++y1) {
            for (size_t z1 = 0; z1 < shape[2]; ++z1) {
                for (size_t x2 = 0; x2 < shape[0]; ++x2) {
                    for (size_t y2 = 0; y2 < shape[1]; ++y2) {
                        for (size_t z2 = 0; z2 < shape[2]; ++z2) {
                            if (x1 != x2 || y1 != y2 || z1 != z2) {
                                value_t distance = abs(getValue(x1, y1, z1) - getValue(x2, y2, z2));
								if (distance == 0) { needPerturbation = true; }
                                else if (distance < minDistance) { minDistance = distance; }
                            }
                        }
                    }
                }
            }
        }
    }

	if (needPerturbation) { return minDistance; }
	else { return 0; }
}


vector<Cube> CubicalGridComplex::getLowerStar(const index_t& x, const index_t& y, const index_t& z) const {
	vector<Cube> L;
	value_t value = getValue(x, y, z);
	L.push_back(Cube(value, x, y, z, 0, 0));

	value_t birth;
	for (int8_t delta = -1; delta <= 0; ++delta) {
		birth = getBirth(x+delta, y, z, 0, 1);
		if (birth == value) { L.push_back(Cube(birth, x+delta, y, z, 0, 1)); }

		birth = getBirth(x, y+delta, z, 1, 1);
		if (birth == value) { L.push_back(Cube(birth, x, y+delta, z, 1, 1)); }

		birth = getBirth(x, y, z+delta, 2, 1);
		if (birth == value) { L.push_back(Cube(birth, x, y, z+delta, 2, 1)); }
	}

	for (int8_t delta0 = -1; delta0 <= 0; ++delta0) {
		for (int8_t delta1 = -1; delta1 <= 0; ++delta1) {
			birth = getBirth(x+delta0, y+delta1, z, 2, 2);
			if (birth == value) { L.push_back(Cube(birth, x+delta0, y+delta1, z, 2, 2)); }

			birth = getBirth(x+delta0, y, z+delta1, 1, 2);
			if (birth == value) { L.push_back(Cube(birth, x+delta0, y, z+delta1, 1, 2)); }

			birth = getBirth(x, y+delta0, z+delta1, 0, 2);
			if (birth == value) { L.push_back(Cube(birth, x, y+delta0, z+delta1, 0, 2)); }
		}
	}

	for (int8_t delta0 = -1; delta0 <= 0; ++delta0) {
		for (int8_t delta1 = -1; delta1 <= 0; ++delta1) {
			for (int8_t delta2 = -1; delta2 <= 0; ++delta2) {
				birth = getBirth(x+delta0, y+delta1, z+delta2, 0, 3);
				if (birth == value) { L.push_back(Cube(birth, x+delta0, y+delta1, z+delta2, 0, 3)); }
			}
		}
	}

	return L;
}


void processLowerStars() {
	
}