#include "data_structures.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;



Cube::Cube() : birth(0), x(0), y(0), z(0), type(0), dim(0) {}


Cube::Cube(value_t _birth, index_t _x, index_t _y, index_t _z, uint8_t _type, uint8_t _dim) :
	birth(_birth), x(_x), y(_y), z(_z), type(_type), dim(_dim) {}


Cube::Cube(const Cube& other) : birth(other.birth), x(other.x), y(other.y), z(other.z), 
	type(other.type), dim(other.dim) {}


Cube& Cube::operator=(const Cube& rhs) {
	if (this != &rhs) {
		birth = rhs.birth;
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
		type = rhs.type;
		dim = rhs.dim;
	}
	return *this;
}


bool Cube::operator==(const Cube& rhs) const{ 
	return (x == rhs.x && y == rhs.y && z == rhs.z && type == rhs.type && dim == rhs.dim);
}


bool Cube::operator!=(const Cube& rhs) const {
    return !(*this == rhs);
}


bool Cube::operator<(const Cube& rhs) const {
	if (birth != rhs.birth) { return birth < rhs.birth; }
	else if (dim != rhs.dim) { return dim < rhs.dim; }
	else if (x != rhs.x) { return x < rhs.x; }
	else if (y != rhs.y) { return y < rhs.y; }
	else if (z != rhs.z) { return z < rhs.z; }
	return type < rhs.type;
}


vector<vector<index_t>> Cube::getVertices() const {
	vector<vector<index_t>> vertices = {{x, y, z}};
	switch(dim) {
		case 0:
			return vertices;

		case 1:
			switch(type) {
				case 0:
					vertices.push_back({x+1, y, z});
					return vertices;

				case 1:
					vertices.push_back({x, y+1, z});
					return vertices;

				case 2:
					vertices.push_back({x, y, z+1});
					return vertices;
			}

		case 2:
			switch(type) {
				case 0:
					vertices.push_back({x, y+1, z});
					vertices.push_back({x, y, z+1});
					vertices.push_back({x, y+1, z+1});
					return vertices;

				case 1:
					vertices.push_back({x+1, y, z});
					vertices.push_back({x, y, z+1});
					vertices.push_back({x+1, y, z+1});
					return vertices;

				case 2:
					vertices.push_back({x+1, y, z});
					vertices.push_back({x, y+1, z});
					vertices.push_back({x+1, y+1, z});
					return vertices;
			}

		case 3:
			vertices.push_back({x+1, y, z});
			vertices.push_back({x, y+1, z});
			vertices.push_back({x, y, z+1});
			vertices.push_back({x+1, y+1, z});
			vertices.push_back({x+1, y, z+1});
			vertices.push_back({x, y+1, z+1});
			vertices.push_back({x+1, y+1, z+1});
			return vertices;
	}

	return vertices;
}


bool Cube::isFaceOf(const Cube& other) const {
	if (dim > other.dim-1) { return false; }

	vector<vector<index_t>> vertices = getVertices();
	vector<vector<index_t>> verticesOther = other.getVertices();
	for (vector<index_t>& vertex : vertices) {
		if (find(verticesOther.begin(), verticesOther.end(), vertex) == verticesOther.end()) { 
			return false;
		}
	}

	return true;
}


void Cube::removeFromPQ(priority_queue<Cube>& PQ) const {
    vector<Cube> temp;
    while (!PQ.empty()) {
        if (PQ.top() != *this) {
            temp.push_back(PQ.top());
        }
        PQ.pop();
    }

    make_heap(temp.begin(), temp.end());
    PQ = priority_queue<Cube>(temp.begin(), temp.end());
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


Cube CubicalGridComplex::getCube(const index_t& x, const index_t& y, const index_t& z) const {
	uint8_t dim = 0;
	if (x%2 != 0) { ++dim; }
	if (y%2 != 0) { ++dim; }
	if (z%2 != 0) { ++dim; }

	index_t xCube = x/2;
	index_t yCube = y/2;
	index_t zCube = z/2;

	uint8_t type;
	switch(dim) {
		case 0:
			type = 0;
			break;

		case 1:
			if (x%2 != 0) { type = 0; }
			if (y%2 != 0) { type = 1; }
			if (z%2 != 0) { type = 2; }
			break;

		case 2:
			if (x%2 != 0 && y%2 != 0) { type = 2; }
			if (x%2 != 0 && z%2 != 0) { type = 1; }
			if (y%2 != 0 && z%2 != 0) { type = 0; }
			break;

		case 3:
			type = 0;
			break;
	}

	value_t birth = getBirth(xCube, yCube, zCube, type, dim);

	return Cube(birth, xCube, yCube, zCube, type, dim);
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


void CubicalGridComplex::processLowerStars() {
	vector<Cube> L;
	vector<Cube> cache;
	priority_queue<Cube> PQzero;
	priority_queue<Cube> PQone;
	Cube alpha;
	Cube gamma;
	Cube delta;

	printGradientVectorfieldImage(); cout << endl;

	for (index_t x = 0; x < shape[0]; ++x) {
		for (index_t y = 0; y < shape[1]; ++y) {
			for (index_t z = 0; z < shape[2]; ++z) {
				
				cout << "pixel: " << x << " " << y << " " << z << endl;

				L = getLowerStar(x, y, z);
				
				if (L.size() == 1) { C.push_back(L[0]); }
				else {
					sort(L.begin(), L.end());
					delta = L[1];
					V.emplace(L[0], delta); Vdual.emplace(delta, L[0]);
					L.erase(L.begin(), L.begin()+2);
					for (const Cube& beta : L) {
						if (beta.dim == 1) { PQzero.push(beta); }
						else {
							if (delta.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) {
								PQone.push(beta);
							} else { cache.push_back(beta); }
						}
					}
					L = cache;
					cache.clear();

					while(!PQzero.empty() || !PQone.empty()) {
						while(!PQone.empty()) {
							alpha = PQone.top();
							PQone.pop();
							if (numUnpairedFaces(alpha, L) == 0) { PQzero.push(alpha); }
							else {
								V.emplace(p, alpha); Vdual.emplace(alpha, p);
								p.removeFromPQ(PQzero);
								for (const Cube& beta : L) {
									if ((alpha.isFaceOf(beta) || p.isFaceOf(beta)) 
											&& numUnpairedFaces(beta, L) == 1) {
										PQone.push(beta);
									} else { cache.push_back(beta); }
								}
								L = cache;
								cache.clear();
							}
						}
					
						if (!PQzero.empty()) {
							gamma = PQzero.top();
							PQzero.pop();
							C.push_back(gamma);
							for (const Cube& beta : L) {
								if (gamma.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) {
									PQone.push(beta);
								} else { cache.push_back(beta); }
							}
							L = cache;
							cache.clear();
						}
					}
				}

				printGradientVectorfieldImage(); cout << endl;

			}
		}
	}
}


void CubicalGridComplex::printGradientVectorfield() const {
	cout << "critical: " << endl;
	for (const Cube& c : C) {
		c.print(); cout << endl;
	}
	cout << "V: " << endl;
	for (auto it = V.begin(); it != V.end(); ++it) {
        const Cube& key = it->first;
		(it->first).print(); cout << " -> "; (it->second).print(); cout << endl;
    }
}


void CubicalGridComplex::printGradientVectorfieldImage() const {
	Cube cube;
	for (size_t y = 0; y < 2*shape[1]-1; ++y) {
		for (size_t x = 0; x < 2*shape[0]-1; ++x) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				cube = getCube(x, y, z);
				if (find(C.begin(), C.end(), cube) != C.end()) { cout << "c "; }
				else if (V.count(cube) != 0) { cout << "p "; } 
				else if (Vdual.count(cube) != 0) { cout << "q "; }
				else { cout << "x "; }
			}
			cout << "  ";
		}
		cout << endl;
	}
}


void CubicalGridComplex::printImage() const {
    value_t value;
    for (size_t y = 0; y < shape[1]; ++y) {
		for (size_t x = 0; x < shape[0]; ++x) {
            for (size_t z = 0; z < shape[2]; ++z) {
                value = getValue(x, y, z);
                if (value < 10) { cout << " " << fixed << setprecision(3) << value << " "; }
				else { cout << fixed << setprecision(3) << value << " "; }            
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


size_t CubicalGridComplex::numUnpairedFaces(const Cube& cube, const vector<Cube>& L) {
	size_t counter;

	for (const Cube& l : L) {
		if (l.isFaceOf(cube) && V.count(l) == 0 && Vdual.count(l) == 0) { 
			++counter;
			p = l;
		}
	}

	return counter;
}