#include "data_structures.h"
#include "enumerators.h"
#include "utils.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>

using namespace std;



Cube::Cube() : birth(0), x(0), y(0), z(0), type(0), dim(0) {}


Cube::Cube(const value_t& _birth, const index_t& _x, const index_t& _y, const index_t& _z, 
			const uint8_t& _type, const uint8_t& _dim) :
	birth(_birth), x(_x), y(_y), z(_z), type(_type), dim(_dim) {}


Cube::Cube(const MorseComplex& mc, const index_t& _x, const index_t& _y, const index_t& _z, 
			const uint8_t& _type, const uint8_t& _dim) :
	birth(mc.getBirth(_x, _y, _z, _type, _dim)), x(_x), y(_y), z(_z), type(_type), dim(_dim) {}


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


void Cube::print() const {
	cout << "(" << birth << "," << x << "," << y << "," << z << "," << unsigned(type) << ","
		<< unsigned(dim) << ")";
}



MorseComplex::MorseComplex(const vector<value_t>&& image, const vector<index_t>&& _shape) : 
	shape(_shape), C(4), perturbed(false) { getGridFromVector(image); }


MorseComplex::~MorseComplex() {
	if (grid != nullptr) {
		for (index_t i = 0; i < shape[0]+2; ++i) {
			for (index_t j = 0; j < shape[1]+2; ++j) { delete[] grid[i][j]; }
			delete[] grid[i];
		}
		delete[] grid;
	}
}


value_t MorseComplex::getValue(const index_t& x, const index_t&  y, const index_t& z) const { 
	return grid[x+1][y+1][z+1];
}


value_t MorseComplex::getBirth(const index_t& x, const index_t& y, const index_t& z, 
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


Cube MorseComplex::getCube(const index_t& x, const index_t& y, const index_t& z) const {
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


vector<Cube> MorseComplex::getFaces(const Cube& cube) { return faces[cube]; }


void MorseComplex::perturbImage() {
	if (perturbed) { 
		cout << "already perturbed!" << endl;
		return;
	}

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


void MorseComplex::processLowerStars() {
	vector<Cube> L;
	priority_queue<Cube, vector<Cube>, ReverseOrder> PQzero;
	priority_queue<Cube, vector<Cube>, ReverseOrder> PQone;
	Cube alpha;
	Cube pair;

	for (index_t x = 0; x < shape[0]; ++x) {
		for (index_t y = 0; y < shape[1]; ++y) {
			for (index_t z = 0; z < shape[2]; ++z) {

				L = getLowerStar(x, y, z);
				if (L.size() == 1) { C[0].push_back(L[0]); }
				else {
					sort(L.begin(), L.end());
					alpha = L[1];
					V.emplace(L[0], alpha); Vdual.emplace(alpha, L[0]);
					L.erase(L.begin(), L.begin()+2);
					
					for (const Cube& beta : L) {
						if (beta.dim == 1) { PQzero.push(beta); }
						else if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { PQone.push(beta); }
					}

					while(!PQzero.empty() || !PQone.empty()) {
						while(!PQone.empty()) {
							alpha = PQone.top(); PQone.pop();
							if (numUnpairedFaces(alpha, L) == 0) { PQzero.push(alpha); }
							else {	
								pair = unpairedFace(alpha, L);
								V.emplace(pair, alpha); Vdual.emplace(alpha, pair);
								removeFromPQ(pair, PQzero);
								for (const Cube& beta : L) {
									if ((alpha.isFaceOf(beta) || pair.isFaceOf(beta)) 
											&& numUnpairedFaces(beta, L) == 1) {
										PQone.push(beta);
									}
								}
							}
						}
						if (!PQzero.empty()) {
							alpha = PQzero.top();
							PQzero.pop();
							C[alpha.dim].push_back(alpha);
							for (const Cube& beta : L) {
								if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) {
									PQone.push(beta);
								}
							}
						}
					}
				}
			}
		}
	}
}


void MorseComplex::extractMorseComplex() {
	BoundaryEnumerator enumerator(*this);
	Cube alpha;
	Cube beta;

	for (uint8_t dim = 1; dim < 4; ++dim) {
		for (const Cube& c : C[dim]) {
			priority_queue<Cube> Qbfs;

			enumerator.setBoundaryEnumerator(c);
			while (enumerator.hasNextFace()) {
				if (find(C[c.dim-1].begin(), C[c.dim-1].end(), enumerator.nextFace) != C[c.dim-1].end()) {
					faces[c].push_back(enumerator.nextFace);
				}
				auto it = V.find(enumerator.nextFace);
				if (it != V.end()) { Qbfs.push(enumerator.nextFace); }
			}

			while (!Qbfs.empty()) {
				alpha = Qbfs.top();
				Qbfs.pop();
				beta = V[alpha];

				enumerator.setBoundaryEnumerator(beta);
				while (enumerator.hasNextFace()) {
					if (enumerator.nextFace != alpha) {
						if (find(C[c.dim-1].begin(), C[c.dim-1].end(), enumerator.nextFace)
									!= C[c.dim-1].end()) { faces[c].push_back(enumerator.nextFace); }
						else {
							auto it = V.find(enumerator.nextFace);
							if (it != V.end()) { Qbfs.push(enumerator.nextFace); }
						}
					}
				}
			}
		}
	}
}


void MorseComplex::checkGradientVectorfield() const {
	size_t counter;
	value_t birth;
	Cube cube;
	for (index_t x = 0; x < shape[0]; ++x) {
		for (index_t y = 0; y < shape[1]; ++y) {
			for (index_t z = 0; z < shape[2]; ++z) {
				for (uint8_t type = 0; type < 3; ++type) {
					for (uint8_t dim = 0; dim < 4; ++dim) {
						if ((dim == 0 && type != 0) || (dim == 3 && type != 0)) { continue; }
						birth = getBirth(x, y, z, type, dim);
						if (birth == INFTY) { continue; }
						cube = Cube(birth, x, y, z, type, dim);
						counter = 0;
						if (V.count(cube) != 0) { ++counter; }
						if (Vdual.count(cube) != 0) { ++counter; }
						if (find(C[dim].begin(), C[dim].end(), cube) != C[dim].end()) { ++counter; }
						if (counter != 1) { cube.print(); cout << " occurs " << counter << " times!" << endl; }
					}
				}
			}
		}
	}
}


void MorseComplex::printGradientVectorfield() const {
	cout << "critical: " << endl;
	for (uint8_t dim = 0; dim < 4; ++dim) {
		for (const Cube& c : C[dim]) {
			c.print(); cout << endl;
		}
	}

	cout << "V: " << endl;
	for (auto it = V.begin(); it != V.end(); ++it) {
        const Cube& key = it->first;
		(it->first).print(); cout << " -> "; (it->second).print(); cout << endl;
    }
}


void MorseComplex::printGradientVectorfieldImage() const {
	Cube cube;
	for (size_t y = 0; y < 2*shape[1]-1; ++y) {
		for (size_t x = 0; x < 2*shape[0]-1; ++x) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				cube = getCube(x, y, z);
				if (find(C[cube.dim].begin(), C[cube.dim].end(), cube) != C[cube.dim].end()) { cout << "c "; }
				else if (V.count(cube) != 0) { cout << "p "; } 
				else if (Vdual.count(cube) != 0) { cout << "q "; }
				else { cout << "  "; }
			}
			cout << "  ";
		}
		cout << endl;
	}
}


void MorseComplex::printGradientVectorfieldDim(uint8_t dim) const {
	Cube cube;
	unordered_map<Cube, index_t, Cube::Hash> paired;
	index_t counter;
	for (size_t y = 0; y < 2*shape[1]-1; ++y) {
		for (size_t x = 0; x < 2*shape[0]-1; ++x) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				cube = getCube(x, y, z);
				if (cube.dim != dim && cube.dim != dim-1) { 
					cout << "xx ";
					continue;
				}
				if (find(C[cube.dim].begin(), C[cube.dim].end(), cube) != C[cube.dim].end() &&
					(cube.dim == dim)) { cout << "CC "; }
				else if (find(C[cube.dim].begin(), C[cube.dim].end(), cube) != C[cube.dim].end() &&
					(cube.dim == dim-1)) { cout << "cc "; }
				else {
					auto it = paired.find(cube);
					if (it != paired.end()) {
						if (it-> second < 10) {
							cout << " " << it->second << " ";
						} else { cout << it->second << " "; }
						continue;
					}
					if (cube.dim == dim-1) {
						auto it = V.find(cube);
						if (it != V.end()) {
							if (counter < 10) {
								cout << " " << counter << " ";
							} else { cout << counter << " "; }
							paired.emplace(it->second, counter);
							++counter;
							continue;
						}
					}
					else if (cube.dim == dim) {
						auto it = Vdual.find(cube);
						if (it != Vdual.end()) {
							if (counter < 10) {
								cout << " " << counter << " ";
							} else { cout << counter << " "; }
							 paired.emplace(it->second, counter);
							 ++counter;
							 continue;
						}
					}
					cout << "XX ";
				}
			}
			cout << "  ";
		}
		cout << endl;
	}
}


void MorseComplex::printFaces() {
	for (uint8_t dim = 0; dim < 4; ++dim) {
		cout << "dim " << unsigned(dim) << ":" << endl;
		printGradientVectorfieldDim(dim);
		for (const Cube& c : C[dim]) {
			cout << "cube: "; c.print(); cout << endl;
			cout << "faces: ";
			for (const Cube& face : faces[c]) {
				face.print(); cout << " ";
			}
			cout << endl;
		}
	}
}


void MorseComplex::printImage() const {
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


value_t*** MorseComplex::allocateMemory() const {
	value_t*** g = new value_t**[shape[0]+2];
    for (index_t i = 0; i < shape[0]+2; ++i) {
        g[i] = new value_t*[shape[1]+2];
        for (index_t j = 0; j < shape[1]+2; ++j) { g[i][j] = new value_t[shape[2]+2]; }
    }
	if (g == NULL) { cerr << "Out of memory!" << endl; }
	return g;
}


void MorseComplex::getGridFromVector(const vector<value_t>& vec) {
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


void MorseComplex::setValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value) {
	grid[x+1][y+1][z+1] = value;
}


void MorseComplex::addValue(const index_t& x, const index_t& y, const index_t& z, const value_t& value) {
	grid[x+1][y+1][z+1] += value;
}


value_t MorseComplex::findMinimumDistance() {
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


vector<Cube> MorseComplex::getLowerStar(const index_t& x, const index_t& y, const index_t& z) const {
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


size_t MorseComplex::numUnpairedFaces(const Cube& cube, const vector<Cube>& L) {
	size_t counter = 0;
	for (const Cube& l : L) { 
		if (l.isFaceOf(cube) && find(C[l.dim].begin(), C[l.dim].end(), l) == C[l.dim].end()
				&& V.count(l) == 0 && Vdual.count(l) == 0) { 
			++counter;
		}
	}

	return counter;
}


Cube MorseComplex::unpairedFace(const Cube& cube, const vector<Cube>& L) {
	for (const Cube& l : L) { 
		if (l.isFaceOf(cube) && find(C[l.dim].begin(), C[l.dim].end(), l) == C[l.dim].end()
				&& V.count(l) == 0 && Vdual.count(l) == 0) { 
			return l;
		}
	}
	
	return cube;
}