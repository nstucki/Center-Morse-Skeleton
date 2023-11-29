#include "data_structures.h"
#include "enumerators.h"
#include "utils.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <cstdlib>

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


MorseComplex::MorseComplex(MorseComplex&& other) : shape(other.shape), C(4) {
	grid = other.grid;
	other.grid = nullptr;
}


MorseComplex::~MorseComplex() {
	if (grid != nullptr) {
		for (index_t i = 0; i < shape[0]+2; ++i) {
			for (index_t j = 0; j < shape[1]+2; ++j) { delete[] grid[i][j]; }
			delete[] grid[i];
		}
		delete[] grid;
	}
}


value_t MorseComplex::getValue(const index_t& x, const index_t& y, const index_t& z) const {
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


void MorseComplex::perturbImage(const value_t& epsilon) {
	if (perturbed) { 
		cout << "Image is already perturbed!" << endl;
		return;
	}

	if (epsilon == INFTY) { perturbation = getMinimumDistance(); }
	else { perturbation = epsilon; }

	if (perturbation != 0) {
		value_t denom = 3*shape[0]*shape[1]*shape[2];
		for (size_t x = 0; x < shape[0]; ++x) {
			for (size_t y = 0; y < shape[1]; ++y) {
				for (size_t z = 0; z < shape[2]; ++z) {
					addValue(x, y, z, perturbation*(x+shape[0]*y+shape[0]*shape[1]*z)/denom);          
				}
			}
		}
	}

	perturbed = true;
}


void MorseComplex::processLowerStars() {
	if (!perturbed) { 
		cerr << "Perturb Image first!" << endl;
		return;
	} else if (processedLowerStars) {
		cerr << "Lower stars already processed!" << endl;
		return;
	}

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
					V.emplace(L[0], alpha); coV.emplace(alpha, L[0]);
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
								V.emplace(pair, alpha); coV.emplace(alpha, pair);
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


vector<pair<Cube, uint8_t>> MorseComplex::getMorseBoundary(const Cube& s) const {
	unordered_map<Cube, uint8_t, Cube::Hash> count;
	count.emplace(s, 1);
	set<Cube> boundary;

	vector<tuple<Cube, Cube,Cube>> flow;
	traverseFlow(s, flow);

	uint8_t n;
	for (const tuple<Cube, Cube, Cube>& f : flow) {
		auto it = count.find(get<2>(f));
		if (it != count.end()) { n = count[get<0>(f)] + it->second; }
		else { n = count[get<0>(f)]; }
		if (n > 3) { count.insert_or_assign(get<2>(f), n%2 + 2); }
		else { count.insert_or_assign(get<2>(f), n); }
		if (get<1>(f) == get<2>(f)) { boundary.insert(get<2>(f)); }
	}

	vector<pair<Cube, uint8_t>> result;
	for (const Cube& b : boundary) {
		result.push_back(pair(b, count[b]));
	}

	return result;
}


vector<pair<Cube, uint8_t>> MorseComplex::getMorseCoboundary(const Cube& s) const {
	unordered_map<Cube, uint8_t, Cube::Hash> count;
	count.emplace(s, 1);
	set<Cube> coboundary;

	vector<tuple<Cube, Cube,Cube>> flow;
	traverseCoflow(s, flow);

	uint8_t n;
	for (const tuple<Cube, Cube, Cube>& f : flow) {
		auto it = count.find(get<2>(f));
		if (it != count.end()) { n = count[get<0>(f)] + it->second; }
		else { n = count[get<0>(f)]; }
		if (n > 3) { count.insert_or_assign(get<2>(f), n%2 + 2); }
		else { count.insert_or_assign(get<2>(f), n); }
		if (get<1>(f) == get<2>(f)) { coboundary.insert(get<2>(f)); }
	}

	vector<pair<Cube, uint8_t>> result;
	for (const Cube& c : coboundary) {
		result.push_back(pair(c, count[c]));
	}

	return result;
}


void MorseComplex::extractMorseSkeleton(const value_t& threshold) {
	vector<tuple<Cube, Cube, Cube>> flow;
	for (uint8_t dim = 0; dim < 4; ++dim) {
		for (const Cube& c : C[dim]) {
			if (c.birth > threshold) { continue; }
			flow.clear();
			traverseFlow(c, flow);
			for (const tuple<Cube, Cube, Cube>& t : flow) {
				if (get<1>(t) != get<2>(t)) { morseSkeleton.insert(get<2>(t)); }
			}
		}
	}

	vector<vector<index_t>> pixels;
	for (const Cube& c : morseSkeleton) {
		pixels = c.getVertices();
		for (const vector<index_t>& p : pixels) {
			morseSkeletonPixels.insert(p);
		}

	}
}


void MorseComplex::cancelPairsBelow(const value_t& threshold, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printC(threshold);
	}

	if (threshold == -INFTY) { return; }

	bool canceled = true;
	vector<Cube> cancelable;

	while ((C[1].size() != 0 || C[2].size() != 0 || C[3].size() != 0) && canceled) {
		canceled = false;
		for (uint8_t dim = 4; dim-- > 1;) {
			for (const Cube& s : C[dim]) {
				if (s.birth >= threshold) { continue; }

				vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(s);

				cancelable.clear();
				for (const pair<Cube, uint8_t> b : boundary) {
					if (get<1>(b) == 1) { cancelable.push_back(get<0>(b)); }
				}
				if (cancelable.size() == 0) { continue; }
				sort(cancelable.begin(), cancelable.end());

				cancelPair(s, cancelable.back());
				if (print) { printC(threshold); }

				canceled = true;
				break;
			}
			if (canceled) { break; }
		}
	}
	if (print) { cout << endl; }
}


void MorseComplex::cancelPairsAbove(const value_t& threshold, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printC(threshold);
	}

	if (threshold == INFTY) { return; }

	bool canceled = true;
	vector<Cube> cancelable;

	while ((C[1].size() != 0 || C[2].size() != 0 || C[3].size() != 0) && canceled) {
		canceled = false;
		for (uint8_t dim = 4; dim-- > 1;) {
			for (const Cube& s : C[dim]) {
				if (s.birth < threshold) { continue; }

				vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(s);

				cancelable.clear();
				for (const pair<Cube, uint8_t> b : boundary) {
					if (get<1>(b) == 1) { cancelable.push_back(get<0>(b)); }
				}
				if (cancelable.size() == 0) { continue; }
				sort(cancelable.begin(), cancelable.end());
				if (cancelable.back().birth < threshold) { continue; }

				cancelPair(s, cancelable.back());
				if (print) { printC(threshold); }

				canceled = true;
				break;
			}
			if (canceled) { break; }
		}
	}
	if (print) { cout << endl; }
}


void MorseComplex::cancelPairsCoordinatedBelow(const value_t& threshold, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printC(threshold);
	}

	if (threshold == -INFTY) { return; }

	for (uint8_t dim = 0; dim < 4; ++dim) { sort(C[dim].begin(), C[dim].end()); }

	bool canceled = true;
	vector<Cube> cancelable;
	while (canceled) {
		canceled = false;
		for (uint8_t dim = 4; dim-- > 1;) {
			for (auto it = C[dim].rbegin(); it != C[dim].rend(); ++it) {
				Cube s = *it;

				if (s.birth >= threshold) { continue; }

				vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(s);

				cancelable.clear();
				for (const pair<Cube, uint8_t> b : boundary) {
					if (get<1>(b) == 1) { cancelable.push_back(get<0>(b)); }
				}
				if (cancelable.size() == 0) { continue; }
				sort(cancelable.begin(), cancelable.end());

				cancelPair(s, cancelable.back());
				canceled = true;

				if (print) { printC(threshold); }
				break;
			}
			if (canceled) { break; }
		}
	}
	if (print) { cout << endl; }
}


void MorseComplex::cancelPairsCoordinatedAbove(const value_t& threshold, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printC(threshold);
	}

	if (threshold == INFTY) { return; }

	for (uint8_t dim = 0; dim < 4; ++dim) { sort(C[dim].begin(), C[dim].end()); }

	bool canceled = true;
	vector<Cube> cancelable;
	while (canceled) {
		canceled = false;
		for (uint8_t dim = 0; dim < 3; ++dim) {
			for (const Cube& t : C[dim]) {
				if (t.birth < threshold) { continue; }

				vector<pair<Cube, uint8_t>> coboundary = getMorseCoboundary(t);

				cancelable.clear();
				for (const pair<Cube, uint8_t> c : coboundary) {
					if (get<1>(c) == 1) { cancelable.push_back(get<0>(c)); }
				}
				if (cancelable.size() == 0) { continue; }
				sort(cancelable.begin(), cancelable.end());

				cancelPair(cancelable.front(), t);
				canceled = true;

				if (print) { printC(threshold); }
				break;
			}
			if (canceled) { break; }
		}
	}
	if (print) { cout << endl; }
}


auto boundaryComparator = [](const auto& lhs, const auto& rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs);
};


void MorseComplex::cancelClosePairsBelow(const value_t& threshold, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printC(threshold);
	}

	bool canceled = true;

	while ((C[1].size() != 0 || C[2].size() != 0 || C[3].size() != 0) && canceled) {
		canceled = false;
		for (uint8_t dim = 4; dim-- > 1;) {
			for (const Cube& s : C[dim]) {
				if (s.birth >= threshold) { continue; }

				vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(s);
				if (boundary.size() == 0) { continue; }
				sort(boundary.begin(), boundary.end(), boundaryComparator);

				value_t maxBirthBoundary = get<0>(boundary.back()).birth;
				for (auto it = boundary.rbegin(); it != boundary.rend(); ++it) {
					if (get<0>(*it).birth < maxBirthBoundary) { break; }
					if (get<1>(*it) == 1) {
						vector<pair<Cube, uint8_t>> coBoundary = getMorseCoboundary(get<0>(*it));
						sort(boundary.begin(), boundary.end(), boundaryComparator);
						for (const pair<Cube, uint8_t>& c : coBoundary) {
							if (get<0>(c).birth < s.birth) { break; }
							if (get<0>(c) == s) {

								cancelPair(s, get<0>(*it));
								if (print) { printC(threshold); }
								
								canceled = true;
								break;
							}
						}
					}
					if (canceled) { break; }
				}
				if (canceled) { break; }
			}
			if (canceled) { break; }
		}
	}
	if (print) { cout << endl; }
}


void MorseComplex::checkV() const {
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
						if (coV.count(cube) != 0) { ++counter; }
						if (find(C[dim].begin(), C[dim].end(), cube) != C[dim].end()) { ++counter; }
						if (counter != 1) { 
							cerr << "Error: "; cube.print(); cout << " occurs " << counter << " times!" << endl;
							exit(EXIT_FAILURE);
						}
					}
				}
			}
		}
	}
	cout << "Gradient vectorfield is ok!" << endl;
}


void MorseComplex::checkBoundaryAndCoboundary() const {
	bool foundCoboundary;
	 for (uint8_t dim = 1; dim < 4; ++dim) {
        for (Cube c : C[dim]) {
            vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(c);
            for (pair<Cube, uint8_t> b : boundary) {
				foundCoboundary = false;
                vector<pair<Cube, uint8_t>> coboundary = getMorseCoboundary(get<0>(b));
                for (pair<Cube, uint8_t> a : coboundary) {
                    if (get<0>(a) == c) {
						foundCoboundary = true;
						if (get<1>(a) != get<1>(b)) {
							cerr << "Mistake in Boundary / Coboundary computation!";
							exit(EXIT_FAILURE);
						}
					}
                }
				if (!foundCoboundary) {
					cerr << "Mistake in Boundary / Coboundary computation!";
					exit(EXIT_FAILURE);
				}
            }
        }
    }
	cout << "Boundary and Coboundary computations are ok!" << endl;
}


value_t MorseComplex::getPerturbation() const {
	return perturbation;
}


vector<vector<Cube>> MorseComplex::getCriticalCells() const { return C; }


vector<vector<size_t>> MorseComplex::getNumberOfCriticalCells(const value_t& threshold) const {
	vector<vector<size_t>> result(3);

	for (uint8_t dim = 0; dim < 4; ++dim) { result[0].push_back(C[dim].size()); }

	if (threshold != INFTY) {
		for (uint8_t dim = 0; dim < 4; ++dim) {
			size_t countBelow = 0;
			size_t countAbove = 0;
			for (const Cube& c : C[dim]) {
				if (c.birth < threshold) { ++countBelow; }
				else { ++countAbove; }
			}
			result[1].push_back(countBelow);
			result[2].push_back(countAbove);
		}
	}

	return result;
}


void MorseComplex::printC(const value_t& threshold) const {
	cout << "\rtotal: ";
	for (uint8_t dim = 0; dim < 4; ++dim) { cout << C[dim].size() << " "; }
	if (threshold != INFTY) {
		cout << "--- < " << threshold << ": ";
		for (uint8_t dim = 0; dim < 4; ++dim) {
			size_t count = 0;
			for (const Cube& c : C[dim]) {
				if (c.birth < threshold) { ++count; }
			}
			cout << count << " ";
		}
		cout << "--- >= " << threshold << ": ";
		for (uint8_t dim = 0; dim < 4; ++dim) {
			size_t count = 0;
			for (const Cube& c : C[dim]) {
				if (c.birth >= threshold) { ++count; }
			}
			cout << count << " ";
		}
	}
	cout << "                   ";
}


void MorseComplex::printV() const {
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


void MorseComplex::printMorseBoundary(const Cube& c) const {
	vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(c);
	for (const pair<Cube, uint8_t>& b : boundary) {
		get<0>(b).print(); cout << " " << unsigned(get<1>(b)) << endl;
	}
}


void MorseComplex::printFaces() {
	for (uint8_t dim = 0; dim < 4; ++dim) {
		cout << "dim " << unsigned(dim) << ":" << endl;
		for (const Cube& c : C[dim]) {
			cout << "cube: "; c.print(); cout << endl;
			cout << "faces: ";
			for (const Cube& face : faces[c]) { face.print(); cout << " "; }
			cout << endl;
		}
	}
}


void MorseComplex::plotV() const {
	Cube cube;
	for (size_t x = 0; x < 2*shape[0]-1; ++x) {
		for (size_t y = 0; y < 2*shape[1]-1; ++y) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				cube = getCube(x, y, z);
				if (find(C[cube.dim].begin(), C[cube.dim].end(), cube) != C[cube.dim].end()) { cout << "c "; }
				else if (V.count(cube) != 0) { cout << "p "; } 
				else if (coV.count(cube) != 0) { cout << "q "; }
				else { cout << "  "; }
			}
			cout << endl;
		}
		cout << endl;
	}
}


void MorseComplex::plotV(uint8_t dim) const {
	Cube cube;
	unordered_map<Cube, index_t, Cube::Hash> paired;
	index_t counter = 0;
	for (size_t x = 0; x < 2*shape[0]-1; ++x) {
		for (size_t y = 0; y < 2*shape[1]-1; ++y) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				cube = getCube(x, y, z);
				if (cube.dim != dim && cube.dim != dim-1) { 
					cout << "    ";
					continue;
				}
				if (find(C[cube.dim].begin(), C[cube.dim].end(), cube) != C[cube.dim].end() &&
					(cube.dim == dim)) { cout << "CCC "; }
				else if (find(C[cube.dim].begin(), C[cube.dim].end(), cube) != C[cube.dim].end() &&
					(cube.dim == dim-1)) { cout << "ccc "; }
				else {
					auto it = paired.find(cube);
					if (it != paired.end()) {
						if (it->second < 10) { cout << "  " << it->second << " "; }
						else if (it->second < 100) { cout << " " << it->second << " "; }
						else { cout << it->second << " "; }
						continue;
					}
					if (cube.dim == dim-1) {
						auto it = V.find(cube);
						if (it != V.end()) {
							if (counter < 10) { cout << "  " << counter << " "; }
							else if (counter < 100) { cout << " " << counter << " "; }
							else { cout << counter << " "; }
							paired.emplace(it->second, counter);
							++counter;
							continue;
						}
					}
					else if (cube.dim == dim) {
						auto it = coV.find(cube);
						if (it != coV.end()) {
							if (counter < 10) { cout << "  " << counter << " "; }
							else if (counter < 100) { cout << " " << counter << " "; }
							else { cout << counter << " "; }
							paired.emplace(it->second, counter);
							++counter;
							continue;
						}
					}
					cout << "xxx ";
				}
			}
			cout << endl;
		}
		cout << endl;
	}
}


void MorseComplex::plotFlow(const Cube& s) const {
	vector<tuple<Cube, Cube, Cube>> flow;
	traverseFlow(s, flow);
	bool printed;

	for (size_t x = 0; x < 2*shape[0]-1; ++x) {
		for (size_t y = 0; y < 2*shape[1]-1; ++y) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				printed = false;
				Cube c = getCube(x, y, z);
				if(c == s) { cout << "CC "; }
				else {
					for (size_t i = 0; i < flow.size(); ++i) {
						if(c == get<1>(flow[i])) { 
							if(i < 10) { cout << " " << i << " ";}
							else { cout << i << " "; }
							printed = true;
							break;
						} else if(c == get<2>(flow[i])) { 
							if(i < 10) { cout << " " << i << " ";}
							else { cout << i << " "; }
							printed = true;
							break;
						}
					}
					if (!printed) { cout << "xx ";}
				}
			}
			cout << endl;
		}
		cout << endl;
	}
}


void MorseComplex::plotCoFlow(const Cube& s) const {
	vector<tuple<Cube, Cube, Cube>> flow;
	traverseCoflow(s, flow);
	bool printed;

	for (size_t x = 0; x < 2*shape[0]-1; ++x) {
		for (size_t y = 0; y < 2*shape[1]-1; ++y) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				printed = false;
				Cube c = getCube(x, y, z);
				if(c == s) { cout << "CC "; }
				else {
					for (size_t i = 0; i < flow.size(); ++i) {
						if(c == get<1>(flow[i])) { 
							if(i < 10) { cout << " " << i << " ";}
							else { cout << i << " "; }
							printed = true;
							break;
						} else if(c == get<2>(flow[i])) { 
							if(i < 10) { cout << " " << i << " ";}
							else { cout << i << " "; }
							printed = true;
							break;
						}
					}
					if (!printed) { cout << "xx ";}
				}
			}
			cout << endl;
		}
		cout << endl;
	}
}


void MorseComplex::plotConnections(const Cube& s, const Cube& t) const {
	vector<tuple<Cube, Cube, Cube>> connections;
	getConnections(s, t, connections);
	bool printed;

	for (size_t x = 0; x < 2*shape[0]-1; ++x) {
		for (size_t y = 0; y < 2*shape[1]-1; ++y) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				printed = false;
				Cube c = getCube(x, y, z);
				if(c == s) { cout << "CC "; }
				else if(c == t) { cout << "cc "; }
				else {
					for (size_t i = 0; i < connections.size(); ++i) {
						if(c == get<1>(connections[i])) { 
							if(i < 10) { cout << " " << i << " ";}
							else { cout << i << " "; }
							printed = true;
							break;
						} else if(c == get<2>(connections[i])) { 
							if(i < 10) { cout << " " << i << " ";}
							else { cout << i << " "; }
							printed = true;
							break;
						}
					}
					if (!printed) { cout << "xx ";}
				}
			}
			cout << endl;
		}
		cout << endl;
	}
}


void MorseComplex::plotMorseSkeleton() const {
	for (size_t x = 0; x < 2*shape[0]-1; ++x) {
		for (size_t y = 0; y < 2*shape[1]-1; ++y) {
			for (size_t z = 0; z < 2*shape[2]-1; ++z) {
				Cube c = getCube(x, y, z);
				auto it = morseSkeleton.find(c);
				if (it != morseSkeleton.end()) { cout << "S "; }
				else { cout << "x ";}
			}
			cout << endl;
		}
		cout << endl;
	}
}


void MorseComplex::plotMorseSkeletonPixels() const {
	for (index_t x = 0; x < shape[0]; ++x) {
		for (index_t y = 0; y < shape[1]; ++y) {
			for (index_t z = 0; z < shape[2]; ++z) {
				auto it = morseSkeletonPixels.find({x,y,z});
				if (it != morseSkeletonPixels.end()) { cout << "S "; }
				else { cout << "x ";}
			}
			cout << endl;
		}
		cout << endl;
	}
}


void MorseComplex::plotImage() const {
    value_t value;
    for (size_t x = 0; x < shape[0]; ++x) {
		for (size_t y = 0; y < shape[1]; ++y) {
            for (size_t z = 0; z < shape[2]; ++z) {
                value = getValue(x, y, z);
                if (value < 10) { cout << " " << fixed << setprecision(3) << value << " "; }
				else { cout << fixed << setprecision(3) << value << " "; }            
            }
            cout << endl;
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


value_t MorseComplex::getMinimumDistance() {
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
				&& V.count(l) == 0 && coV.count(l) == 0) { 
			++counter;
		}
	}

	return counter;
}


Cube MorseComplex::unpairedFace(const Cube& cube, const vector<Cube>& L) {
	for (const Cube& l : L) { 
		if (l.isFaceOf(cube) && find(C[l.dim].begin(), C[l.dim].end(), l) == C[l.dim].end()
				&& V.count(l) == 0 && coV.count(l) == 0) { 
			return l;
		}
	}
	
	return cube;
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


void MorseComplex::traverseFlow(const Cube& s, vector<tuple<Cube, Cube, Cube>>& flow, bool coordinated) const {
	unordered_map<Cube, size_t, Cube::Hash> nrIn;
	unordered_map<Cube, size_t, Cube::Hash> count;
	if (coordinated) {
		traverseFlow(s, flow, false);
		for (const tuple<Cube, Cube, Cube>& f : flow) {
			auto it = nrIn.find(get<2>(f));
			if (it != nrIn.end()) { ++(it->second); }
			else { nrIn.emplace(get<2>(f), 1); }
		}
		flow.clear();
	}

	BoundaryEnumerator enumerator(*this);
	priority_queue<Cube> queue;
	set<Cube> seen;
	queue.push(s);
	seen.insert(s);

	Cube a;
	Cube b;
	Cube c;
	while (!queue.empty()) {
		a = queue.top(); queue.pop();
		enumerator.setBoundaryEnumerator(a);
		while (enumerator.hasNextFace()) {
			b = enumerator.nextFace;
			c = a;
			if (find(C[b.dim].begin(), C[b.dim].end(), b) != C[b.dim].end()) { c = b; }
			else {
				auto it = V.find(b);
				if (it != V.end()) { c = it->second; }
			}
			if (c != a) {
				flow.push_back(tuple(a, b, c));
				if (c != b) {
					auto it = seen.find(c);
					if (it == seen.end()) {
						if (coordinated) {
							auto itCount = count.find(c);
							auto itNrIn = nrIn.find(c);
							if (itCount != count.end()) { 
								++(itCount->second);
								if (itCount->second != itNrIn->second) { continue; }
							}
							else {
								count.emplace(c, 1);
								if (itNrIn->second != 1) { continue; }
							}
						}
						queue.push(c);
						seen.insert(c);
					}
				}
			}
		}
	}
}


void MorseComplex::traverseCoflow(const Cube& s, vector<tuple<Cube, Cube, Cube>>& flow, bool coordinated) const {
	unordered_map<Cube, size_t, Cube::Hash> nrIn;
	unordered_map<Cube, size_t, Cube::Hash> count;
	if (coordinated) {
		traverseCoflow(s, flow, false);
		for (const tuple<Cube, Cube, Cube>& f : flow) {
			auto it = nrIn.find(get<2>(f));
			if (it != nrIn.end()) { ++(it->second); }
			else { nrIn.emplace(get<2>(f), 1); }
		}
		flow.clear();
	}

	CoboundaryEnumerator enumerator(*this);
	priority_queue<Cube, vector<Cube>, ReverseOrder> queue;
	set<Cube> seen;
	queue.push(s);
	seen.insert(s);

	Cube a;
	Cube b;
	Cube c;
	while (!queue.empty()) {
		a = queue.top(); queue.pop();
		enumerator.setCoboundaryEnumerator(a);
		while (enumerator.hasNextCoface()) {
			b = enumerator.nextCoface;
			c = a;
			if (find(C[b.dim].begin(), C[b.dim].end(), b) != C[b.dim].end()) { c = b; }
			else {
				auto it = coV.find(b);
				if (it != coV.end()) { c = it->second; }
			}
			if (c != a) {
				flow.push_back(tuple(a, b, c));
				if (c != b) {
					auto it = seen.find(c);
					if (it == seen.end()) {
						if (coordinated) {
							auto itCount = count.find(c);
							auto itNrIn = nrIn.find(c);
							if (itCount != count.end()) { 
								++(itCount->second);
								if (itCount->second != itNrIn->second) { continue; }
							}
							else {
								count.emplace(c, 1);
								if (itNrIn->second != 1) { continue; }
							}
						}
						queue.push(c);
						seen.insert(c);
					}
				}
			}
		}
	}
}


void MorseComplex::getConnections(const Cube&s, const Cube& t, vector<tuple<Cube, Cube, Cube>>& connections) const {
	set<Cube> active;
	active.insert(t);

	vector<tuple<Cube, Cube, Cube>> flow;
	traverseCoflow(t, flow);
	for (const tuple<Cube, Cube, Cube>& t : flow) { active.insert(get<2>(t)); }

	flow.clear();
	traverseFlow(s, flow);
	for (const tuple<Cube, Cube, Cube>& t : flow) {
		auto it = active.find(get<1>(t));
		if (it != active.end()) { connections.push_back(t); }
	}
}


void MorseComplex::cancelPair(const Cube&s, const Cube& t) {
	vector<tuple<Cube, Cube, Cube>> connection;
	getConnections(s, t, connection);

	C[s.dim].erase(remove(C[s.dim].begin(), C[s.dim].end(), s), C[s.dim].end());
	C[t.dim].erase(remove(C[t.dim].begin(), C[t.dim].end(), t), C[t.dim].end());

	for (tuple<Cube, Cube, Cube> con : connection) {
		if (get<1>(con) != get<2>(con)) {
			auto it = V.find(get<1>(con));
			if (it != V.end()) { V.erase(it); }
			else { cerr << "Error: key not found!" << endl; exit(EXIT_FAILURE); }
			it = coV.find(get<2>(con));
			if (it != coV.end()) { coV.erase(it); }
			else { cerr << "Error: key not found!" << endl; exit(EXIT_FAILURE);}
		}
		V.emplace(get<1>(con), get<0>(con));
		coV.emplace(get<0>(con), get<1>(con));
	}
}