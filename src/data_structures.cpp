#include "data_structures.h"
#include "enumerators.h"
#include "utils.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include <future>

using namespace std;



Cube::Cube() : birth(0), x(0), y(0), z(0), type(-1), dim(-1) {}


Cube::Cube(const value_t& _birth, const index_t& _x, const index_t& _y, const index_t& _z, 
			const int8_t& _type, const int8_t& _dim) :
	birth(_birth), x(_x), y(_y), z(_z), type(_type), dim(_dim) {}


Cube::Cube(const MorseComplex& mc, const index_t& _x, const index_t& _y, const index_t& _z, 
			const int8_t& _type, const int8_t& _dim) :
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


vector<tuple<index_t,index_t,index_t>> Cube::getVertices() const {
	vector<tuple<index_t,index_t,index_t>> vertices = {make_tuple(x, y, z)};
	switch(dim) {
		case 0:
			return vertices;

		case 1:
			switch(type) {
				case 0:
					vertices.push_back(make_tuple(x+1, y, z));
					return vertices;

				case 1:
					vertices.push_back(make_tuple(x, y+1, z));
					return vertices;

				case 2:
					vertices.push_back(make_tuple(x, y, z+1));
					return vertices;
			}

		case 2:
			switch(type) {
				case 0:
					vertices.push_back(make_tuple(x, y+1, z));
					vertices.push_back(make_tuple(x, y, z+1));
					vertices.push_back(make_tuple(x, y+1, z+1));
					return vertices;

				case 1:
					vertices.push_back(make_tuple(x+1, y, z));
					vertices.push_back(make_tuple(x, y, z+1));
					vertices.push_back(make_tuple(x+1, y, z+1));
					return vertices;

				case 2:
					vertices.push_back(make_tuple(x+1, y, z));
					vertices.push_back(make_tuple(x, y+1, z));
					vertices.push_back(make_tuple(x+1, y+1, z));
					return vertices;
			}

		case 3:
			vertices.push_back(make_tuple(x+1, y, z));
			vertices.push_back(make_tuple(x, y+1, z));
			vertices.push_back(make_tuple(x, y, z+1));
			vertices.push_back(make_tuple(x+1, y+1, z));
			vertices.push_back(make_tuple(x+1, y, z+1));
			vertices.push_back(make_tuple(x, y+1, z+1));
			vertices.push_back(make_tuple(x+1, y+1, z+1));
			return vertices;
	}

	return vertices;
}


bool Cube::isFaceOf(const Cube& other) const {
	if (dim > other.dim-1) { return false; }

	vector<tuple<index_t,index_t,index_t>> vertices = getVertices();
	vector<tuple<index_t,index_t,index_t>> verticesOther = other.getVertices();
	for (tuple<index_t,index_t,index_t>& vertex : vertices) {
		if (find(verticesOther.begin(), verticesOther.end(), vertex) == verticesOther.end()) { 
			return false;
		}
	}

	return true;
}


void Cube::print() const {
	cout << "(" << birth << "," << x << "," << y << "," << z << "," << static_cast<int16_t>(type) << ","
		<< static_cast<int16_t>(dim) << ")";
}



MorseComplex::MorseComplex(const vector<value_t>&& image, const vector<index_t>&& _shape) : 
	shape(_shape), mYZ(shape[1]*shape[2]), C(4), perturbed(false), processedLowerStars(false)
#ifdef USE_CUBE_MAP
	, V(shape), coV(shape)
#endif
	{ getGridFromVector(image); }


MorseComplex::MorseComplex(const vector<value_t>& image, const vector<index_t>& _shape) :
	shape(_shape), mYZ(shape[1]*shape[2]), C(4), perturbed(false), processedLowerStars(false) 
#ifdef USE_CUBE_MAP
	, V(shape), coV(shape)
#endif
	{ getGridFromVector(image); }


MorseComplex::MorseComplex(MorseComplex&& other) : shape(other.shape), mYZ(other.mYZ), C(4),
	perturbed(other.perturbed), processedLowerStars(other.processedLowerStars)
#ifdef USE_CUBE_MAP
	, V(other.V), coV(other.coV)
#endif
	{
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


void MorseComplex::perturbImage(const value_t& epsilon) {
	if (perturbed) { 
		cout << "Image is already perturbed!" << endl;
		return;
	}

	if (epsilon == 0) { perturbation = getMinimumDistance(); }
	else { perturbation = epsilon; }

	if (perturbation != 0) {
		value_t denom = 3*shape[0]*shape[1]*shape[2];
		for (size_t x = 0; x < shape[0]; ++x) {
			for (size_t y = 0; y < shape[1]; ++y) {
				for (size_t z = 0; z < shape[2]; ++z) {
					addValue(x, y, z, perturbation*(x*mYZ + y*shape[2] + z) / denom);          
				}
			}
		}
	}

	perturbed = true;
}


void MorseComplex::perturbImageMinimal() {
	if (perturbed) { 
		cout << "Image is already perturbed!" << endl;
		return;
	}
	
	perturbation = numeric_limits<value_t>::min();

	for (size_t x = 0; x < shape[0]; ++x) {
		for (size_t y = 0; y < shape[1]; ++y) {
			for (size_t z = 0; z < shape[2]; ++z) {
				addValue(x, y, z, perturbation*(x*mYZ + y*shape[2] + z));          
			}
		}
	}

	perturbation *= 3*shape[0]*shape[1]*shape[2];
	perturbed = true;
}


void MorseComplex::processLowerStars(const value_t& threshold, const index_t& xPatch, const index_t& yPatch, const index_t& zPatch) {
	if (!perturbed) {
		if (xPatch == 1 && yPatch == 1 && zPatch == 1) { processLowerStarsWithoutPerturbation(threshold); }
		else { processLowerStarsWithoutPerturbationParallel(xPatch, yPatch, zPatch, threshold); }	
	} else {
		if (xPatch == 1 && yPatch == 1 && zPatch == 1) { processLowerStarsWithPerturbation(threshold); }
		else { processLowerStarsWithPerturbationParallel(xPatch, yPatch, zPatch, threshold); }
	}
}


void MorseComplex::cancelPairByIndex(const uint8_t& dimS, const size_t& indexS, const uint8_t& dimT, const size_t& indexT) {
	Cube& s = C[dimS][indexS];
	Cube& t = C[dimT][indexT];
	vector<tuple<Cube, Cube, Cube>> connection;
	getConnections(s, t, connection);

	C[s.dim].erase(remove(C[s.dim].begin(), C[s.dim].end(), s), C[s.dim].end());
	C[t.dim].erase(remove(C[t.dim].begin(), C[t.dim].end(), t), C[t.dim].end());

	for (tuple<Cube, Cube, Cube> con : connection) {
		if (get<1>(con) != get<2>(con)) {
#ifdef USE_CUBE_MAP
			V.erase(get<1>(con));
			coV.erase(get<2>(con));
#else
			auto it = V.find(get<1>(con));
			if (it != V.end()) { V.erase(it); }
			else { cerr << "Error: key not found!" << endl; exit(EXIT_FAILURE); }
			it = coV.find(get<2>(con));
			if (it != coV.end()) { coV.erase(it); }
			else { cerr << "Error: key not found!" << endl; exit(EXIT_FAILURE);}
#endif
		}
		V.emplace(get<1>(con), get<0>(con));
		coV.emplace(get<0>(con), get<1>(con));
	}
}


void MorseComplex::cancelPairsBelow(const value_t& threshold, string orderDim, string orderValue, bool print) {
	if (print) { 
		cout << endl << "Canceling pairs < " << threshold << endl;
		cout << "Canceling order:" << endl;
        cout << "dimension: " << orderDim << endl;
        cout << "value: " << orderValue << endl;
		cout << endl;
	
		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == -INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	for (uint8_t dim = 0; dim < 4; ++dim) { sort(C[dim].begin(), C[dim].end()); }

	vector<uint8_t> dimensions;
	if (orderDim == ">") { dimensions = {3,2,1}; }
	else if (orderDim == "<") { dimensions = {1,2,3}; }
	else if (orderDim == "><") { dimensions = {3,1,2}; }
	else if (orderDim == "<>") { dimensions = {2,3,1}; }

	bool canceled = true;
	vector<Cube> cancelable;

	if (orderValue == ">") {
		while (canceled) {
			canceled = false;
			for (uint8_t dim : dimensions) {
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

					if (print) { printNumberOfCriticalCells(threshold); }
					break;
				}
				if (canceled) { break; }
			}
		}
	} 
	else if (orderValue == "<") {
		while (canceled) {
			canceled = false;
			for (uint8_t dim : dimensions) {
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
					canceled = true;

					if (print) { printNumberOfCriticalCells(threshold); }
					break;
				}
				if (canceled) { break; }
			}
		}
	}
	if (print) { cout << endl << endl; }
}


void MorseComplex::cancelPairsBelowInDim(const uint8_t& dim, const value_t& threshold, string orderValue, bool print) {
	if (print) { 
		cout << endl << "Canceling pairs < " << threshold << endl;
		cout << "Canceling order:" << endl;
        cout << "dimension: " << unsigned(dim) << endl;
        cout << "value: " << orderValue << endl;
		cout << endl;
	
		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == -INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	sort(C[dim].begin(), C[dim].end());

	bool canceled = true;
	vector<Cube> cancelable;

	if (orderValue == ">") {
		while (canceled) {
			canceled = false;
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

				if (print) { printNumberOfCriticalCells(threshold); }
				break;
			}
		}
	} 
	else if (orderValue == "<") {
		while (canceled) {
			canceled = false;
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
				canceled = true;

				if (print) { printNumberOfCriticalCells(threshold); }
				break;
			}
		}
	}
	if (print) { cout << endl << endl; }
}


void MorseComplex::cancelPairsAbove(const value_t& threshold, string orderDim, string orderValue, bool print) {
	if (print) {
		cout << endl << "Canceling pairs >= " << threshold << endl;
		cout << "Canceling order:" << endl;
        cout << "dimension: " << orderDim << endl;
        cout << "value: " << orderValue << endl;
		cout << endl;

		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	for (uint8_t dim = 0; dim < 4; ++dim) { sort(C[dim].begin(), C[dim].end()); }

	vector<uint8_t> dimensions;
	if (orderDim == ">") { dimensions = {2,1,0}; }
	else if (orderDim == "<") { dimensions = {0,1,2}; }
	else if (orderDim == "><") { dimensions = {3,1,2}; }
	else if (orderDim == "<>") { dimensions = {2,3,1}; }

	bool canceled = true;
	vector<Cube> cancelable;

	if (orderValue == ">") {
		while (canceled) {
			canceled = false;
			for (uint8_t dim : dimensions) {
				for (auto it = C[dim].rbegin(); it != C[dim].rend(); ++it) {
					Cube t = *it;
					
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

					if (print) { printNumberOfCriticalCells(threshold); }
					break;
				}
				if (canceled) { break; }
			}
		}
		if (print) { cout << endl << endl; }
	}
	else if (orderValue == "<") {
		while (canceled) {
			canceled = false;
			for (uint8_t dim : dimensions) {
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

					if (print) { printNumberOfCriticalCells(threshold); }
					break;
				}
				if (canceled) { break; }
			}
		}
		if (print) { cout << endl << endl; }
	}	
}


void MorseComplex::cancelLowPersistencePairsBelow(const value_t& threshold, const value_t& epsilon, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == -INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	vector<Cube> cancelable;
	for (uint8_t dim = 4; dim-- > 1;) {
		sort(C[dim].begin(), C[dim].end());

		bool canceled = true;	
		while (canceled) {
			canceled = false;
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
				Cube t = cancelable.back();
				if (s.birth - t.birth > epsilon) { continue; }

				cancelPair(s, t);
				canceled = true;

				if (print) { printNumberOfCriticalCells(threshold); }
				break;
			}
		}
	}
	if (print) { cout << endl; }
}


void MorseComplex::cancelLowPersistencePairsInDimBelow(const uint8_t& dim, const value_t& threshold, const value_t& epsilon, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == -INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	vector<Cube> cancelable;
	
	sort(C[dim].begin(), C[dim].end());

	bool canceled = true;	
	while (canceled) {
		canceled = false;
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
			Cube t = cancelable.back();
			if (s.birth - t.birth > epsilon) { continue; }

			cancelPair(s, t);
			canceled = true;

			if (print) { printNumberOfCriticalCells(threshold); }
			break;
		}
	}

	if (print) { cout << endl; }
}


void MorseComplex::cancelClosePairsBelow(const value_t& threshold, const value_t& epsilon, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == -INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	vector<Cube> cancelable;
	for (uint8_t dim = 4; dim-- > 1;) {
		sort(C[dim].begin(), C[dim].end());

		bool canceled = true;	
		while (canceled) {
			canceled = false;
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
				Cube t = cancelable.back();
				if (s.birth - t.birth > epsilon) { continue; }

				while (true) {
					vector<pair<Cube, uint8_t>> coboundary = getMorseCoboundary(t);
					cancelable.clear();
					for (const pair<Cube, uint8_t> c : coboundary) {
						if (get<1>(c) == 1) { cancelable.push_back(get<0>(c)); }
					}
					sort(cancelable.begin(), cancelable.end());

					if (cancelable.front() < s) { 
						s = cancelable.front();

						vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(s);
						cancelable.clear();
						for (const pair<Cube, uint8_t> b : boundary) {
							if (get<1>(b) == 1) { cancelable.push_back(get<0>(b)); }
						}

						sort(cancelable.begin(), cancelable.end());

						if (t < cancelable.back()) { t = cancelable.back(); }
						else { break; }
						
					}
					else { break; }
				}
				
				cancelPair(s, t);
				canceled = true;

				if (print) { printNumberOfCriticalCells(threshold); }
				break;
			}
		}
	}
	if (print) { cout << endl; }
}


void MorseComplex::cancelBoundaryPairsBelow(const value_t& threshold, const value_t& delta, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == -INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	vector<Cube> cancelable;
	for (uint8_t dim = 4; dim-- > 1;) {
		sort(C[dim].begin(), C[dim].end());

		bool canceled = true;
		while (canceled) {
			canceled = false;
			for (auto it = C[dim].rbegin(); it != C[dim].rend(); ++it) {
				Cube s = *it;
				if (s.birth >= threshold || s.birth < threshold - delta) { continue; }

				vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(s);
				cancelable.clear();
				for (const pair<Cube, uint8_t> b : boundary) {
					if (get<1>(b) == 1) { cancelable.push_back(get<0>(b)); }
				}
				if (cancelable.size() == 0) { continue; }

				sort(cancelable.begin(), cancelable.end());
				Cube t = cancelable.back();
				if (t.birth < threshold - delta) { continue; }

				cancelPair(s, t);
				canceled = true;

				if (print) { printNumberOfCriticalCells(threshold); }
				break;
			}
		}
	}
	if (print) { cout << endl; }
}


void MorseComplex::prepareMorseSkeletonTestBelow(const value_t& threshold, const value_t& epsilon, const value_t& delta, 
													bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == -INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	vector<Cube> cancelable;
	for (uint8_t dim = 4; dim-- > 1;) {
		sort(C[dim].begin(), C[dim].end());

		bool canceled = true;
		while (canceled) {
			canceled = false;
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
				Cube t = cancelable.back();
				if (t.birth < threshold - delta && s.birth-t.birth > epsilon) { continue; }

				cancelPair(s, t);
				canceled = true;

				if (print) { printNumberOfCriticalCells(threshold); }
				break;
			}
		}
	}
	if (print) { cout << endl; }
}


void MorseComplex::prepareMorseSkeletonBelow(const value_t& threshold, const value_t& epsilon, const value_t& delta, bool print) {
	if (epsilon >= 0) {
		if (print) { cout << "Canceling Pairs below " << threshold << " of persistence < " << epsilon << ":" << endl; }
		cancelLowPersistencePairsBelow(threshold, epsilon, print);
	}

	if (delta >= 0) {
		if (print) { cout << endl << "Canceling Pairs below " << threshold << " with distance < " << delta << " to the boundary:" << endl; }
		cancelBoundaryPairsBelow(threshold, delta, print);
	}
	
}


void MorseComplex::prepareMorseSkeletonAbove(const value_t& threshold, const value_t& tolerance, bool print) {
	if (print) {
		cout << "Critical cells:" << endl;
		printNumberOfCriticalCells(threshold);
	}

	if (threshold == INFTY) {
		if (print) { cout << endl << endl; }
		return;
	}

	for (uint8_t dim = 0; dim < 4; ++dim) { sort(C[dim].begin(), C[dim].end()); }

	bool canceled = true;
	vector<Cube> cancelable;
	while (canceled) {
		canceled = false;
		for (uint8_t dim = 0; dim < 2; ++dim) {
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

				if (print) { printNumberOfCriticalCells(threshold); }
				break;
			}
			if (canceled) { break; }
		}

		if (tolerance != 0) {
			for (const Cube& s : C[3]) {
				if (s.birth < threshold || s.birth >= threshold + tolerance) { continue; }

				vector<pair<Cube, uint8_t>> boundary = getMorseBoundary(s);

				cancelable.clear();
				for (const pair<Cube, uint8_t> b : boundary) {
					if (get<1>(b) == 1) { cancelable.push_back(get<0>(b)); }
				}
				if (cancelable.size() == 0) { continue; }
				sort(cancelable.begin(), cancelable.end());
				if (cancelable.back().birth < threshold) { continue; }

				cancelPair(s, cancelable.back());
				canceled = true;

				if (print) { printNumberOfCriticalCells(threshold); }
				break;
			}
		}
	}
	if (print) { cout << endl << endl; }
}


void MorseComplex::extractMorseSkeletonBelow(const value_t& threshold, const uint8_t& dimension) {
	morseSkeletonBelow.clear();
	morseSkeletonVoxelsBelow.clear();
	
	vector<tuple<Cube, Cube, Cube>> flow;
	for (const Cube& c : C[0]) {
		if (c.birth >= threshold) { continue; }

		morseSkeletonBelow.insert(c);
	}
	for (uint8_t dim = 1; dim < dimension+1; ++dim) {
		for (const Cube& c : C[dim]) {
			if (c.birth >= threshold) { continue; }

			morseSkeletonBelow.insert(c);
			flow.clear();
			traverseFlow(c, flow);
			for (const tuple<Cube, Cube, Cube>& t : flow) {
				if (get<1>(t) != get<2>(t)) { morseSkeletonBelow.insert(get<2>(t)); }
			}
		}
	}

	vector<tuple<index_t,index_t,index_t>> voxels;
	for (const Cube& c : morseSkeletonBelow) {
		voxels = c.getVertices();
		for (const tuple<index_t,index_t,index_t>& voxel : voxels) {
			morseSkeletonVoxelsBelow.push_back(voxel);
		}

	}
}


void MorseComplex::extractMorseSkeletonParallelBelow(const value_t& threshold, const uint8_t& dimension) {
	morseSkeletonBelow.clear();
	morseSkeletonVoxelsBelow.clear();

	vector<std::future<vector<tuple<index_t,index_t,index_t>>>> futures;
	for (uint8_t dim = 0; dim < dimension+1; ++dim) {
		futures.push_back(async(launch::async, &MorseComplex::extractMorseSkeletonInDimBelow,
								this, dim, threshold));
	}

	for (auto& future : futures) {
		vector<tuple<index_t,index_t,index_t>> skeleton = future.get();
		vector<tuple<index_t,index_t,index_t>>::iterator itr;
   
		for (itr = skeleton.begin(); itr != skeleton.end(); ++itr) {
			morseSkeletonVoxelsBelow.push_back(*itr);
		}
	}
}


void MorseComplex::extractMorseSkeletonBatchwiseBelow(const value_t& threshold, const uint8_t& dimension, const size_t& batches) {
	morseSkeletonBelow.clear();
	morseSkeletonVoxelsBelow.clear();

	vector<std::future<vector<tuple<index_t,index_t,index_t>>>> futures;
	size_t start;
	size_t end;
	for (uint8_t dim = 0; dim < dimension+1; ++dim) {
		size_t batchSize = C[dim].size()/batches + 1;
		start = 0;
		end = batchSize;
		for (size_t batch = 0; batch < batches; ++batch) {
			futures.push_back(async(launch::async, &MorseComplex::extractMorseSkeletonOfBatchInDimBelow,
								this, dim, start, end, threshold));
			start = end;
			end += batchSize;
		}
	}

	for (auto& future : futures) {
		vector<tuple<index_t,index_t,index_t>> skeleton = future.get();
		vector<tuple<index_t,index_t,index_t>>::iterator itr;
   
		for (itr = skeleton.begin(); itr != skeleton.end(); ++itr) {
			morseSkeletonVoxelsBelow.push_back(*itr);
		}
	}
}


void MorseComplex::prepareAndExtractMorseSkeletonBelow(const value_t& threshold, const value_t& epsilon, const vector<uint8_t>& dimensions) {
	morseSkeletonBelow.clear();
	morseSkeletonVoxelsBelow.clear();

	vector<std::future<void>> futures;
	if(find(dimensions.begin(), dimensions.end(), 3) != dimensions.end()) {
		futures.push_back(async(launch::async, &MorseComplex::cancelLowPersistencePairsInDimBelow,
								this, 3, threshold, epsilon, false));
	}
	if(find(dimensions.begin(), dimensions.end(), 1) != dimensions.end()) {
		futures.push_back(async(launch::async, &MorseComplex::cancelLowPersistencePairsInDimBelow,
								this, 1, threshold, epsilon, false));
	}
	for (auto& future : futures) {
        future.wait();
    }

	futures.clear();
	if(find(dimensions.begin(), dimensions.end(), 2) != dimensions.end()) {
		futures.push_back(async(launch::async, &MorseComplex::cancelLowPersistencePairsInDimBelow,
								this, 2, threshold, epsilon, false));
	}
	//futures.push_back(async(launch::async, &MorseComplex::extractMorseSkeletonInDimBelow,
	//							this, 3, threshold));
	//futures.push_back(async(launch::async, &MorseComplex::extractMorseSkeletonInDimBelow,
	//							this, 0, threshold));
	if(find(dimensions.begin(), dimensions.end(), 2) != dimensions.end()) {
		futures[0].wait();
	}

	//futures.push_back(async(launch::async, &MorseComplex::extractMorseSkeletonInDimBelow,
	//							this, 1, threshold));
	//futures.push_back(async(launch::async, &MorseComplex::extractMorseSkeletonInDimBelow,
	//							this, 2, threshold));
	for (auto& future : futures) {
        future.wait();
    }

	vector<tuple<index_t,index_t,index_t>> voxels;
	for (const Cube& c : morseSkeletonBelow) {
		voxels = c.getVertices();
		for (const tuple<index_t,index_t,index_t>& voxel : voxels) {
			morseSkeletonVoxelsBelow.push_back(voxel);
		}
	}
}


void MorseComplex::extractMorseSkeletonAbove(const value_t& threshold) {
	morseSkeletonAbove.clear();
	morseSkeletonVoxelsAbove.clear();
	
	vector<tuple<Cube, Cube, Cube>> flow;
	for (uint8_t dim = 0; dim < 4; ++dim) {
		for (const Cube& c : C[dim]) {
			if (c.birth < threshold) { continue; }
			morseSkeletonAbove.insert(c);
			flow.clear();
			traverseCoflow(c, flow);
			for (const tuple<Cube, Cube, Cube>& t : flow) {
				if (get<1>(t) != get<2>(t)) { morseSkeletonAbove.insert(get<2>(t)); }
			}
		}
	}

	vector<tuple<index_t,index_t,index_t>> voxels;
	for (const Cube& c : morseSkeletonAbove) {
		voxels = c.getVertices();
		for (const tuple<index_t,index_t,index_t>& voxel : voxels) {
			morseSkeletonVoxelsAbove.push_back(voxel);
		}

	}
}


vector<pair<Cube, uint8_t>> MorseComplex::getMorseBoundary(const Cube& s) const {
#ifdef USE_CUBE_MAP
	CubeMap<uint8_t> count(shape);
	count.emplace(s, 1);
#else
	unordered_map<Cube, uint8_t, Cube::Hash> count;
	count.emplace(s, 1);
#endif

	vector<tuple<Cube, Cube,Cube>> flow;
	traverseFlow(s, flow);

	set<Cube> boundary;
	uint8_t n;
	for (const tuple<Cube, Cube, Cube>& f : flow) {
#ifdef USE_CUBE_MAP
		n = count.getValue(get<0>(f)) + count.getValue(get<2>(f));
		if (n > 3) { count.emplace(get<2>(f), n%2 + 2); }
		else { count.emplace(get<2>(f), n); }
#else
		auto it = count.find(get<2>(f));
		if (it != count.end()) { n = count[get<0>(f)] + it->second; }
		else { n = count[get<0>(f)]; }
		if (n > 3) { count.insert_or_assign(get<2>(f), n%2 + 2); }
		else { count.insert_or_assign(get<2>(f), n); }
#endif
		if (get<1>(f) == get<2>(f)) { boundary.insert(get<2>(f)); }
	}

	vector<pair<Cube, uint8_t>> result;
	for (const Cube& b : boundary) {
#ifdef USE_CUBE_MAP
		result.push_back(pair(b, count.getValue(b)));
#else
		result.push_back(pair(b, count[b]));
#endif
	}

	return result;
}


vector<pair<Cube, uint8_t>> MorseComplex::getMorseCoboundary(const Cube& s) const {
#ifdef USE_CUBE_MAP
	CubeMap<uint8_t> count(shape);
	count.emplace(s, 1);
#else
	unordered_map<Cube, uint8_t, Cube::Hash> count;
	count.emplace(s, 1);
#endif
	vector<tuple<Cube, Cube,Cube>> flow;
	traverseCoflow(s, flow);

	set<Cube> coboundary;
	uint8_t n;
	for (const tuple<Cube, Cube, Cube>& f : flow) {
#ifdef USE_CUBE_MAP
		n = count.getValue(get<0>(f)) + count.getValue(get<2>(f));
		if (n > 3) { count.emplace(get<2>(f), n%2 + 2); }
		else { count.emplace(get<2>(f), n); }
#else
		auto it = count.find(get<2>(f));
		if (it != count.end()) { n = count[get<0>(f)] + it->second; }
		else { n = count[get<0>(f)]; }
		if (n > 3) { count.insert_or_assign(get<2>(f), n%2 + 2); }
		else { count.insert_or_assign(get<2>(f), n); }
#endif
		if (get<1>(f) == get<2>(f)) { coboundary.insert(get<2>(f)); }
	}

	vector<pair<Cube, uint8_t>> result;
	for (const Cube& c : coboundary) {
#ifdef USE_CUBE_MAP
		result.push_back(pair(c, count.getValue(c)));
#else
		result.push_back(pair(c, count[c]));
#endif
	}

	return result;
}


vector<tuple<index_t,index_t,index_t>> MorseComplex::getMorseSkeletonVoxelsBelow() const { 
	return vector<tuple<index_t,index_t,index_t>>(morseSkeletonVoxelsBelow.begin(), morseSkeletonVoxelsBelow.end());
}


vector<tuple<index_t,index_t,index_t>> MorseComplex::getMorseSkeletonVoxelsAbove() const { 
	return vector<tuple<index_t,index_t,index_t>>(morseSkeletonVoxelsAbove.begin(), morseSkeletonVoxelsAbove.end());
}


value_t MorseComplex::getPerturbation() const { return perturbation; }


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
#ifdef USE_CUBE_MAP
						if (V.getValue(cube) != Cube()) { ++counter; }
						if (coV.getValue(cube) != Cube()) { ++counter; }
#else
						if (V.count(cube) != 0) { ++counter; }
						if (coV.count(cube) != 0) { ++counter; }
#endif
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


void MorseComplex::printNumberOfCriticalCells(const value_t& threshold) const {
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


size_t MorseComplex::hashVoxel(const vector<index_t>& voxelCoordinates) const {
	return voxelCoordinates[0]*mYZ + voxelCoordinates[1]*shape[2] + voxelCoordinates[2];
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


vector<index_t> MorseComplex::getParentVoxel(const Cube& cube) const {
	index_t x = cube.x;
	index_t y = cube.y;
	index_t z = cube.z;
	switch (cube.dim) {
		case 0:
			return {x,y,z};

		case 1:
		switch (cube.type) {
			case 0:
				if (cube.birth == getValue(x+1, y, z)) { return {x+1,y,z}; }
				else { return {x,y,z}; }
				break;

			case 1:
				if (cube.birth == getValue(x, y+1, z)) { return {x,y+1,z}; }
				else { return {x,y,z}; }
				break;

			case 2:
				if (cube.birth == getValue(x, y, z+1)) { return {x,y,z+1}; }
				else { return {x,y,z}; }
				break;
		}

		case 2:
		switch(cube.type) {
			case 0:
				if (cube.birth == getValue(x, y+1, z+1)) { return {x,y+1,z+1}; }
				else if (cube.birth == getValue(x, y+1, z)) { return {x,y+1,z}; }
				else if (cube.birth == getValue(x, y, z+1)) { return {x,y,z+1}; }
				else { return {x,y,z}; }
				break;

			case 1:
				if (cube.birth == getValue(x+1, y, z+1)) { return {x+1,y,z+1}; }
				else if (cube.birth == getValue(x+1, y, z)) { return {x+1,y,z}; }
				else if (cube.birth == getValue(x, y, z+1)) { return {x,y,z+1}; }
				else { return {x,y,z}; }
				break;

			case 2:
				if (cube.birth == getValue(x+1, y+1, z)) { return {x+1,y+1,z}; } 
				else if (cube.birth == getValue(x+1, y, z)) { return {x+1,y,z}; }
				else if (cube.birth == getValue(x, y+1, z)) { return {x,y+1,z}; } 
				else { return {x,y,z}; }
				break;
		}
		
		case 3:
			if (cube.birth == getValue(x+1, y+1, z+1)) { return {x+1,y+1,z+1}; }
			else if (cube.birth == getValue(x+1, y+1, z)) { return {x+1,y+1,z}; }
			else if (cube.birth == getValue(x+1, y, z+1)) { return {x+1,y,z+1}; }
			else if (cube.birth == getValue(x+1, y, z)) { return {x+1,y,z}; }
			else if (cube.birth == getValue(x, y+1, z+1)) { return {x,y+1,z+1}; }
			else if (cube.birth == getValue(x, y+1, z)) { return {x,y+1,z}; }
			else if (cube.birth == getValue(x, y, z+1)) { return {x,y,z+1}; } 
			else { return {x,y,z}; }
			break;
	}
	cerr << "parent voxel not found!" << endl;
	return {0,0,0};

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


vector<Cube> MorseComplex::getLowerStarWithoutPerturbation(const index_t& x, const index_t& y, const index_t& z) const {
	vector<Cube> L;
	value_t value = getValue(x, y, z);

	value_t valueX;
	value_t valueY;
	value_t valueZ;
	value_t valueXY;
	value_t valueXZ;
	value_t valueYZ;
	value_t valueXYZ;

	valueX = getValue(x-1, y, z);
	valueY = getValue(x, y-1, z);
	valueZ = getValue(x, y, z-1);
	valueXY = getValue(x-1, y-1, z);
	valueXZ = getValue(x-1, y, z-1);
	valueYZ = getValue(x, y-1, z-1);
	valueXYZ = getValue(x-1, y-1, z-1);
	if (valueX <= value) { L.push_back(Cube(value, x-1, y, z, 0, 1)); }
	if (valueY <= value) { L.push_back(Cube(value, x, y-1, z, 1, 1)); }
	if (valueZ <= value) { L.push_back(Cube(value, x, y, z-1, 2, 1)); }
	if (valueX <= value && valueY <= value && valueXY <= value) { L.push_back(Cube(value, x-1, y-1, z, 2, 2)); }
	if (valueX <= value && valueZ <= value && valueXZ <= value) { L.push_back(Cube(value, x-1, y, z-1, 1, 2)); }
	if (valueY <= value && valueZ <= value && valueYZ <= value) { L.push_back(Cube(value, x, y-1, z-1, 0, 2)); }
	if (valueX <= value && valueY <= value && valueZ <= value && valueXY <= value 
		&& valueXZ <= value && valueYZ <= value && valueXYZ <= value) { L.push_back(Cube(value, x-1, y-1, z-1, 0, 3)); }

	valueX = getValue(x+1, y, z);
	valueXY = getValue(x+1, y-1, z);
	valueXZ = getValue(x+1, y, z-1);
	valueXYZ = getValue(x+1, y-1, z-1);
	if (valueX < value) { L.push_back(Cube(value, x, y, z, 0, 1)); }
	if (valueX < value && valueY <= value && valueXY < value) { L.push_back(Cube(value, x, y-1, z, 2, 2)); }
	if (valueX < value && valueZ <= value && valueXZ < value) { L.push_back(Cube(value, x, y, z-1, 1, 2)); }
	if (valueX < value && valueY <= value && valueZ <= value && valueXY < value 
		&& valueXZ < value && valueYZ <= value && valueXYZ < value) { L.push_back(Cube(value, x, y-1, z-1, 0, 3)); }
	
	valueY = getValue(x, y+1, z);
	valueXY = getValue(x+1, y+1, z);
	valueYZ = getValue(x, y+1, z-1);
	valueXYZ = getValue(x+1, y+1, z-1);
	if (valueY < value) { L.push_back(Cube(value, x, y, z, 1, 1)); }
	if (valueX < value && valueY < value && valueXY < value) { L.push_back(Cube(value, x, y, z, 2, 2)); }
	if (valueY < value && valueZ <= value && valueYZ < value) { L.push_back(Cube(value, x, y, z-1, 0, 2)); }
	if (valueX < value && valueY < value && valueZ <= value && valueXY < value 
		&& valueXZ < value && valueYZ < value && valueXYZ < value) { L.push_back(Cube(value, x, y, z-1, 0, 3)); }
	
	valueZ = getValue(x, y, z+1);
	valueXZ = getValue(x+1, y, z+1);
	valueYZ = getValue(x, y+1, z+1);
	valueXYZ = getValue(x+1, y+1, z+1);
	if (valueZ < value) { L.push_back(Cube(value, x, y, z, 2, 1)); }
	if (valueX < value && valueZ < value && valueXZ < value) { L.push_back(Cube(value, x, y, z, 1, 2)); }
	if (valueY < value && valueZ < value && valueYZ < value) { L.push_back(Cube(value, x, y, z, 0, 2)); }
	if (valueX < value && valueY < value && valueZ < value && valueXY < value 
		&& valueXZ < value && valueYZ < value && valueXYZ < value) { L.push_back(Cube(value, x, y, z, 0, 3)); }

	valueY = getValue(x, y-1, z);
	valueYZ = getValue(x, y-1, z+1);
	valueXYZ = getValue(x+1, y-1, z+1);
	if (valueY <= value && valueZ < value && valueYZ <= value) { L.push_back(Cube(value, x, y-1, z, 0, 2)); }
	if (valueX < value && valueY <= value && valueZ < value && valueXY < value 
		&& valueXZ < value && valueYZ <= value && valueXYZ < value) { L.push_back(Cube(value, x, y-1, z, 0, 3)); }

	valueX = getValue(x-1, y, z);
	valueXY = getValue(x-1, y-1, z);
	valueXZ = getValue(x-1, y, z+1);
	valueXYZ = getValue(x-1, y-1, z+1);
	if (valueX <= value && valueZ < value && valueXZ <= value) { L.push_back(Cube(value, x-1, y, z, 1, 2)); }
	if (valueX <= value && valueY <= value && valueZ < value && valueXY <= value 
		&& valueXZ <= value && valueYZ <= value && valueXYZ <= value) { L.push_back(Cube(value, x-1, y-1, z, 0, 3)); }

	valueY = getValue(x, y+1, z);
	valueXY = getValue(x-1, y+1, z);
	valueYZ = getValue(x, y+1, z+1);
	valueXYZ = getValue(x-1, y+1, z+1);
	if (valueX <= value && valueY < value && valueXY <= value) { L.push_back(Cube(value, x-1, y, z, 2, 2)); }
	if (valueX <= value && valueY < value && valueZ < value && valueXY <= value 
		&& valueXZ <= value && valueYZ < value && valueXYZ <= value) { L.push_back(Cube(value, x-1, y, z, 0, 3)); }
	
	valueZ = getValue(x, y, z-1);
	valueXZ = getValue(x-1, y, z-1);
	valueYZ = getValue(x, y+1, z-1);
	valueXYZ = getValue(x-1, y+1, z-1);
	if (valueX <= value && valueY < value && valueZ <= value && valueXY <= value 
		&& valueXZ <= value && valueYZ < value && valueXYZ <= value) { L.push_back(Cube(value, x-1, y, z-1, 0, 3)); }

	return L;
}


void MorseComplex::getLowerStarsWithoutPerturbation(vector<vector<Cube>>& lowerStars, const value_t& threshold) const {
	lowerStars = vector<vector<Cube>>(shape[0]*shape[1]*shape[2]);
	value_t birth;
	Cube cube;
	for (index_t x = 0; x < shape[0]; ++x) {
		for (index_t y = 0; y < shape[1]; ++y) {
			for (index_t z = 0; z < shape[2]; ++z) {
				birth = getBirth(x, y, z, 0, 3);
				if (birth != INFTY && birth <= threshold) {
					cube = Cube(birth, x, y, z, 0, 3);
					lowerStars[hashVoxel(getParentVoxel(cube))].push_back(cube);
				}
				for (uint8_t type = 0; type < 3; ++type) {
					birth = getBirth(x, y, z, type, 1);
					if (birth != INFTY && birth <= threshold) {
						cube = Cube(birth, x, y, z, type, 1);
						lowerStars[hashVoxel(getParentVoxel(cube))].push_back(cube);
					}
					birth = getBirth(x, y, z, type, 2);
					if (birth != INFTY && birth <= threshold) {
						cube = Cube(birth, x, y, z, type, 2);
						lowerStars[hashVoxel(getParentVoxel(cube))].push_back(cube);
					}
				}
			}
		}
	}
}


void MorseComplex::getLowerStarsWithoutPerturbationBetween(vector<vector<Cube>>& lowerStars, 
															const index_t& xMin, const index_t& xMax,
															const index_t& yMin, const index_t& yMax,
															const index_t& zMin, const index_t& zMax,
															const value_t& threshold) const {
	index_t xL = (xMin == 0) ? 0 : xMin-1;
	index_t yL = (yMin == 0) ? 0 : yMin-1;
	index_t zL = (zMin == 0) ? 0 : zMin-1;
	index_t xU = (xMax == shape[0]) ? shape[0] : xMax+1;
	index_t yU = (yMax == shape[1]) ? shape[1] : yMax+1;
	index_t zU = (zMax == shape[2]) ? shape[2] : zMax+1;

	lowerStars = vector<vector<Cube>>(shape[0]*shape[1]*shape[2]);
	value_t birth;
	Cube cube;
	for (index_t x = xL; x < xU; ++x) {
		for (index_t y = yL; y < yU; ++y) {
			for (index_t z = zL; z < zU; ++z) {
				birth = getBirth(x, y, z, 0, 3);
				if (birth != INFTY && birth <= threshold) {
					cube = Cube(birth, x, y, z, 0, 3);
					lowerStars[hashVoxel(getParentVoxel(cube))].push_back(cube);
				}
				for (uint8_t type = 0; type < 3; ++type) {
					birth = getBirth(x, y, z, type, 1);
					if (birth != INFTY && birth <= threshold) {
						cube = Cube(birth, x, y, z, type, 1);
						lowerStars[hashVoxel(getParentVoxel(cube))].push_back(cube);
					}
					birth = getBirth(x, y, z, type, 2);
					if (birth != INFTY && birth <= threshold) {
						cube = Cube(birth, x, y, z, type, 2);
						lowerStars[hashVoxel(getParentVoxel(cube))].push_back(cube);
					}
				}
			}
		}
	}
}


size_t MorseComplex::numUnpairedFaces(const Cube& cube, const vector<Cube>& L) const {
	size_t counter = 0;
	for (const Cube& l : L) { 
		if (l.isFaceOf(cube)) { ++counter; }
	}

	return counter;
}


Cube MorseComplex::getUnpairedFace(const Cube& cube, const vector<Cube>& L) const {
	for (const Cube& l : L) { if (l.isFaceOf(cube)) { return l; } }
	
	return cube;
}


void MorseComplex::insertToC(const Cube& cube) {
	lock_guard<std::mutex> lock(mutexC);
	C[cube.dim].push_back(cube);
}


void MorseComplex::insertToV(const Cube& cube0, const Cube& cube1) {
	lock_guard<std::mutex> lock(mutexV);
	V.emplace(cube0, cube1); coV.emplace(cube1, cube0);
}


void MorseComplex::processLowerStarsWithPerturbation(const value_t& threshold) {
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
				if (getValue(x, y, z) > threshold) { continue; }

				L = getLowerStar(x, y, z);

				if (L.size() == 0) { C[0].push_back(Cube(*this, x, y, z, 0, 0)); }
				else {
					sort(L.begin(), L.end());

					alpha = L[0];
					V.emplace(Cube(*this, x, y, z, 0, 0), alpha); coV.emplace(alpha, Cube(*this, x, y, z, 0, 0));
					L.erase(L.begin(), L.begin()+1);
					
					for (const Cube& beta : L) {
						if (beta.dim == 1) { PQzero.push(beta); }
						else if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { PQone.push(beta); }
					}

					while(!PQzero.empty() || !PQone.empty()) {
						while(!PQone.empty()) {
							alpha = PQone.top(); PQone.pop();
							if (numUnpairedFaces(alpha, L) == 0) { PQzero.push(alpha); }
							else {	
								pair = getUnpairedFace(alpha, L);
								V.emplace(pair, alpha); coV.emplace(alpha, pair);
								removeFromPQ(pair, PQzero);
								L.erase(remove(L.begin(), L.end(), pair), L.end());
								L.erase(remove(L.begin(), L.end(), alpha), L.end());
								for (const Cube& beta : L) {
									if ((alpha.isFaceOf(beta) || pair.isFaceOf(beta)) && numUnpairedFaces(beta, L) == 1) {
										PQone.push(beta);
									}
								}
							}
						}
						if (!PQzero.empty()) {
							alpha = PQzero.top();
							PQzero.pop();
							C[alpha.dim].push_back(alpha);
							L.erase(remove(L.begin(), L.end(), alpha), L.end());
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

	processedLowerStars = true;
}


void MorseComplex::processLowerStarsWithoutPerturbation(const value_t& threshold) {
	if (processedLowerStars) {
		cerr << "Lower stars already processed!" << endl;
		return;
	}

	priority_queue<Cube, vector<Cube>, ReverseOrder> PQzero;
	priority_queue<Cube, vector<Cube>, ReverseOrder> PQone;
	Cube alpha;
	Cube pair;

	vector<vector<Cube>> lowerStars;
	getLowerStarsWithoutPerturbation(lowerStars, threshold);

	for (index_t x = 0; x < shape[0]; ++x) {
		for (index_t y = 0; y < shape[1]; ++y) {
			for (index_t z = 0; z < shape[2]; ++z) {
				if (getValue(x, y, z) > threshold) { continue; }

				vector<Cube>& L = lowerStars[hashVoxel(vector<index_t> {x,y,z})];

				if (L.size() == 0) { C[0].push_back(Cube(*this, x, y, z, 0, 0)); }
				else {
					sort(L.begin(), L.end());

					alpha = L[0];
					V.emplace(Cube(*this, x, y, z, 0, 0), alpha); coV.emplace(alpha, Cube(*this, x, y, z, 0, 0));
					L.erase(L.begin(), L.begin()+1);
					
					for (const Cube& beta : L) {
						if (beta.dim == 1) { PQzero.push(beta); }
						else if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { PQone.push(beta); }
					}

					while(!PQzero.empty() || !PQone.empty()) {
						while(!PQone.empty()) {
							alpha = PQone.top(); PQone.pop();
							if (numUnpairedFaces(alpha, L) == 0) { PQzero.push(alpha); }
							else {	
								pair = getUnpairedFace(alpha, L);
								V.emplace(pair, alpha); coV.emplace(alpha, pair);
								removeFromPQ(pair, PQzero);
								L.erase(remove(L.begin(), L.end(), pair), L.end());
								L.erase(remove(L.begin(), L.end(), alpha), L.end());
								for (const Cube& beta : L) {
									if ((alpha.isFaceOf(beta) || pair.isFaceOf(beta)) && numUnpairedFaces(beta, L) == 1) {
										PQone.push(beta);
									}
								}
							}
						}
						if (!PQzero.empty()) {
							alpha = PQzero.top();
							PQzero.pop();
							C[alpha.dim].push_back(alpha);
							L.erase(remove(L.begin(), L.end(), alpha), L.end());
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

	processedLowerStars = true;
}


void MorseComplex::processLowerStarsBetween(const index_t& xMin, const index_t& xMax, const index_t& yMin, const index_t& yMax, 
											const index_t& zMin, const index_t& zMax, const value_t& threshold) {
	priority_queue<Cube, vector<Cube>, ReverseOrder> PQzero;
	priority_queue<Cube, vector<Cube>, ReverseOrder> PQone;
	Cube alpha;
	Cube pair;

	index_t xBound = min(xMax, shape[0]);
	index_t yBound = min(yMax, shape[1]);
	index_t zBound = min(zMax, shape[2]);

	for (index_t x = xMin; x < xBound; ++x) {
		for (index_t y = yMin; y < yBound; ++y) {
			for (index_t z = zMin; z < zBound; ++z) {
				if (getValue(x, y, z) > threshold) { continue; }

				vector<Cube> L = getLowerStar(x, y, z);

				if (L.size() == 0) { insertToC(Cube(*this, x, y, z, 0, 0)); }
				else {
					sort(L.begin(), L.end());
					alpha = L[0];
					insertToV(Cube(*this, x, y, z, 0, 0), alpha);
					L.erase(L.begin(), L.begin()+1);
					
					for (const Cube& beta : L) {
						if (beta.dim == 1) { PQzero.push(beta); }
						else if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { 
							PQone.push(beta);
						}
					}

					while(!PQzero.empty() || !PQone.empty()) {
						while(!PQone.empty()) {
							alpha = PQone.top(); PQone.pop();
							if (numUnpairedFaces(alpha, L) == 0) { PQzero.push(alpha); }
							else {	
								pair = getUnpairedFace(alpha, L);
								insertToV(pair, alpha);
								removeFromPQ(pair, PQzero);
								L.erase(remove(L.begin(), L.end(), pair), L.end());
								L.erase(remove(L.begin(), L.end(), alpha), L.end());
								for (const Cube& beta : L) {
									if ((alpha.isFaceOf(beta) || pair.isFaceOf(beta)) && numUnpairedFaces(beta, L) == 1) {
										PQone.push(beta);
									}
								}
							}
						}
						if (!PQzero.empty()) {
							alpha = PQzero.top();
							PQzero.pop();
							insertToC(alpha);
							L.erase(remove(L.begin(), L.end(), alpha), L.end());
							for (const Cube& beta : L) {
								if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { PQone.push(beta); }
							}
						}
					}
				}
			}
		}
	}
}


vector<pair<Cube,Cube>> MorseComplex::processLowerStarsWithoutPerturbationBetween(const index_t& xMin, const index_t& xMax, const index_t& yMin, 
																					const index_t& yMax, const index_t& zMin, const index_t& zMax,
																					const value_t& threshold) {
	vector<pair<Cube,Cube>> pairs;

	priority_queue<Cube, vector<Cube>, ReverseOrder> PQzero;
	priority_queue<Cube, vector<Cube>, ReverseOrder> PQone;
	Cube alpha;
	Cube pair;

	index_t xBound = min(xMax, shape[0]);
	index_t yBound = min(yMax, shape[1]);
	index_t zBound = min(zMax, shape[2]);

	vector<vector<Cube>> lowerStars;
	getLowerStarsWithoutPerturbationBetween(lowerStars, xMin, xBound, yMin, yBound, zMin, zBound, threshold);

	for (index_t x = xMin; x < xBound; ++x) {
		for (index_t y = yMin; y < yBound; ++y) {
			for (index_t z = zMin; z < zBound; ++z) {
				if (getValue(x, y, z) > threshold) { continue; }

				vector<Cube>& L = lowerStars[hashVoxel(vector<index_t> {x,y,z})];

				if (L.size() == 0) { 
					pairs.push_back(make_pair(Cube(*this, x, y, z, 0, 0), Cube(*this, x, y, z, 0, 0)));
					}
				else {
					sort(L.begin(), L.end());
					alpha = L[0];
					pairs.push_back(make_pair(Cube(*this, x, y, z, 0, 0), alpha));
					L.erase(L.begin(), L.begin()+1);
					
					for (const Cube& beta : L) {
						if (beta.dim == 1) { PQzero.push(beta); }
						else if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { PQone.push(beta); }
					}

					while(!PQzero.empty() || !PQone.empty()) {
						while(!PQone.empty()) {
							alpha = PQone.top(); PQone.pop();
							if (numUnpairedFaces(alpha, L) == 0) { PQzero.push(alpha); }
							else {	
								pair = getUnpairedFace(alpha, L);
								pairs.push_back(make_pair(pair, alpha));
								removeFromPQ(pair, PQzero);
								L.erase(remove(L.begin(), L.end(), pair), L.end());
								L.erase(remove(L.begin(), L.end(), alpha), L.end());
								for (const Cube& beta : L) {
									if ((alpha.isFaceOf(beta) || pair.isFaceOf(beta)) && numUnpairedFaces(beta, L) == 1) {
										PQone.push(beta);
									}
								}
							}
						}
						if (!PQzero.empty()) {
							alpha = PQzero.top();
							PQzero.pop();
							pairs.push_back(make_pair(alpha, alpha));
							L.erase(remove(L.begin(), L.end(), alpha), L.end());
							for (const Cube& beta : L) {
								if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { PQone.push(beta); }
							}
						}
					}
				}
			}
		}
	}

	return pairs;
}


// void MorseComplex::processLowerStarsWithoutPerturbationBetween(const index_t& xMin, const index_t& xMax, const index_t& yMin, 
// 																const index_t& yMax, const index_t& zMin, const index_t& zMax,
// 																const value_t& threshold) {
// 	priority_queue<Cube, vector<Cube>, ReverseOrder> PQzero;
// 	priority_queue<Cube, vector<Cube>, ReverseOrder> PQone;
// 	Cube alpha;
// 	Cube pair;

// 	index_t xBound = min(xMax, shape[0]);
// 	index_t yBound = min(yMax, shape[1]);
// 	index_t zBound = min(zMax, shape[2]);

// 	vector<vector<Cube>> lowerStars;
// 	getLowerStarsWithoutPerturbationBetween(lowerStars, xMin, xBound, yMin, yBound, zMin, zBound, threshold);

// 	for (index_t x = xMin; x < xBound; ++x) {
// 		for (index_t y = yMin; y < yBound; ++y) {
// 			for (index_t z = zMin; z < zBound; ++z) {
// 				if (getValue(x, y, z) > threshold) { continue; }

// 				vector<Cube>& L = lowerStars[hashVoxel(vector<index_t> {x,y,z})];

// 				if (L.size() == 0) { insertToC(Cube(*this, x, y, z, 0, 0)); }
// 				else {
// 					sort(L.begin(), L.end());
// 					alpha = L[0];
// 					insertToV(Cube(*this, x, y, z, 0, 0), alpha);
// 					L.erase(L.begin(), L.begin()+1);
					
// 					for (const Cube& beta : L) {
// 						if (beta.dim == 1) { PQzero.push(beta); }
// 						else if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { PQone.push(beta); }
// 					}

// 					while(!PQzero.empty() || !PQone.empty()) {
// 						while(!PQone.empty()) {
// 							alpha = PQone.top(); PQone.pop();
// 							if (numUnpairedFaces(alpha, L) == 0) { PQzero.push(alpha); }
// 							else {	
// 								pair = getUnpairedFace(alpha, L);
// 								insertToV(pair, alpha);
// 								removeFromPQ(pair, PQzero);
// 								L.erase(remove(L.begin(), L.end(), pair), L.end());
// 								L.erase(remove(L.begin(), L.end(), alpha), L.end());
// 								for (const Cube& beta : L) {
// 									if ((alpha.isFaceOf(beta) || pair.isFaceOf(beta)) && numUnpairedFaces(beta, L) == 1) {
// 										PQone.push(beta);
// 									}
// 								}
// 							}
// 						}
// 						if (!PQzero.empty()) {
// 							alpha = PQzero.top();
// 							PQzero.pop();
// 							insertToC(alpha);
// 							L.erase(remove(L.begin(), L.end(), alpha), L.end());
// 							for (const Cube& beta : L) {
// 								if (alpha.isFaceOf(beta) && numUnpairedFaces(beta, L) == 1) { PQone.push(beta); }
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }


void MorseComplex::processLowerStarsWithPerturbationParallel(const index_t& xPatch, const index_t& yPatch, const index_t& zPatch,
																const value_t& threshold) {
	if (!perturbed) { 
		cerr << "Perturb Image first!" << endl;
		return;
	} else if (processedLowerStars) {
		cerr << "Lower stars already processed!" << endl;
		return;
	}

	index_t batchX = shape[0]/xPatch + 1;
	index_t batchY = shape[1]/yPatch + 1;
	index_t batchZ = shape[2]/zPatch + 1;

	vector<std::future<void>> futures;

	for (index_t x = 0; x < xPatch; ++x) {
		for (index_t y = 0; y < yPatch; ++y) {
			for (index_t z = 0; z < zPatch; ++z) {
				index_t xMin = x * batchX;
				index_t xMax = xMin + batchX;
				index_t yMin = y * batchY;
				index_t yMax = yMin + batchY;
				index_t zMin = z * batchZ;
				index_t zMax = zMin + batchZ;
				futures.push_back(async(launch::async, &MorseComplex::processLowerStarsBetween,
									this, xMin, xMax, yMin, yMax, zMin, zMax, threshold));
			}
		}
	}

	for (auto& future : futures) {
        future.wait();
    }

	processedLowerStars = true;
}


void MorseComplex::processLowerStarsWithoutPerturbationParallel(const index_t& xPatch, const index_t& yPatch, const index_t& zPatch,
																const value_t& threshold) {
	if (processedLowerStars) {
		cerr << "Lower stars already processed!" << endl;
		return;
	}

	index_t batchX = shape[0]/xPatch + 1;
	index_t batchY = shape[1]/yPatch + 1;
	index_t batchZ = shape[2]/zPatch + 1;

	vector<std::future<vector<pair<Cube,Cube>>>> futures;

	for (index_t x = 0; x < xPatch; ++x) {
		for (index_t y = 0; y < yPatch; ++y) {
			for (index_t z = 0; z < zPatch; ++z) {
				index_t xMin = x * batchX;
				index_t xMax = xMin + batchX;
				index_t yMin = y * batchY;
				index_t yMax = yMin + batchY;
				index_t zMin = z * batchZ;
				index_t zMax = zMin + batchZ;
				futures.push_back(async(launch::async, &MorseComplex::processLowerStarsWithoutPerturbationBetween,
									this, xMin, xMax, yMin, yMax, zMin, zMax, threshold));
			}
		}
	}

	for (auto& future : futures) {
        vector<pair<Cube,Cube>> pairs = future.get();
		for (pair<Cube,Cube>& p : pairs) {
			if (p.first == p.second) { C[p.first.dim].push_back(p.first); }
			else { V.emplace(p.first, p.second); coV.emplace(p.second, p.first); }
		}
    }

	processedLowerStars = true;
}


// void MorseComplex::processLowerStarsWithoutPerturbationParallel(const index_t& xPatch, const index_t& yPatch, const index_t& zPatch,
// 																const value_t& threshold) {
// 	if (processedLowerStars) {
// 		cerr << "Lower stars already processed!" << endl;
// 		return;
// 	}

// 	index_t batchX = shape[0]/xPatch + 1;
// 	index_t batchY = shape[1]/yPatch + 1;
// 	index_t batchZ = shape[2]/zPatch + 1;

// 	vector<std::future<void>> futures;

// 	for (index_t x = 0; x < xPatch; ++x) {
// 		for (index_t y = 0; y < yPatch; ++y) {
// 			for (index_t z = 0; z < zPatch; ++z) {
// 				index_t xMin = x * batchX;
// 				index_t xMax = xMin + batchX;
// 				index_t yMin = y * batchY;
// 				index_t yMax = yMin + batchY;
// 				index_t zMin = z * batchZ;
// 				index_t zMax = zMin + batchZ;
// 				futures.push_back(async(launch::async, &MorseComplex::processLowerStarsWithoutPerturbationBetween,
// 									this, xMin, xMax, yMin, yMax, zMin, zMax, threshold));
// 			}
// 		}
// 	}

// 	for (auto& future : futures) {
//         future.wait();
//     }

// 	processedLowerStars = true;
// }


void MorseComplex::traverseFlow(const Cube& s, vector<tuple<Cube, Cube, Cube>>& flow, bool coordinated) const {
#ifdef USE_CUBE_MAP
	CubeMap<size_t> nrIn(shape);
	CubeMap<size_t> count(shape);
#else
	unordered_map<Cube, size_t, Cube::Hash> nrIn;
	unordered_map<Cube, size_t, Cube::Hash> count;
#endif
	if (coordinated) {
		traverseFlow(s, flow, false);
		for (const tuple<Cube, Cube, Cube>& f : flow) {
#ifdef USE_CUBE_MAP
			nrIn.increment(get<2>(f));
#else
			auto it = nrIn.find(get<2>(f));
			if (it != nrIn.end()) { ++(it->second); }
			else { nrIn.emplace(get<2>(f), 1); }
#endif
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
			if (find(C[b.dim].begin(), C[b.dim].end(), b) != C[b.dim].end()) { c = b; }
			else {
#ifdef USE_CUBE_MAP
				c = V.getValue(b);
				if (c == Cube()) { c = a; }
#else
				auto it = V.find(b);
				if (it != V.end()) { c = it->second; }
				else { c = a; }
#endif
			}
			if (c != a) {
				flow.push_back(tuple(a, b, c));
				if (c != b) {
					auto it = seen.find(c);
					if (it == seen.end()) {
						if (coordinated) {
#ifdef USE_CUBE_MAP
							count.increment(c);
							if (count.getValue(c) != nrIn.getValue(c)) { continue; }
#else
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
#endif
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
#ifdef USE_CUBE_MAP
	CubeMap<size_t> nrIn(shape);
	CubeMap<size_t> count(shape);
#else
	unordered_map<Cube, size_t, Cube::Hash> nrIn;
	unordered_map<Cube, size_t, Cube::Hash> count;
#endif
	if (coordinated) {
		traverseCoflow(s, flow, false);
		for (const tuple<Cube, Cube, Cube>& f : flow) {
#ifdef USE_CUBE_MAP
			nrIn.increment(get<2>(f));
#else
			auto it = nrIn.find(get<2>(f));
			if (it != nrIn.end()) { ++(it->second); }
			else { nrIn.emplace(get<2>(f), 1); }
#endif
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
			if (find(C[b.dim].begin(), C[b.dim].end(), b) != C[b.dim].end()) { c = b; }
			else {
#ifdef USE_CUBE_MAP
				c = coV.getValue(b);
				if (c == Cube()) { c = a; }
#else
				auto it = coV.find(b);
				if (it != coV.end()) { c = it->second; }
				else { c = a; }
#endif
			}
			if (c != a) {
				flow.push_back(tuple(a, b, c));
				if (c != b) {
					auto it = seen.find(c);
					if (it == seen.end()) {
						if (coordinated) {
#ifdef USE_CUBE_MAP
							count.increment(c);
							if (count.getValue(c) != nrIn.getValue(c)) { continue; }
#else
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
#endif
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


vector<tuple<index_t,index_t,index_t>> MorseComplex::extractMorseSkeletonInDimBelow(const uint8_t& dim, const value_t& threshold) {
	vector<tuple<index_t,index_t,index_t>> skeleton;
	vector<tuple<index_t,index_t,index_t>> voxels;

	if (dim == 0) {
		for (const Cube& c : C[0]) {
			if (c.birth >= threshold) { continue; }

			voxels = c.getVertices();
			for (const tuple<index_t,index_t,index_t>& voxel : voxels) { skeleton.push_back(voxel); }
		}
	} else {
		vector<tuple<Cube, Cube, Cube>> flow;
		for (const Cube& c : C[dim]) {
			if (c.birth >= threshold) { continue; }

			voxels = c.getVertices();
			for (const tuple<index_t,index_t,index_t>& voxel : voxels) { skeleton.push_back(voxel); }

			flow.clear();
			traverseFlow(c, flow, false);
			for (const tuple<Cube, Cube, Cube>& t : flow) {
				if (get<1>(t) != get<2>(t)) { 
					voxels = get<2>(t).getVertices();
					for (const tuple<index_t,index_t,index_t>& voxel : voxels) { skeleton.push_back(voxel); }
				}
			}
		}
	}

	return skeleton;
}


vector<tuple<index_t,index_t,index_t>> MorseComplex::extractMorseSkeletonOfBatchInDimBelow(const uint8_t& dim, const size_t& start, const size_t& end, const value_t& threshold) {
	vector<tuple<index_t,index_t,index_t>> skeleton;
	vector<tuple<index_t,index_t,index_t>> voxels;
	size_t bound = min(end, C[dim].size());

	if (dim == 0) {
		for (size_t i = start; i < bound; ++i) {
			Cube& c = C[0][i];
			if (c.birth >= threshold) { continue; }

			voxels = c.getVertices();
			for (const tuple<index_t,index_t,index_t>& voxel : voxels) { skeleton.push_back(voxel); }
		}
	} else {
		vector<tuple<Cube, Cube, Cube>> flow;
		for (size_t i = start; i < bound; ++i) {
			Cube& c = C[dim][i];
			if (c.birth >= threshold) { continue; }

			voxels = c.getVertices();
			for (const tuple<index_t,index_t,index_t>& voxel : voxels) { skeleton.push_back(voxel); }

			flow.clear();
			traverseFlow(c, flow, false);
			for (const tuple<Cube, Cube, Cube>& t : flow) {
				if (get<1>(t) != get<2>(t)) { 
					voxels = get<2>(t).getVertices();
					for (const tuple<index_t,index_t,index_t>& voxel : voxels) { skeleton.push_back(voxel); }
				}
			}
		}
	}

	return skeleton;
}


void MorseComplex::cancelPair(const Cube&s, const Cube& t) {
	vector<tuple<Cube, Cube, Cube>> connection;
	getConnections(s, t, connection);

	C[s.dim].erase(remove(C[s.dim].begin(), C[s.dim].end(), s), C[s.dim].end());
	C[t.dim].erase(remove(C[t.dim].begin(), C[t.dim].end(), t), C[t.dim].end());

	for (tuple<Cube, Cube, Cube> con : connection) {
		if (get<1>(con) != get<2>(con)) {
#ifdef USE_CUBE_MAP
			V.erase(get<1>(con));
			coV.erase(get<2>(con));
#else
			auto it = V.find(get<1>(con));
			if (it != V.end()) { V.erase(it); }
			else { cerr << "Error: key not found!" << endl; exit(EXIT_FAILURE); }
			it = coV.find(get<2>(con));
			if (it != coV.end()) { coV.erase(it); }
			else { cerr << "Error: key not found!" << endl; exit(EXIT_FAILURE);}
#endif
		}
		V.emplace(get<1>(con), get<0>(con));
		coV.emplace(get<0>(con), get<1>(con));
	}
}


auto boundaryComparator = [](const auto& lhs, const auto& rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs);
};