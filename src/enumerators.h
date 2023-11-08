#pragma once

#include "data_structures.h"



class BoundaryEnumerator {
	public:
	Cube nextFace;
	
	BoundaryEnumerator(const MorseComplex& mc);
	void setBoundaryEnumerator(const Cube& cube);
	bool hasNextFace();

	private:
	const MorseComplex& mc;
	Cube cube;
	uint8_t position;
};



class CoboundaryEnumerator {
	public:
	Cube nextCoface;
	
	CoboundaryEnumerator(const MorseComplex& mc);
	void setCoboundaryEnumerator(const Cube& cube);
	bool hasNextCoface();

	private:
	const MorseComplex& mc;
	Cube cube;
	uint8_t position;
};