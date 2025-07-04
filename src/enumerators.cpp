#include "enumerators.h"



BoundaryEnumerator::BoundaryEnumerator(const MorseComplex& _mc) : mc(_mc) { nextFace = Cube(); }


void BoundaryEnumerator::setBoundaryEnumerator(const Cube& _cube) {
	cube = _cube;
	position = 0; 
}


bool BoundaryEnumerator::hasNextFace() {
	switch (cube.dim) {
		case 0:
		return false;
		break;

		case 1:
		switch (cube.type) {
			case 0:
			switch (position) {
				case 0:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 0, 0);
				break;

				case 1:
				nextFace = Cube(mc, cube.x+1, cube.y, cube.z, 0, 0);
				break;

				case 2:
				return false;
			}
			break;

			case 1:
			switch (position) {
				case 0:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 0, 0);
				break;

				case 1:
				nextFace = Cube(mc, cube.x, cube.y+1, cube.z, 0, 0);
				break;

				case 2:
				return false;
			}
			break;
			
			case 2:
			switch (position) {
				case 0:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 0, 0);
				break;

				case 1:
				nextFace = Cube(mc, cube.x, cube.y, cube.z+1, 0, 0);
				break;

				case 2:
				return false;
			}
			break;
		}
		break;

		case 2:
		switch (cube.type) {
			case 0:
			switch (position) {
				case 0:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 1, 1);
				break;

				case 1:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 2, 1);
				break;

				case 2:
				nextFace = Cube(mc, cube.x, cube.y, cube.z+1, 1, 1);
				break;

				case 3:
				nextFace = Cube(mc, cube.x, cube.y+1, cube.z, 2, 1);
				break;

				case 4:
				return false;
			}
			break;

			case 1:
			switch (position) {
				case 0:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 0, 1);
				break;

				case 1:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 2, 1);
				break;

				case 2:
				nextFace = Cube(mc, cube.x, cube.y, cube.z+1, 0, 1);
				break;

				case 3:
				nextFace = Cube(mc, cube.x+1, cube.y, cube.z, 2, 1);
				break;
				
				case 4:
				return false;
			}
			break;
			
			case 2:
			switch (position) {
				case 0:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 0, 1);
				break;

				case 1:
				nextFace = Cube(mc, cube.x, cube.y, cube.z, 1, 1);
				break;

				case 2:
				nextFace = Cube(mc, cube.x, cube.y+1, cube.z, 0, 1);
				break;

				case 3:
				nextFace = Cube(mc, cube.x+1, cube.y, cube.z, 1, 1);
				break;

				case 4:
				return false;
			}
			break;
		}
		break;

		case 3:
		switch (position) {
			case 0:
			nextFace = Cube(mc, cube.x, cube.y, cube.z, 0, 2);
			break;

			case 1:
			nextFace = Cube(mc, cube.x, cube.y, cube.z, 1, 2);
			break;

			case 2:
			nextFace = Cube(mc, cube.x, cube.y, cube.z, 2, 2);
			break;

			case 3:
			nextFace = Cube(mc, cube.x, cube.y, cube.z+1, 2, 2);
			break;

			case 4:
			nextFace = Cube(mc, cube.x, cube.y+1, cube.z, 1, 2);
			break;

			case 5:
			nextFace = Cube(mc, cube.x+1, cube.y, cube.z, 0, 2);
			break;

			case 6:
			return false;
		}
		break;
	}
	++position;

	return true;
}



CoboundaryEnumerator::CoboundaryEnumerator(const MorseComplex& _mc) : mc(_mc) { nextCoface = Cube(); }


void CoboundaryEnumerator::setCoboundaryEnumerator(const Cube& _cube) {
	cube = _cube;
	position = 0; 
}


bool CoboundaryEnumerator::hasNextCoface() {
	switch (cube.dim) {
		case 0:
		switch (position) {
			case 0:
			nextCoface = Cube(mc, cube.x-1, cube.y, cube.z, 0, 1);
			if (nextCoface.birth != INFTY) { break; } 
			else { ++position; }

			case 1:
			nextCoface = Cube(mc, cube.x, cube.y-1, cube.z, 1, 1);
			if (nextCoface.birth != INFTY) { break; } 
			else { ++position; }

			case 2:
			nextCoface = Cube(mc, cube.x, cube.y, cube.z-1, 2, 1);
			if (nextCoface.birth != INFTY) { break; } 
			else { ++position; }

			case 3:
			nextCoface = Cube(mc, cube.x, cube.y, cube.z, 0, 1);
			if (nextCoface.birth != INFTY) { break; } 
			else { ++position; }

			case 4:
			nextCoface = Cube(mc, cube.x, cube.y, cube.z, 1, 1);
			if (nextCoface.birth != INFTY) { break; } 
			else { ++position; }

			case 5:
			nextCoface = Cube(mc, cube.x, cube.y, cube.z, 2, 1);
			if (nextCoface.birth != INFTY) { break; } 
			else { ++position; }

			case 6:
			return false;
		}
		break;

		case 1:
		switch (cube.type) {
			case 0:
			switch (position) {
				case 0:
				nextCoface = Cube(mc, cube.x, cube.y-1, cube.z, 2, 2);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 1:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z-1, 1, 2);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }
				
				case 2:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 1, 2);
				if (nextCoface.birth != INFTY) { break; }
				else { ++position; }

				case 3:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 2, 2);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 4:
				return false;
			}
			break;

			case 1:
			switch (position) {
				case 0:
				nextCoface = Cube(mc, cube.x-1, cube.y, cube.z, 2, 2);
				if (nextCoface.birth != INFTY) { break; }
				else { ++position; }

				case 1:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z-1, 0, 2);
				if (nextCoface.birth != INFTY) { break; }
				else { ++position; }

				case 2:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 0, 2);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 3:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 2, 2);
				if (nextCoface.birth != INFTY) { break; }
				else { ++position; }

				case 4:
				return false;
			}
			break;
			
			case 2:
			switch (position) {
				case 0:
				nextCoface = Cube(mc, cube.x-1, cube.y, cube.z, 1, 2);
				if (nextCoface.birth != INFTY) { break; }
				else { ++position; }

				case 1:
				nextCoface = Cube(mc, cube.x, cube.y-1, cube.z, 0, 2);
				if (nextCoface.birth != INFTY) { break; }
				else { ++position; }

				case 2:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 0, 2);
				if (nextCoface.birth != INFTY) { break; }
				else { ++position; }

				case 3:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 1, 2);
				if (nextCoface.birth != INFTY) { break; }
				else { ++position; }

				case 4:
				return false;
			}
			break;
		}
		break;

		case 2:
		switch(cube.type) {
			case 0:
			switch (position) {
				case 0:
				nextCoface = Cube(mc, cube.x-1, cube.y, cube.z, 0, 3);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 1:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 0, 3);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 2:
				return false;
			}
			break;

			case 1:
			switch(position) {
				case 0:
				nextCoface = Cube(mc, cube.x, cube.y-1, cube.z, 0, 3);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 1:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 0, 3);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 2:
				return false;
			}
			break;

			case 2:
			switch(position) {
				case 0:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z-1, 0, 3);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 1:
				nextCoface = Cube(mc, cube.x, cube.y, cube.z, 0, 3);
				if (nextCoface.birth != INFTY) { break; } 
				else { ++position; }

				case 2:
				return false;
			}
			break;
		}
		break;

		case 3:
		return false;
	}
	++position;

	return true;
}