#include "config.h"
#include "data_structures.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

using namespace std;
namespace py = pybind11;



PYBIND11_MODULE(morse_complex, m) {
    py::class_<MorseComplex>(m, "MorseComplex")
        .def(py::init([](py::array_t<value_t>& input) {
                vector<index_t> shape(input.shape(), input.shape() + input.ndim());
                vector<value_t> inputVector(input.mutable_data(), input.mutable_data() + input.size());
                return MorseComplex(std::move(inputVector), std::move(shape));
            }))

        .def_readonly("shape", &MorseComplex::shape)

        .def("get_value", &MorseComplex::getValue)

        .def("get_perturbation", &MorseComplex::getPerturbation)

        .def("perturb_image", &MorseComplex::perturbImage, py::arg("epsilon")=INFTY)
        
        .def("process_lower_stars", &MorseComplex::processLowerStars)

        .def("check_gradient_vectorfield", &MorseComplex::checkV)
        
        .def("cancel_pairs_below", &MorseComplex::cancelPairsBelow, py::arg("threshold"), py::arg("print")=false)
        
        .def("cancel_pairs_above", &MorseComplex::cancelPairsAbove, py::arg("threshold"), py::arg("print")=false)

        .def("get_number_of_critical_cells", &MorseComplex::getNumberOfCriticalCells, py::arg("threshold")=INFTY)

        .def("get_critical_cells", &MorseComplex::getCriticalCells)

        .def("get_morse_boundary", &MorseComplex::getMorseBoundary, py::arg("cube"));

        


    py::class_<Cube>(m, "Cube")
        .def_readonly("birth", &Cube::birth)
        .def_readonly("x", &Cube::x)
        .def_readonly("y", &Cube::y)
        .def_readonly("z", &Cube::z)
        .def_readonly("type", &Cube::type)
        .def_readonly("dim", &Cube::dim)
        .def("__repr__", [](Cube &self) {
          return "birth=" + to_string(self.birth) + ", x=" + to_string(self.x) + ", y=" + to_string(self.y) 
                    + ", z=" + to_string(self.z) + ", type=" + to_string(self.type) + ", dim=" + to_string(self.dim);
        });
}