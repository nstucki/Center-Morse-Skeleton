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

        .def("get_value", &MorseComplex::getValue)

        .def("get_perturbation", &MorseComplex::getPerturbation)

        .def("perturb_image", &MorseComplex::perturbImage, py::arg("epsilon")=INFTY)

        .def("perturb_image_minimal", &MorseComplex::perturbImageMinimal)

        .def("process_lower_stars", &MorseComplex::processLowerStars, py::arg("x_patch")=1, py::arg("y_patch")=1, py::arg("z_patch")=1 )

        .def("check_gradient_vectorfield", &MorseComplex::checkV)

        .def("get_number_of_critical_cells", &MorseComplex::getNumberOfCriticalCells, py::arg("threshold")=INFTY)

        .def("get_critical_cells", &MorseComplex::getCriticalCells)

        .def("get_morse_boundary", &MorseComplex::getMorseBoundary, py::arg("cube"))

        .def("get_morse_coboundary", &MorseComplex::getMorseCoboundary, py::arg("cube"))

        .def("cancel_pairs", &MorseComplex::cancelPairs, py::arg("threshold")=INFTY, py::arg("order_dim_below")=">", 
                                                            py::arg("order_value_below")=">", py::arg("order_dim_above")="<",
                                                            py::arg("order_value_above")="<", py::arg("print")=true)

        .def("cancel_low_persistence_pairs_below", &MorseComplex::cancelLowPersistencePairsBelow, py::arg("threshold")=INFTY, py::arg("delta")=0,
                                                                                                    py::arg("print")=true)

        .def("cancel_boundary_pairs_below", &MorseComplex::cancelBoundaryPairsBelow, py::arg("threshold")=INFTY, py::arg("delta")=0,
                                                                                        py::arg("print")=true)

        .def("prepare_morse_skeleton_below", &MorseComplex::prepareMorseSkeletonBelow, py::arg("threshold")=INFTY, py::arg("epsilon")=0,
                                                                                        py::arg("delta")=-1, py::arg("print")=true)

        .def("prepare_morse_skeleton_test_below", &MorseComplex::prepareMorseSkeletonTestBelow, py::arg("threshold")=INFTY, py::arg("epsilon")=0,
                                                                                                py::arg("delta")=-1, py::arg("print")=true)

        .def("prepare_morse_skeleton_above", &MorseComplex::prepareMorseSkeletonAbove, py::arg("threshold")=INFTY, py::arg("tolerance")=0, 
                                                                                        py::arg("print")=true)

        .def("extract_morse_skeleton_below", &MorseComplex::extractMorseSkeletonBelow, py::arg("threshold")=INFTY)

        .def("extract_morse_skeleton_above", &MorseComplex::extractMorseSkeletonAbove, py::arg("threshold")=INFTY)

        .def("get_morse_skeleton_below", &MorseComplex::getMorseSkeletonVoxelsBelow)

        .def("get_morse_skeleton_above", &MorseComplex::getMorseSkeletonVoxelsAbove);

        


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
        })
        
        .def("get_voxels", &Cube::getVertices);
}