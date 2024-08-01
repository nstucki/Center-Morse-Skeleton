#include "config.h"
#include "data_structures.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

using namespace std;
namespace py = pybind11;


py::array_t<int64_t> convertListToNumpyArray(vector<vector<index_t>> &vector) {
    int size = vector.size();
    int dim = vector[0].size();
    py::array_t<int64_t> numpyArray(
        {size, dim}, {dim * sizeof(int64_t), sizeof(int64_t)});
    auto numpyArrayView = numpyArray.mutable_unchecked();
    for (int i = 0; i < vector.size(); i++) {
        for (int j = 0; j < vector[i].size(); j++) {
            numpyArrayView(i, j) = vector[i][j];
        }
    }
    return numpyArray;
}


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

        .def("process_lower_stars", &MorseComplex::processLowerStars, py::arg("threshold")=INFTY, py::arg("x_patch")=2, py::arg("y_patch")=2, py::arg("z_patch")=2)

        .def("sort_critical_cells", &MorseComplex::sortCriticalCells, py::arg("dimensions")=vector<uint8_t>{0,1,2,3})

        .def("check_gradient_vectorfield", &MorseComplex::checkV)

        .def("get_number_of_critical_cells", &MorseComplex::getNumberOfCriticalCells, py::arg("threshold")=INFTY)

        .def("get_critical_cells", &MorseComplex::getCriticalCells)

        .def("get_morse_boundary", &MorseComplex::getMorseBoundary, py::arg("cube"))

        .def("get_morse_coboundary", &MorseComplex::getMorseCoboundary, py::arg("cube"))

        .def("cancel_pairs_below", &MorseComplex::cancelPairsBelow, py::arg("threshold")=INFTY, py::arg("order_dim")=">", 
                                                                    py::arg("order_value")=">", py::arg("print")=true)

        .def("cancel_pair", &MorseComplex::cancelPairByIndex, py::arg("dim_s"), py::arg("index_s"), py::arg("dim_t"), py::arg("index_t"))

        .def("cancel_pairs_below_in_dim", &MorseComplex::cancelPairsBelowInDim, py::arg("dim"), py::arg("threshold")=INFTY, 
                                                                    py::arg("order_value")=">", py::arg("print")=true)

        .def("cancel_pairs_above", &MorseComplex::cancelPairsAbove, py::arg("threshold")=INFTY, py::arg("order_dim")=">", 
                                                                    py::arg("order_value")=">", py::arg("print")=true)

        .def("cancel_all_pairs_below", &MorseComplex::cancelAllPairsBelow, py::arg("threshold")=INFTY, py::arg("dimensions")=vector<uint8_t>{3,2,1})

        .def("cancel_all_pairs_above", &MorseComplex::cancelAllPairsAbove, py::arg("threshold")=INFTY, py::arg("dimensions")=vector<uint8_t>{0,1,2})

        .def("cancel_low_persistence_pairs_below", &MorseComplex::cancelLowPersistencePairsBelow, py::arg("threshold")=INFTY, py::arg("epsilon")=0, py::arg("dimensions")=vector<uint8_t>{3,2,1})

        .def("cancel_low_persistence_pairs_above", &MorseComplex::cancelLowPersistencePairsAbove, py::arg("threshold")=INFTY, py::arg("epsilon")=0, py::arg("dimensions")=vector<uint8_t>{0,1,2})

        .def("cancel_boundary_pairs_below", &MorseComplex::cancelBoundaryPairsBelow, py::arg("threshold")=INFTY, py::arg("delta")=0, py::arg("dimensions")=vector<uint8_t>{3,2,1})

        .def("cancel_boundary_pairs_above", &MorseComplex::cancelBoundaryPairsAbove, py::arg("threshold")=INFTY, py::arg("delta")=0, py::arg("dimensions")=vector<uint8_t>{0,1,2})

        .def("cancel_close_pairs_below", &MorseComplex::cancelClosePairsBelow, py::arg("threshold")=INFTY, py::arg("delta")=0,
                                                                                py::arg("print")=true)

        .def("extract_morse_skeleton_below", &MorseComplex::extractMorseSkeletonBelow, py::arg("threshold")=INFTY, py::arg("dimension")=3)

        .def("extract_morse_skeleton_above", &MorseComplex::extractMorseSkeletonAbove, py::arg("threshold")=-INFTY)

        .def("extract_morse_skeleton_parallel_below", &MorseComplex::extractMorseSkeletonParallelBelow, py::arg("threshold")=INFTY)

        .def("extract_morse_skeleton_parallel_above", &MorseComplex::extractMorseSkeletonParallelAbove, py::arg("threshold")=-INFTY)

        .def("extract_morse_skeleton_batchwise_below", &MorseComplex::extractMorseSkeletonBatchwiseBelow, py::arg("threshold")=INFTY, py::arg("batches")=128)

        .def("extract_morse_skeleton_batchwise_above", &MorseComplex::extractMorseSkeletonBatchwiseAbove, py::arg("threshold")=-INFTY, py::arg("batches")=128)


        .def("get_lower_morse_skeleton", [](MorseComplex &self) {
            auto skeleton = self.getLowerMorseSkeleton();
            return convertListToNumpyArray(skeleton);
        })

        .def("get_upper_morse_skeleton", [](MorseComplex &self) {
            auto skeleton = self.getUpperMorseSkeleton();
            return convertListToNumpyArray(skeleton);
        });

        


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