/**
 * PyFastPFOR
 *
 * Python bindings for the FastPFOR library:
 * https://github.com/lemire/FastPFor
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "headers/codecfactory.h"
#include "headers/deltautil.h"

namespace py = pybind11;

void exportCodecs(py::module& m);

using namespace FastPForLib;

const char * module_name = "pyfastpfor";

// Reports the SIMD instruction sets the extension was actually compiled
// with, based on preprocessor macros set by the compiler flags (see
// setup.py:simd_flags). This reflects the build, not the running CPU.
static std::vector<std::string> compiledSimdFeatures() {
  std::vector<std::string> features;
#if defined(__AVX512F__)
  features.push_back("AVX512F");
#endif
#if defined(__AVX2__)
  features.push_back("AVX2");
#endif
#if defined(__AVX__)
  features.push_back("AVX");
#endif
#if defined(__SSE4_2__)
  features.push_back("SSE4.2");
#endif
#if defined(__SSE4_1__)
  features.push_back("SSE4.1");
#endif
#if defined(__SSSE3__)
  features.push_back("SSSE3");
#endif
#if defined(__SSE3__)
  features.push_back("SSE3");
#endif
#if defined(__SSE2__)
  features.push_back("SSE2");
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
  features.push_back("NEON");
#endif
  return features;
}

struct IntegerCODECWrapper {
public:
  IntegerCODECWrapper(const std::string& codecName) {
    codec_ = factory.getFromName(codecName).get();
  }
  size_t encodeArray(
         py::array_t<uint32_t, py::array::c_style> input, size_t inputSize,
         py::array_t<uint32_t, py::array::c_style> output, size_t outputSize) {
    py::gil_scoped_release l;

    const uint32_t* inpBuff = input.data();

    uint32_t*       outBuff = output.mutable_data();
    size_t          compSize = outputSize;

    codec_->encodeArray(inpBuff, inputSize,
                        outBuff, compSize);

    return compSize;
  }
  size_t decodeArray(
         py::array_t<uint32_t, py::array::c_style> input, size_t inputSize,
         py::array_t<uint32_t, py::array::c_style> output, size_t outputSize) {
    py::gil_scoped_release l;

    const uint32_t* inpBuff = input.data();

    uint32_t*       outBuff = output.mutable_data();
    size_t          uncompSize = outputSize;

    codec_->decodeArray(inpBuff, inputSize, outBuff, uncompSize);

    return uncompSize;
  }
private:
  CODECFactory factory;
  IntegerCODEC* codec_;
};

/*
 * PYBIND11_MODULE is a replacement for PYBIND11_PLUGIN
 * introduced in Pybind 2.2. However, we don't require
 * Pybind to be >= 2.0 so we attempt to support older
 * Pybind versions as well.
 */
#ifdef PYBIND11_MODULE
PYBIND11_MODULE(pyfastpfor, m) {
  m.doc() = "Python Bindings for FastPFor library (fast integer compression).";
#else
PYBIND11_PLUGIN(pyfastpfor) {
  py::module m(module_name, "Python Bindings for FastPFor library (fast integer compression).");
#endif

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

  py::module codecModule = m.def_submodule("codecs", "Codecs class wrapper.");

  exportCodecs(codecModule);

  m.def("compiledSimdFeatures", &compiledSimdFeatures,
    "Returns\n"
    "----------\n"
    "    A list of SIMD instruction sets (e.g. ['SSE4.2'] or ['NEON'])\n"
    "    that this module was compiled with.");

  m.def("getCodec",
    [](const std::string & codecName) {
      // We know that FastPFor will keep this shared pointer alive forever
      // so it is safe just to reference codec
      return py::cast(new IntegerCODECWrapper(codecName),
                      py::return_value_policy::take_ownership);
    },
    py::arg("codecName"),
    "This is a codec-factory method.\n\n"
    "Parameters\n"
    "----------\n"
    "codecName: str\n"
    "    A name of the codec, e.g., simdfastpfor256\n"
    "\n"
    "Returns\n"
    "----------\n"
    "    A reference to the codec object");

  m.def("getCodecList", []() {
      py::list ret;
      for (const std::string& codecId : CODECFactory().allNames()) {
        ret.append(codecId);
      }
      return ret;
    },
    "Return a list of available codecs.\n\n"
    "Returns\n"
    "----------\n"
    "A list with codec names");

  m.def("delta1", [](py::array_t<uint32_t, py::array::c_style> input, size_t inputSize) -> void {
      uint32_t* buff = input.mutable_data();
      py::gil_scoped_release l;

      Delta::fastDelta(buff, inputSize);
  }, py::arg("input"), py::arg("inputSize"),
    "In-place computation of differences between adjacent numbers.\n\n"
    "Parameters\n"
    "----------\n"
    "input: input numpy C-style contiguous array to be uncompressed, e.g.:\n"
    "     input = numpy.array(range(256), dtype = np.uint32).ravel()\n"
    "inputSize: a number of integers to process\n"
    "\n"
    "Returns\n"
    "----------\n"
    "    None"
  );

  m.def("delta4",
    [](py::array_t<uint32_t, py::array::c_style> input, size_t inputSize) -> void {
      uint32_t* buff = input.mutable_data();
      py::gil_scoped_release l;

      Delta::deltaSIMD(buff, inputSize);
    }, py::arg("input"), py::arg("inputSize"),
    "In-place computation of differences between numbers that are 4 indices apart.\n"
    "Using delta4 and prefixSum4 increases space usage, but processing is faster.\n\n"
    "Parameters\n"
    "----------\n"
    "input: input numpy C-style contiguous array to be uncompressed, e.g.:\n"
    "     input = numpy.array(range(256), dtype = np.uint32).ravel()\n"
    "inputSize: a number of integers to process\n"
    "\n"
    "Returns\n"
    "----------\n"
    "    None"
  )
  ;

  m.def("prefixSum1", [](py::array_t<uint32_t, py::array::c_style> input, size_t inputSize) -> void {
      uint32_t* buff = input.mutable_data();
      py::gil_scoped_release l;

      Delta::fastinverseDelta2(buff, inputSize);
  }, py::arg("input"), py::arg("inputSize"),
    "In-place inversion of delta1.\n\n"
    "Parameters\n"
    "----------\n"
    "input: input numpy C-style contiguous array to be uncompressed, e.g.:\n"
    "     input = numpy.array(range(256), dtype = np.uint32).ravel()\n"
    "inputSize: a number of integers to process\n"
    "\n"
    "Returns\n"
    "----------\n"
    "    None"
  );

  m.def("prefixSum4",
    [](py::array_t<uint32_t, py::array::c_style> input, size_t inputSize) -> void {
      uint32_t* buff = input.mutable_data();
      py::gil_scoped_release l;

      Delta::inverseDeltaSIMD(buff, inputSize);
    }, py::arg("input"), py::arg("inputSize"),
    "In-place computation inversion of delta4.\n"
    "Using delta4 and prefixSum4 increases space usage, but processing is faster.\n\n"
    "Parameters\n"
    "----------\n"
    "input: input numpy C-style contiguous array to be uncompressed, e.g.:\n"
    "     input = numpy.array(range(256), dtype = np.uint32).ravel()\n"
    "inputSize: a number of integers to process\n"
    "\n"
    "Returns\n"
    "----------\n"
    "    None"
  );

#ifndef PYBIND11_MODULE
  return m.ptr();
#endif
}

void exportCodecs(py::module& m) {
  py::class_<IntegerCODECWrapper>(m, "IntegerCODEC")
  .def("encodeArray", &IntegerCODECWrapper::encodeArray,
      py::arg("input"), py::arg("inputSize"),
      py::arg("output"), py::arg("outputSize"),
      "Compress input array.\n\n"
      "Parameters\n"
      "----------\n"
      "input: numpy C-style contiguous array to be compressed, e.g.:\n"
      "     input = numpy.array(range(256), dtype = np.uint32).ravel()\n"
      "inputSize: a number of integers to compress: it can be less than\n"
      "     than the total number of integers in the numpy array.\n"
      "output: numpy C-style contiguous array with compressed data, e.g.:\n"
      "     output = np.zeros(buffSize, dtype = np.uint32).ravel()\n"
      "outputSize: a capacity of the output buffer: it can be less than\n"
      "     the total number of integers in the numpy array.\n"
      "\n"
      "Returns\n"
      "----------\n"
      "     A number of integers in the compressed output.")
  .def("decodeArray", &IntegerCODECWrapper::decodeArray,
      py::arg("input"), py::arg("inputSize"),
      py::arg("output"), py::arg("outputSize"),
      "Uncompress input array.\n\n"
      "Parameters\n"
      "----------\n"
      "input: numpy C-style contiguous array to be uncompressed, e.g.:\n"
      "     input = numpy.array(range(256), dtype = np.uint32).ravel()\n"
      "inputSize: a number of integers to compress: it can be less than\n"
      "     than the total number of integers in the numpy array.\n"
      "output: numpy C-style contiguous array with compressed data, e.g.:\n"
      "     output = np.zeros(buffSize, dtype = np.uint32).ravel()\n"
      "outputSize: a capacity of the output buffer: it can be less than\n"
      "     the total number of integers in the numpy array.\n"
      "\n"
      "Returns\n"
      "----------\n"
      "     A number of integers in the decompressed output."
      )
  ;
}
