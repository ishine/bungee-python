#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bungee/Bungee.h"
#include "bungee/Stream.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

// BungeePy wrapper class
typedef Bungee::Stretcher<Bungee::Basic> StretcherBasic;
struct BungeePy
{
    BungeePy(int sample_rate, int channels, double speed = 1.0, double pitch = 1.0,
             int log2_synthesis_hop_adjust = -1) // Removed preroll_scale
        : stretcher({sample_rate, sample_rate}, channels, log2_synthesis_hop_adjust),
          stream(stretcher, sample_rate, channels), 
          speed_param(speed),
          pitch_param(pitch),
          channels(channels),
          sample_rate(sample_rate)
    {
        // All preroll and initialSilenceOutputSamples logic has been removed.
    }

    py::array_t<float> process(py::array_t<float> input)
    {
        auto buf = input.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Input must be a 2D NumPy array");
        py::ssize_t frames = buf.shape[0]; // Use py::ssize_t for shapes
        py::ssize_t nch_input = buf.shape[1];
        if (nch_input != channels)
            throw std::runtime_error("Expected " + std::to_string(channels) + " channels, got " + std::to_string(nch_input));
        const float *data = static_cast<const float *>(buf.ptr);

        std::vector<float> output_flat;
        output_flat.reserve((size_t)(frames / speed_param + 10) * channels);

        // 分块处理，避免输入长度超过内部缓冲区
        py::ssize_t chunkSize = sample_rate;
        py::ssize_t offset = 0;
        while (offset < frames)
        {
            py::ssize_t this_frames = std::min(chunkSize, frames - offset);
            const float *chunk_data = data + offset * channels;

            py::ssize_t totalInputFramesForStream = this_frames;
            // All logic related to inputPaddingFrames and initialSilenceOutputSamples removed.

            // Create deinterleaved input buffer
            std::vector<float> deinterleaved_input_buffer(totalInputFramesForStream * channels);
            for (py::ssize_t i = 0; i < totalInputFramesForStream; ++i) {
                for (int ch = 0; ch < channels; ++ch) {
                    deinterleaved_input_buffer[ch * totalInputFramesForStream + i] = chunk_data[i * channels + ch];
                }
            }

            std::vector<const float *> inputChannelPointers(channels);
            std::vector<float *> outputChannelPointers(channels);

            for (int ch = 0; ch < channels; ++ch)
            {
                inputChannelPointers[ch] = deinterleaved_input_buffer.data() + ch * totalInputFramesForStream;
            }

            // Calculate ideal_output_frames
            double ideal_output_frames_double = 0;
            if (speed_param != 0) {
                 ideal_output_frames_double = static_cast<double>(totalInputFramesForStream) / speed_param; // Assuming output sample rate is same as input
            }
            py::ssize_t ideal_output_frames = static_cast<py::ssize_t>(std::ceil(ideal_output_frames_double));

            std::vector<float> output_buffer(ideal_output_frames * channels);
            size_t channel_stride_output = ideal_output_frames;

            for (int ch = 0; ch < channels; ++ch)
            {
                outputChannelPointers[ch] = output_buffer.data() + ch * channel_stride_output;
            }

            size_t output_frames_count = stream.process(
                inputChannelPointers.data(),
                outputChannelPointers.data(),
                totalInputFramesForStream, 
                ideal_output_frames,
                this->pitch_param); // Use the member variable pitch_param

            // All logic related to outputStartIndex and initialSilenceOutputSamples removed.

            if (debug_enabled)
            {
                std::cout << "Processed chunk. Offset: " << offset
                          << ", Input frames: " << this_frames
                          << ", Output frames: " << output_frames_count << std::endl;
            }

            for (size_t i = 0; i < output_frames_count; ++i)
            {
                for (int ch = 0; ch < channels; ++ch)
                {
                    output_flat.push_back(output_buffer[ch * channel_stride_output + i]);
                }
            }

            offset += this_frames;
        }

        // 构造最终 NumPy 数组
        py::ssize_t totalOutputFrames = output_flat.size() / channels;
        py::array_t<float> result({totalOutputFrames, (py::ssize_t)channels});
        std::memcpy(result.mutable_data(), output_flat.data(), output_flat.size() * sizeof(float));
        return result;
    }

    // 设置/获取调试标志
    void set_debug(bool enable)
    {
        debug_enabled = enable;
    }

    bool getDebug() const
    {
        return debug_enabled;
    }

private:
    StretcherBasic stretcher;
    Bungee::Stream<Bungee::Basic> stream;
    double speed_param;
    double pitch_param;
    int channels;
    bool debug_enabled = true;
    int sample_rate;
};

PYBIND11_MODULE(bungee, m)
{
    py::class_<BungeePy>(m, "Bungee")
        .def(py::init<int, int, double, double, int>(), // Removed preroll_scale from init
             py::arg("sample_rate"),
             py::arg("channels"),
             py::arg("speed") = 1.0,
             py::arg("pitch") = 1.0,
             py::arg("log2_synthesis_hop_adjust") = -1)
        .def("process", &BungeePy::process, py::arg("input"))
        .def("set_debug", &BungeePy::set_debug, py::arg("enabled"));
}
