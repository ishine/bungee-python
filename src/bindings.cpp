#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bungee/Bungee.h"
#include "bungee/Stream.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <string>
#include <iostream>

namespace py = pybind11;

class BungeePy
{
public:
    BungeePy(int sample_rate, int channels)
        : sample_rate_(sample_rate), channels_(channels),
          stretcher_(std::make_unique<Bungee::Stretcher<Bungee::Basic>>(Bungee::SampleRates{sample_rate, sample_rate}, channels)),
          speed_(1.0), pitch_(1.0), instrumentation_(false)
    {
        stretcher_->enableInstrumentation(false);
    }

    void set_speed(double speed)
    {
        speed_ = speed;
    }
    void set_pitch(double pitch)
    {
        pitch_ = pitch;
    }
    void set_instrumentation(bool enable)
    {
        instrumentation_ = enable;
        stretcher_->enableInstrumentation(enable);
    }

    py::array_t<float> process(py::array_t<float, py::array::c_style | py::array::forcecast> input_py_array)
    {
        if (input_py_array.ndim() != 2)
            throw std::runtime_error("Input must be 2D array (frames, channels)");
        if (input_py_array.shape(1) != channels_)
            throw std::runtime_error("Input channel count mismatch");

        ssize_t total_input_frames = input_py_array.shape(0);
        if (total_input_frames == 0)
        {
            return py::array_t<float>(std::vector<ssize_t>{0, static_cast<ssize_t>(channels_)});
        }

        auto input_py_unchecked = input_py_array.unchecked<2>();

        Bungee::Stream<Bungee::Basic> stream(*stretcher_, 4096, channels_);
        const int max_block_size = 4096;

        std::vector<std::vector<float>> input_chunk_cpp(channels_, std::vector<float>(max_block_size));

        double current_processing_speed = std::max(std::abs(speed_), 1e-9);
        ssize_t output_buffer_capacity_for_chunk = static_cast<ssize_t>(std::ceil(static_cast<double>(max_block_size) / current_processing_speed)) + 10;
        if (output_buffer_capacity_for_chunk <= 0)
        {
            output_buffer_capacity_for_chunk = max_block_size * 4; // Default safety net
        }

        std::vector<std::vector<float>> output_chunk_cpp(channels_, std::vector<float>(output_buffer_capacity_for_chunk));

        std::vector<const float *> input_chunk_ptrs(channels_);
        std::vector<float *> output_chunk_ptrs(channels_);
        for (int c = 0; c < channels_; ++c)
        {
            input_chunk_ptrs[c] = input_chunk_cpp[c].data();
            output_chunk_ptrs[c] = output_chunk_cpp[c].data();
        }

        std::vector<std::vector<float>> full_output_channels(channels_);
        ssize_t current_input_frame_offset = 0;

        while (current_input_frame_offset < total_input_frames)
        {
            ssize_t current_chunk_input_frames = std::min(static_cast<ssize_t>(max_block_size), total_input_frames - current_input_frame_offset);

            for (int c = 0; c < channels_; ++c)
            {
                for (ssize_t f = 0; f < current_chunk_input_frames; ++f)
                {
                    input_chunk_cpp[c][f] = input_py_unchecked(current_input_frame_offset + f, c);
                }
            }

            double ideal_output_frames_for_this_chunk = static_cast<double>(current_chunk_input_frames) / current_processing_speed;
            // Ensure the output buffer for the chunk is large enough.
            // output_chunk_cpp was already sized based on max_block_size and speed.
            // ideal_output_frames_for_this_chunk will be <= output_buffer_capacity_for_chunk.

            int processed_frames_in_chunk = stream.process(
                input_chunk_ptrs.data(), output_chunk_ptrs.data(),
                static_cast<int>(current_chunk_input_frames),
                ideal_output_frames_for_this_chunk,
                pitch_);

            if (processed_frames_in_chunk < 0)
            {
                throw std::runtime_error("Bungee stream.process returned negative frame count.");
            }
            // Check if Bungee wrote more than the buffer capacity (it shouldn't if ideal_output_frames_for_this_chunk is respected)
            if (static_cast<ssize_t>(processed_frames_in_chunk) > output_buffer_capacity_for_chunk)
            {
                throw std::runtime_error("Bungee stream.process wrote more frames than output buffer capacity for chunk.");
            }

            for (int c = 0; c < channels_; ++c)
            {
                full_output_channels[c].insert(full_output_channels[c].end(), output_chunk_cpp[c].begin(), output_chunk_cpp[c].begin() + processed_frames_in_chunk);
            }

            current_input_frame_offset += current_chunk_input_frames;
        }

        ssize_t final_total_output_frames = 0;
        if (channels_ > 0 && !full_output_channels[0].empty())
        {
            final_total_output_frames = full_output_channels[0].size();
        }

        py::array_t<float> final_output_py({final_total_output_frames, static_cast<ssize_t>(channels_)});
        if (final_total_output_frames > 0)
        {
            auto final_out_py_unchecked = final_output_py.mutable_unchecked<2>();
            for (int c = 0; c < channels_; ++c)
            {
                for (ssize_t f = 0; f < final_total_output_frames; ++f)
                {
                    final_out_py_unchecked(f, c) = full_output_channels[c][f];
                }
            }
        }
        return final_output_py;
    }

    double get_speed() const { return speed_; }
    double get_pitch() const { return pitch_; }
    bool get_instrumentation() const { return instrumentation_; }

private:
    int sample_rate_;
    int channels_;
    double speed_;
    double pitch_;
    bool instrumentation_;
    std::unique_ptr<Bungee::Stretcher<Bungee::Basic>> stretcher_;
};

PYBIND11_MODULE(bungee, m)
{
    py::class_<BungeePy>(m, "Bungee")
        .def(py::init<int, int>(), py::arg("sample_rate"), py::arg("channels"))
        .def_property("speed", &BungeePy::get_speed, &BungeePy::set_speed)
        .def_property("pitch", &BungeePy::get_pitch, &BungeePy::set_pitch)
        .def_property("instrumentation", &BungeePy::get_instrumentation, &BungeePy::set_instrumentation)
        .def("set_speed", &BungeePy::set_speed)
        .def("set_pitch", &BungeePy::set_pitch)
        .def("process", &BungeePy::process, py::arg("input"), R"pbdoc(
            处理音频: 输入float32二维数组 (frames, channels), 返回处理后音频数组
        )pbdoc");
}
