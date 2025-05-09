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
             int log2_synthesis_hop_adjust = -1, double preroll_scale = 1.0)
        : stretcher({sample_rate, sample_rate}, channels, log2_synthesis_hop_adjust),
          stream(stretcher, sample_rate, channels), // sample_rate is used as maxInputSampleCount for Stream's InputBuffer
          speed_param(speed),
          pitch_param(pitch),
          channels(channels),
          preroll_scale(preroll_scale),
          sample_rate(sample_rate) // 存储采样率，用于分块
    {

        Bungee::Request prerollReq;
        prerollReq.position = 0; // Nominal start position for calculation
        prerollReq.speed = speed_param;
        prerollReq.pitch = pitch_param;
        prerollReq.reset = true;

        stretcher.preroll(prerollReq); // This modifies prerollReq.position to an earlier time

        // Calculate the number of input samples effectively processed during preroll
        // If prerollReq.position became -N, it means N input samples are for preroll.
        double inputSamplesForPreroll = -prerollReq.position;
        
        // 应用预填充缩放系数，增加预填充帧数
        inputSamplesForPreroll *= preroll_scale;  // 使用缩放系数放大预填充帧数
        
        if (debug_enabled) {
            std::cout << "初始 prerollReq.position: " << prerollReq.position << std::endl;
            std::cout << "原始 inputSamplesForPreroll: " << (-prerollReq.position) << std::endl;
            std::cout << "应用缩放系数后 inputSamplesForPreroll: " << inputSamplesForPreroll << std::endl;
            std::cout << "预填充缩放系数: " << preroll_scale << std::endl;
        }
        
        if (inputSamplesForPreroll < 0)
        {
            inputSamplesForPreroll = 0; // Should be positive or zero
            if (debug_enabled) {
                std::cout << "调整后的 inputSamplesForPreroll (负值调整为0): " << inputSamplesForPreroll << std::endl;
            }
        }

        // Convert preroll input samples to output samples based on the speed parameter
        if (speed_param == 0)
        { // Avoid division by zero; though speed 0 is problematic anyway
            this->initialSilenceOutputSamples = 0;
            if (debug_enabled) {
                std::cout << "速度为0，initialSilenceOutputSamples 设置为 0" << std::endl;
            }
        }
        else
        {
            this->initialSilenceOutputSamples = static_cast<int>(std::round(inputSamplesForPreroll / speed_param));
            if (debug_enabled) {
                std::cout << "speed_param: " << speed_param << std::endl;
                std::cout << "计算得到的 initialSilenceOutputSamples: " << initialSilenceOutputSamples << std::endl;
            }
        }
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
        while (offset < frames) {
            py::ssize_t this_frames = std::min(chunkSize, frames - offset);
            const float *chunk_data = data + offset * channels;
            
            py::ssize_t totalInputFramesForStream = this_frames;
            py::ssize_t inputPaddingFrames = 0; // Frames taken from current input for padding
            std::vector<std::vector<float>> in_storage; // To hold padded input if needed

            if (!processedFirstChunk && initialSilenceOutputSamples > 0 && this_frames > 0) {
                // We are on the first chunk, padding is needed, and we have audio data to use for padding.
                // Calculate how many *input* frames would ideally be needed to generate
                // the 'initialSilenceOutputSamples' (target output to skip) at the current speed.
                py::ssize_t idealInputPaddingFrames = static_cast<py::ssize_t>(std::ceil(initialSilenceOutputSamples * speed_param));
                
                // We can only use up to 'this_frames' from the current input for this padding.
                inputPaddingFrames = std::min(idealInputPaddingFrames, this_frames); 

                totalInputFramesForStream = this_frames + inputPaddingFrames; // Total frames to feed to Bungee
                
                if (debug_enabled) {
                    std::cout << "First chunk: Applying audio pre-padding." << std::endl;
                    std::cout << "  Target output skip (initialSilenceOutputSamples): " << initialSilenceOutputSamples << std::endl;
                    std::cout << "  speed_param: " << speed_param << std::endl;
                    std::cout << "  Calculated idealInputPaddingFrames (initialSilenceOutputSamples * speed_param): " << idealInputPaddingFrames << std::endl;
                    std::cout << "  Frames available in current input: " << this_frames << std::endl;
                    std::cout << "  Actual inputPaddingFrames to use (min of ideal and available): " << inputPaddingFrames << std::endl;
                    std::cout << "  Total input frames for stream (frames + actualInputPaddingFrames): " << totalInputFramesForStream << std::endl;
                }

                in_storage.resize(channels, std::vector<float>(totalInputFramesForStream));
                for (int c = 0; c < channels; ++c) {
                    // Main data: copy entire chunk after padding (silence region is already zero)
                    for (py::ssize_t i = 0; i < this_frames; ++i) {
                        in_storage[c][i + inputPaddingFrames] = chunk_data[i * channels + c];
                    }
                }
                processedFirstChunk = true;
            } else {
                // Not the first chunk, or no padding needed (initialSilenceOutputSamples <= 0), 
                // or no input frames (frames == 0).
                // totalInputFramesForStream remains 'this_frames'.
                // inputPaddingFrames remains 0.
                // Just deinterleave the input as is.
                in_storage.resize(channels, std::vector<float>(this_frames)); // size is just 'this_frames'
                for (py::ssize_t i = 0; i < this_frames; ++i) {
                    for (int c = 0; c < channels; ++c) {
                        in_storage[c][i] = chunk_data[i * channels + c];
                    }
                }
                if (debug_enabled) {
                     if (processedFirstChunk) { // Subsequent chunk
                        std::cout << "Subsequent chunk: No pre-padding. totalInputFramesForStream: " << totalInputFramesForStream << std::endl;
                     } else { // First chunk but no padding applied (or frames == 0)
                        std::cout << "First chunk but no padding applied (initialSilenceOutputSamples=" << initialSilenceOutputSamples 
                                  << " or frames=" << this_frames << "). totalInputFramesForStream: " << totalInputFramesForStream << std::endl;
                     }
                }
            }
            
            std::vector<const float *> in_ptrs(channels);
            for (int c = 0; c < channels; ++c)
                in_ptrs[c] = in_storage[c].data();

            // Calculate outputSampleCountTarget based on totalInputFramesForStream
            double outputSampleCountTarget = 0.0;
            if (std::abs(speed_param) > 1e-9) { // Avoid division by zero or very small numbers
                 outputSampleCountTarget = static_cast<double>(totalInputFramesForStream) / speed_param;
            }

            // Make out_cap robust for speed_param = 0 or very small totalInputFramesForStream
            int out_cap = static_cast<int>(std::ceil(std::abs(outputSampleCountTarget))) + 1024; 
            if (totalInputFramesForStream == 0 && out_cap < 1024) out_cap = 1024; // Ensure some capacity even if no input

            std::vector<std::vector<float>> out(channels, std::vector<float>(out_cap));
            std::vector<float *> out_ptrs(channels);
            for (int c = 0; c < channels; ++c)
                out_ptrs[c] = out[c].data();

            int total_out_frames = stream.process(in_ptrs.data(), out_ptrs.data(), static_cast<int>(totalInputFramesForStream), outputSampleCountTarget, pitch_param);
            
            py::ssize_t outputStartIndex = 0;
            if (inputPaddingFrames > 0) {
                // This means we *did* prepend audio from the input in this call.
                // We want to skip the portion of the output that corresponds to the *original preroll requirement* (initialSilenceOutputSamples).
                outputStartIndex = initialSilenceOutputSamples; 

                if (outputStartIndex > total_out_frames) {
                    if (debug_enabled) {
                        std::cout << "Warning: outputStartIndex (" << outputStartIndex 
                                  << ") > total_out_frames (" << total_out_frames 
                                  << "). Clamping outputStartIndex." << std::endl;
                    }
                    outputStartIndex = total_out_frames; // Cannot skip more than available
                }
                
                if (debug_enabled) {
                    std::cout << "Audio pre-padding was applied from input." << std::endl;
                    std::cout << "  Target output skip (initialSilenceOutputSamples): " << initialSilenceOutputSamples << std::endl;
                    std::cout << "  Actual outputStartIndex for trimming: " << outputStartIndex << std::endl;
                    std::cout << "  Total output frames from stream: " << total_out_frames << std::endl;
                }
            } else if (debug_enabled) {
                // This covers subsequent chunks OR the first chunk where no padding was applied.
                std::cout << "No audio pre-padding applied in this call (inputPaddingFrames=0), or subsequent chunk. outputStartIndex: 0" << std::endl;
            }
            
            // Calculate effective output frames
            py::ssize_t effectiveOutFrames = total_out_frames - outputStartIndex;
            if (effectiveOutFrames < 0) effectiveOutFrames = 0;
            
            if (debug_enabled) {
                std::cout << "Final effectiveOutFrames (total_out_frames - outputStartIndex): " << effectiveOutFrames << std::endl;
            }

            // 将 out[c][i + outputStartIndex] 按通道展平追加到 output_flat
            for (py::ssize_t i = 0; i < effectiveOutFrames; ++i) {
                for (int c = 0; c < channels; ++c) {
                    output_flat.push_back(out[c][i + outputStartIndex]);
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
    void setDebug(bool enable) {
        debug_enabled = enable;
    }

    bool getDebug() const {
        return debug_enabled;
    }
    
private:
    StretcherBasic stretcher;
    Bungee::Stream<Bungee::Basic> stream;
    double speed_param; // Renamed from speed
    double pitch_param; // Renamed from pitch
    int channels;
    int initialSilenceOutputSamples; // Stores the calculated initial silence in output samples
    bool processedFirstChunk = false; // 标记是否已经处理过第一个音频块
    bool debug_enabled = true; // 调试开关，默认开启
    double preroll_scale; // 新增的预填充缩放系数
    int sample_rate; // 新增
};

PYBIND11_MODULE(bungee, m)
{
    py::class_<BungeePy>(m, "Bungee")
        .def(py::init<int, int, double, double, int, double>(),
             py::arg("sample_rate"), py::arg("channels"),
             py::arg("speed") = 1.0, py::arg("pitch") = 1.0,
             py::arg("log2_synthesis_hop_adjust") = -1,
             py::arg("preroll_scale") = 2.0)
        .def("process", &BungeePy::process, py::arg("input"), R"pbdoc(
            Processes audio using Bungee::Stretcher API.
            Input: float32 NumPy array (frames, channels).
            Output: float32 NumPy array (frames, channels).
        )pbdoc");
}
