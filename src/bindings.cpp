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
// 日志级别枚举
enum LogLevel
{
    NONE = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4
};

struct BungeePy
{
    BungeePy(int sample_rate, int channels, double speed = 1.0, double pitch = 1.0,
             int log2_synthesis_hop_adjust = -1)
        : stretcher({sample_rate, sample_rate}, channels, log2_synthesis_hop_adjust),
          stream(stretcher, sample_rate, channels),
          speed_param(speed),
          pitch_param(pitch),
          channels(channels),
          sample_rate(sample_rate),
          input_channel_pointers_cache(channels),
          output_channel_pointers_cache(channels),
          log_level(ERROR)
    {
        // 初始化缓冲池
        int buffer_size = sample_rate * channels; // 1秒音频的缓冲大小
        deinterleaved_buffer_pool.resize(buffer_size);
        output_buffer_pool.resize(buffer_size);
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

            // 确保缓冲池足够大
            size_t required_buffer_size = totalInputFramesForStream * channels;
            if (deinterleaved_buffer_pool.size() < required_buffer_size)
            {
                deinterleaved_buffer_pool.resize(required_buffer_size);
            }

            // 使用缓冲池创建非交错输入缓冲区
            float *deinterleaved_input = deinterleaved_buffer_pool.data();
            for (py::ssize_t i = 0; i < totalInputFramesForStream; ++i)
            {
                for (int ch = 0; ch < channels; ++ch)
                {
                    deinterleaved_input[ch * totalInputFramesForStream + i] = chunk_data[i * channels + ch];
                }
            }

            // 使用缓存的指针
            for (int ch = 0; ch < channels; ++ch)
            {
                input_channel_pointers_cache[ch] = deinterleaved_input + ch * totalInputFramesForStream;
            }

            // Calculate ideal_output_frames
            double ideal_output_frames_double = 0;
            if (speed_param != 0)
            {
                ideal_output_frames_double = static_cast<double>(totalInputFramesForStream) / speed_param; // Assuming output sample rate is same as input
            }
            py::ssize_t ideal_output_frames = static_cast<py::ssize_t>(std::ceil(ideal_output_frames_double));

            // 确保输出缓冲池足够大
            size_t required_output_size = ideal_output_frames * channels;
            if (output_buffer_pool.size() < required_output_size)
            {
                output_buffer_pool.resize(required_output_size);
            }

            // 使用缓冲池
            float *output_buffer = output_buffer_pool.data();
            size_t channel_stride_output = ideal_output_frames;

            for (int ch = 0; ch < channels; ++ch)
            {
                output_channel_pointers_cache[ch] = output_buffer + ch * channel_stride_output;
            }

            size_t output_frames_count = stream.process(
                input_channel_pointers_cache.data(),
                output_channel_pointers_cache.data(),
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
        stretcher.enableInstrumentation(enable);
    }

    bool getDebug() const
    {
        return debug_enabled;
    }

    // 动态设置速度参数
    void set_speed(double speed)
    {
        if (speed == 0.0)
            throw std::runtime_error("Speed cannot be zero");
        speed_param = speed;
    }

    // 动态设置音高参数
    void set_pitch(double pitch)
    {
        pitch_param = pitch;
    }

    // 获取当前处理延迟（以样本数计）
    double get_latency() const
    {
        return stream.latency();
    }

    // preroll 支持，预处理以获得更好的音频质量
    void preroll()
    {
        Bungee::Request request;
        request.position = 0;
        request.speed = speed_param;
        request.pitch = pitch_param;
        request.reset = true;
        stretcher.preroll(request);
    }

    // 高级功能：时间伸缩（保持音高）
    py::array_t<float> time_stretch(py::array_t<float> input, double stretch_factor)
    {
        double original_speed = speed_param;
        try
        {
            set_speed(1.0 / stretch_factor);
            auto result = process(input);
            set_speed(original_speed);
            return result;
        }
        catch (const std::exception &e)
        {
            set_speed(original_speed);
            throw;
        }
    }

    // 高级功能：音高变换（保持时长）
    py::array_t<float> pitch_shift(py::array_t<float> input, double semitones)
    {
        double original_pitch = pitch_param;
        try
        {
            // 音高变化计算公式: 2^(半音数/12)
            set_pitch(std::pow(2.0, semitones / 12.0));
            auto result = process(input);
            set_pitch(original_pitch);
            return result;
        }
        catch (const std::exception &e)
        {
            set_pitch(original_pitch);
            throw;
        }
    }

    // 设置日志级别
    void set_log_level(LogLevel level)
    {
        log_level = level;
        stretcher.enableInstrumentation(level >= INFO);
    }

    // 获取当前日志级别
    LogLevel get_log_level() const
    {
        return log_level;
    }

    // 记录日志信息
    void log(LogLevel level, const std::string &message)
    {
        if (level <= log_level)
        {
            std::string prefix;
            switch (level)
            {
            case ERROR:
                prefix = "[ERROR] ";
                break;
            case WARN:
                prefix = "[WARN] ";
                break;
            case INFO:
                prefix = "[INFO] ";
                break;
            case DEBUG:
                prefix = "[DEBUG] ";
                break;
            default:
                break;
            }
            std::cout << prefix << message << std::endl;
        }
    }

private:
    StretcherBasic stretcher;
    Bungee::Stream<Bungee::Basic> stream;
    double speed_param;
    double pitch_param;
    int channels;
    bool debug_enabled = false;
    int sample_rate;
    LogLevel log_level;

    // 内存缓冲池用于减少频繁的内存分配
    std::vector<float> deinterleaved_buffer_pool;
    std::vector<float> output_buffer_pool;

    // 输入通道指针和输出通道指针缓存，避免重复分配
    std::vector<const float *> input_channel_pointers_cache;
    std::vector<float *> output_channel_pointers_cache;
};

PYBIND11_MODULE(bungee, m)
{
    // 定义 LogLevel 枚举
    py::enum_<LogLevel>(m, "LogLevel")
        .value("NONE", NONE)
        .value("ERROR", ERROR)
        .value("WARN", WARN)
        .value("INFO", INFO)
        .value("DEBUG", DEBUG)
        .export_values();

    py::class_<BungeePy>(m, "Bungee")
        .def(py::init<int, int, double, double, int>(), // Removed preroll_scale from init
             py::arg("sample_rate"),
             py::arg("channels"),
             py::arg("speed") = 1.0,
             py::arg("pitch") = 1.0,
             py::arg("log2_synthesis_hop_adjust") = -1)
        .def("process", &BungeePy::process, py::arg("input"))
        .def("set_debug", &BungeePy::set_debug, py::arg("enabled"))
        .def("get_debug", &BungeePy::getDebug)
        .def("set_speed", &BungeePy::set_speed, py::arg("speed"))
        .def("set_pitch", &BungeePy::set_pitch, py::arg("pitch"))
        .def("get_latency", &BungeePy::get_latency)
        .def("preroll", &BungeePy::preroll)
        .def("time_stretch", &BungeePy::time_stretch, py::arg("input"), py::arg("stretch_factor"))
        .def("pitch_shift", &BungeePy::pitch_shift, py::arg("input"), py::arg("semitones"))
        .def("set_log_level", &BungeePy::set_log_level, py::arg("level"))
        .def("get_log_level", &BungeePy::get_log_level)
        .def("log", &BungeePy::log, py::arg("level"), py::arg("message"));
}
