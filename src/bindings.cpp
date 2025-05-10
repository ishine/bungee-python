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
        py::ssize_t frames = buf.shape[0];
        py::ssize_t nch_input = buf.shape[1];
        if (nch_input != channels)
            throw std::runtime_error("Expected " + std::to_string(channels) + " channels, got " + std::to_string(nch_input));
        const float *data = static_cast<const float *>(buf.ptr);

        std::vector<float> output_flat;
        // Reserve space more accurately, considering speed and potential flush
        output_flat.reserve(static_cast<size_t>((frames / speed_param) * channels + sample_rate * channels * 2)); // Add buffer for 2s for flush

        // 预先准备处理器，但不要在此阶段访问stream.latency
        {
            Bungee::Request _req{};
            _req.position = 0;
            _req.speed = speed_param;
            _req.pitch = pitch_param;
            _req.reset = true;
            stretcher.preroll(_req);
        }

        // 我们只有在至少处理了一帧数据后才能安全地访问latency
        py::ssize_t chunkSize = sample_rate; // Process 1 second of audio at a time
        py::ssize_t offset = 0;

        // 用于计算前导静音的变量
        double initial_input_latency_samples = 0;
        py::ssize_t leading_silence_to_trim_output_frames = 0;
        bool first_process_done = false;

        // Main processing loop for actual input data
        while (offset < frames)
        {
            py::ssize_t this_frames = std::min(chunkSize, frames - offset);
            const float *chunk_data = data + offset * channels;
            py::ssize_t totalInputFramesForStream = this_frames;

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
            size_t channel_stride_output = ideal_output_frames > 0 ? ideal_output_frames : 1; // Avoid stride 0

            for (int ch = 0; ch < channels; ++ch)
            {
                output_channel_pointers_cache[ch] = output_buffer + ch * channel_stride_output;
            }

            size_t output_frames_count = stream.process(
                input_channel_pointers_cache.data(),
                output_channel_pointers_cache.data(),
                totalInputFramesForStream,
                ideal_output_frames,
                this->pitch_param); 

            if (!first_process_done) {
                initial_input_latency_samples = stream.latency();
                if (speed_param != 0) {
                    leading_silence_to_trim_output_frames = static_cast<py::ssize_t>(std::ceil(initial_input_latency_samples / speed_param));
                }
                if (debug_enabled) {
                    log(DEBUG, "Initial input latency (samples): " + std::to_string(initial_input_latency_samples));
                    log(DEBUG, "Frames to trim from start: " + std::to_string(leading_silence_to_trim_output_frames));
                }
                first_process_done = true;
            }

            if (debug_enabled)
            {
                log(DEBUG, "Processed chunk. Offset: " + std::to_string(offset) +
                          ", Input frames: " + std::to_string(this_frames) +
                          ", Ideal output: " + std::to_string(ideal_output_frames) +
                          ", Actual output frames: " + std::to_string(output_frames_count) +
                          ", Current latency (input samples): " + std::to_string(stream.latency()));
            }

            for (size_t i = 0; i < output_frames_count; ++i)
            {
                for (int ch = 0; ch < channels; ++ch)
                {
                    output_flat.push_back(output_channel_pointers_cache[ch][i]);
                }
            }
            offset += this_frames;
        }

        // Flushing loop: process silent input to get remaining buffered audio
        if (debug_enabled) {
            log(DEBUG, "Flushing stream... Current latency (input samples): " + std::to_string(stream.latency()));
        }
        int consecutive_zero_output_count = 0;
        py::ssize_t total_silent_input_for_flush_estimate = static_cast<py::ssize_t>(std::ceil(stream.latency() > 0 ? stream.latency() : sample_rate / 2.0)); // Estimate based on current latency or half a second
        if (frames == 0 && initial_input_latency_samples > 0) { // If only preroll was called, flush based on initial latency
             total_silent_input_for_flush_estimate = static_cast<py::ssize_t>(std::ceil(initial_input_latency_samples));
        }


        py::ssize_t flushed_silent_input_total = 0;
        py::ssize_t flush_chunk_size = chunkSize / 4; 
        if (flush_chunk_size == 0) flush_chunk_size = 128;


        while(true) { 
            bool should_break = false;
            // Break if latency is very low AND we've had a few consecutive zero outputs
            if (stream.latency() < flush_chunk_size / 4.0 && consecutive_zero_output_count >= 3) {
                 if (debug_enabled) log(DEBUG, "Flush deemed complete: low latency and multiple zero outputs.");
                should_break = true;
            }
            // Safety break: if we've fed much more silent input than estimated necessary
            if (flushed_silent_input_total > total_silent_input_for_flush_estimate * 3 && total_silent_input_for_flush_estimate > 0) {
                 if (debug_enabled) log(DEBUG, "Flush safety break: processed much more silent input than initial latency estimate for flush.");
                 should_break = true;
            }
            // Absolute safety break after feeding a large amount of silent input (e.g., 5 seconds worth)
            if (flushed_silent_input_total > sample_rate * 5) { 
                if (debug_enabled) log(WARN, "Flush safety break: Max silent input (5s) reached during flush.");
                should_break = true;
            }

            if (should_break) break;

            py::ssize_t silent_input_frames_this_flush = flush_chunk_size;
            
            double ideal_flush_output_frames_double = 0;
            if (speed_param != 0) {
                ideal_flush_output_frames_double = static_cast<double>(silent_input_frames_this_flush) / speed_param;
            }
            py::ssize_t ideal_flush_output_frames = static_cast<py::ssize_t>(std::ceil(ideal_flush_output_frames_double));
            if (ideal_flush_output_frames == 0 && silent_input_frames_this_flush > 0 && speed_param > 0) ideal_flush_output_frames = 1; // Ensure at least 1 if expecting output


            size_t required_flush_output_size = ideal_flush_output_frames * channels;
            if (output_buffer_pool.size() < required_flush_output_size) {
                output_buffer_pool.resize(required_flush_output_size);
            }
            float* flush_output_buffer_ptr = output_buffer_pool.data();
            size_t flush_channel_stride_output = ideal_flush_output_frames > 0 ? ideal_flush_output_frames : 1; 
            for (int ch = 0; ch < channels; ++ch) {
                output_channel_pointers_cache[ch] = flush_output_buffer_ptr + ch * flush_channel_stride_output;
            }

            size_t output_frames_count_flush = stream.process(
                nullptr, // Silent input
                output_channel_pointers_cache.data(),
                silent_input_frames_this_flush,
                ideal_flush_output_frames,
                this->pitch_param);

            if (debug_enabled) {
                log(DEBUG, "Flushing: silent_input=" + std::to_string(silent_input_frames_this_flush) +
                          ", ideal_output=" + std::to_string(ideal_flush_output_frames) +
                          ", actual_output=" + std::to_string(output_frames_count_flush) +
                          ", current_latency(input samples)=" + std::to_string(stream.latency()));
            }

            if (output_frames_count_flush > 0) {
                for (size_t i = 0; i < output_frames_count_flush; ++i) {
                    for (int ch = 0; ch < channels; ++ch) {
                        output_flat.push_back(output_channel_pointers_cache[ch][i]);
                    }
                }
                consecutive_zero_output_count = 0;
            } else {
                consecutive_zero_output_count++;
            }
            flushed_silent_input_total += silent_input_frames_this_flush;
        }
        if (debug_enabled) {
             log(DEBUG, "Flush finished. Total silent input for flush: " + std::to_string(flushed_silent_input_total));
        }

        // Trim leading silence from output_flat
        py::ssize_t final_output_frames_in_flat = output_flat.size() / channels;
        py::ssize_t frames_to_skip = leading_silence_to_trim_output_frames;

        // 计算理论输出帧数（用于截断多余尾部静音）
        py::ssize_t expected_output_frames = 0;
        if (speed_param > 0) {
            expected_output_frames = static_cast<py::ssize_t>(std::ceil(static_cast<double>(frames) / speed_param));
        }
        // 先裁剪前导静音，再截断到理论输出帧数
        py::ssize_t available_frames = final_output_frames_in_flat - frames_to_skip;
        py::ssize_t max_valid_output_frames = std::min(available_frames, expected_output_frames);
        if (max_valid_output_frames < 0) max_valid_output_frames = 0;

        if (debug_enabled) {
            log(DEBUG, "Total output frames in flat vector before trim: " + std::to_string(final_output_frames_in_flat));
            log(DEBUG, "Frames to skip from start (leading silence): " + std::to_string(frames_to_skip));
            log(DEBUG, "Expected output frames: " + std::to_string(expected_output_frames));
            log(DEBUG, "Final result frames after trim and trunc: " + std::to_string(max_valid_output_frames));
        }

        float* start_of_data_to_copy = output_flat.data() + frames_to_skip * channels;
        py::array_t<float> result({max_valid_output_frames, (py::ssize_t)channels});
        if (max_valid_output_frames > 0) {
            std::memcpy(result.mutable_data(), start_of_data_to_copy, max_valid_output_frames * channels * sizeof(float));
        }
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
