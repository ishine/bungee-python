#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "bungee/Bungee.h"
#include <vector>    // 包含 vector
#include <stdexcept> // 包含 runtime_error
#include <cmath>     // 包含 NAN
#include <algorithm> // 包含 std::max, std::min

namespace py = pybind11;

// 绑定 Bungee::Stretcher<Bungee::Basic>
class PyBungee
{
public:
    PyBungee(int sample_rate, int channels)
        : stretcher({sample_rate, sample_rate}, channels), channels(channels)
    {
        request.position = 0.0;
        request.speed = 1.0;
        request.pitch = 1.0;
        request.reset = true;       // 初始重置
        stretcher.preroll(request); // 调用 preroll
        request.reset = false;      // preroll 后取消重置标志
    }

    void set_speed(float speed) { request.speed = speed; }
    void set_pitch(float pitch) { request.pitch = pitch; }
    void set_position(float position) { request.position = position; }
    void reset() { request.reset = true; } // 设置重置标志，下次 process 会生效

    // 输入 shape: (frames, channels)，输出 shape: (frames, channels)
    py::array_t<float> process(py::array_t<float, py::array::c_style | py::array::forcecast> input)
    {
        py::buffer_info in_info = input.request();
        if (in_info.ndim != 2 || in_info.shape[1] != channels)
            throw std::runtime_error("输入必须为 shape=(frames, channels) 的二维 float32 数组");

        int total_input_frames = in_info.shape[0];
        const float *in_ptr = static_cast<const float *>(in_info.ptr);

        std::vector<float> output_buffer;
        // 预估输出大小，可以根据 speed 调整，这里简单预留输入大小的两倍
        output_buffer.reserve(total_input_frames * channels * 2);

        int current_input_frame = 0;
        bool input_finished = false;

        // 主处理循环 + Flushing 循环
        while (true)
        {
            // 如果输入数据处理完毕，并且需要 flush，则设置 position 为 NAN
            if (input_finished)
            {
                if (stretcher.isFlushed())
                {
                    break; // Flushing 完成，退出循环
                }
                request.position = NAN; // 设置 NAN 以进行 flushing
                request.reset = false;  // Flushing 时不应重置
            }
            else
            {
                // 正常处理，更新下一 grain 的位置
                // 注意：第一次循环时，request 的 position/reset 由构造函数或 reset() 设置
                // 后续循环由 stretcher.next() 更新
                if (current_input_frame > 0 || request.reset)
                {
                    // 只有在处理过至少一帧或者需要重置时才调用 next
                    // 或者在 reset 后，第一次调用 specifyGrain 前不调用 next
                    // request.reset = false; // 在 specifyGrain 之前处理 reset 标志
                }
            }

            // 1. 指定 Grain 并获取所需输入块
            // 假设输入 NumPy 数组代表从 0 开始的音频
            auto input_chunk = stretcher.specifyGrain(request, 0.0);
            request.reset = false; // specifyGrain 会处理 reset 标志，之后清除它

            // 如果 position 是 NAN (flushing 阶段)，input_chunk 可能无效或不需要数据
            bool flushing = std::isnan(request.position);

            // 准备 analyseGrain 的参数
            const float *grain_data_ptr = nullptr;
            int mute_head = 0;
            int mute_tail = 0;

            if (!flushing)
            {
                // 计算实际需要的输入范围 (相对于当前 input buffer)
                int needed_begin = input_chunk.begin;
                int needed_end = input_chunk.end;
                int needed_frames = needed_end - needed_begin;

                if (needed_frames > 0)
                {
                    // 计算可用数据的范围 (相对于当前 input buffer)
                    int available_begin = 0; // 输入 buffer 总是从 0 开始
                    int available_end = total_input_frames;

                    // 计算实际提供给 analyseGrain 的数据指针和静音帧数
                    int provide_begin = std::max(needed_begin, available_begin);
                    int provide_end = std::min(needed_end, available_end);
                    int provide_frames = provide_end - provide_begin;

                    if (provide_frames > 0)
                    {
                        grain_data_ptr = in_ptr + provide_begin * channels;
                        mute_head = provide_begin - needed_begin; // 开头有多少需要的帧无法提供
                        mute_tail = needed_end - provide_end;     // 末尾有多少需要的帧无法提供
                    }
                    else
                    {
                        // 需要的数据完全在可用范围之外
                        mute_head = needed_frames;
                        mute_tail = 0;
                    }

                    // 更新当前处理到的输入帧位置 (粗略估计)
                    // Bungee 内部会更精确地管理 position
                    current_input_frame = needed_end;
                    if (current_input_frame >= total_input_frames)
                    {
                        input_finished = true; // 标记输入已处理完
                    }
                }
                else
                {
                    // specifyGrain 返回了无效的块？或者 position 超前了？
                    // 标记输入结束，进入 flushing
                    input_finished = true;
                    continue; // 跳过 analyse/synthesise，直接进入下一轮 flushing 判断
                }
            }
            else
            {
                // Flushing 阶段，不需要实际数据，传递 nullptr
                grain_data_ptr = nullptr;
                // analyseGrain 可能需要知道整个预期块都是静音的
                // 但 Bungee 文档说 NAN grain 不产生输出，可能 analyseGrain 会忽略数据
                // 传递 0 试试
                mute_head = 0;
                mute_tail = 0;
            }

            // 2. 分析 Grain
            // 使用 channel stride = channels (对于 shape=(frames, channels) 的 C-style numpy array)
            stretcher.analyseGrain(grain_data_ptr, channels, mute_head, mute_tail);

            // 3. 合成 Grain
            Bungee::OutputChunk out_chunk;
            stretcher.synthesiseGrain(out_chunk);

            // 4. 拷贝输出 (处理非交错数据)
            if (out_chunk.frameCount > 0 && out_chunk.data)
            {
                output_buffer.reserve(output_buffer.size() + out_chunk.frameCount * channels); // 确保容量
                for (int frame = 0; frame < out_chunk.frameCount; ++frame)
                {
                    for (int ch = 0; ch < channels; ++ch)
                    {
                        // 从 Bungee 的 planar 输出读取数据
                        float sample = out_chunk.data[ch * out_chunk.channelStride + frame];
                        // 写入到 interleaved 的 vector
                        output_buffer.push_back(sample);
                    }
                }
            }

            // 5. 准备下一个 Request (仅在非 flushing 阶段)
            if (!flushing)
            {
                stretcher.next(request);
            }
            // Flushing 阶段 request.position 保持 NAN
        }

        // 返回结果
        ssize_t out_frames = output_buffer.size() / channels;
        // 创建 NumPy 数组，这里会拷贝 vector 的数据
        py::array_t<float> result({out_frames, (ssize_t)channels});
        std::memcpy(result.mutable_data(), output_buffer.data(), output_buffer.size() * sizeof(float));
        return result;
    }

private:
    Bungee::Stretcher<Bungee::Basic> stretcher;
    Bungee::Request request;
    int channels;
};

PYBIND11_MODULE(bungee, m)
{
    m.doc() = "bungee python bindings via pybind11";

    py::class_<PyBungee>(m, "Bungee")
        .def(py::init<int, int>(), py::arg("sample_rate"), py::arg("channels"))
        .def("set_speed", &PyBungee::set_speed, py::arg("speed"))
        .def("set_pitch", &PyBungee::set_pitch, py::arg("pitch"))
        .def("set_position", &PyBungee::set_position, py::arg("position"))
        .def("reset", &PyBungee::reset)
        .def("process", &PyBungee::process, py::arg("input"),
             "处理输入的音频数据 (NumPy 数组 shape=(frames, channels), dtype=float32).\n"
             "返回处理后的音频数据 (NumPy 数组 shape=(frames, channels), dtype=float32).");
}