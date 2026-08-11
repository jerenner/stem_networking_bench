/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "stem_aux_output.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef STEM_DAQIRI_HAVE_HDF5
#include <H5Cpp.h>
#endif

#ifdef STEM_DAQIRI_HAVE_ZMQ
#include <zmq.h>
#endif

namespace stem {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) { return; }
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(status));
}

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char ch) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  });
  return value;
}

void replace_all(std::string& value, const std::string& needle,
                 const std::string& replacement) {
  size_t position = 0;
  while ((position = value.find(needle, position)) != std::string::npos) {
    value.replace(position, needle.size(), replacement);
    position += replacement.size();
  }
}

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (ch < 0x20) {
          char buffer[7];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
          output << buffer;
        } else {
          output << static_cast<char>(ch);
        }
    }
  }
  return output.str();
}

#ifdef STEM_DAQIRI_HAVE_HDF5
std::string normalized_dataset_path(const std::string& path) {
  if (path.empty()) { return "/frames"; }
  return path.front() == '/' ? path : "/" + path;
}

void write_u64_attribute(H5::H5Object& object, const std::string& name,
                         uint64_t value) {
  H5::DataSpace scalar(H5S_SCALAR);
  auto attribute = object.createAttribute(name, H5::PredType::NATIVE_UINT64,
                                          scalar);
  attribute.write(H5::PredType::NATIVE_UINT64, &value);
}

void write_u32_attribute(H5::H5Object& object, const std::string& name,
                         uint32_t value) {
  H5::DataSpace scalar(H5S_SCALAR);
  auto attribute = object.createAttribute(name, H5::PredType::NATIVE_UINT32,
                                          scalar);
  attribute.write(H5::PredType::NATIVE_UINT32, &value);
}

void write_string_attribute(H5::H5Object& object, const std::string& name,
                            const std::string& value) {
  H5::DataSpace scalar(H5S_SCALAR);
  H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
  auto attribute = object.createAttribute(name, type, scalar);
  const char* pointer = value.c_str();
  attribute.write(type, &pointer);
}
#endif

}  // namespace

ProcessingStage parse_processing_stage(const std::string& value) {
  const std::string normalized = lowercase(value);
  if (normalized == "raw") { return ProcessingStage::kRaw; }
  if (normalized == "dark" || normalized == "dark_subtracted" ||
      normalized == "dark-subtracted") {
    return ProcessingStage::kDarkSubtracted;
  }
  if (normalized == "dark_blr" || normalized == "dark+blr" ||
      normalized == "blr_corrected" || normalized == "blr-corrected") {
    return ProcessingStage::kDarkBlr;
  }
  if (normalized == "corrected" || normalized == "fully_corrected") {
    return ProcessingStage::kCorrected;
  }
  if (normalized == "thresholded") {
    return ProcessingStage::kThresholded;
  }
  if (normalized == "counted") { return ProcessingStage::kCounted; }
  throw std::runtime_error(
      "processing stage must be raw, dark_subtracted, dark_blr, corrected, "
      "thresholded, or counted");
}

const char* processing_stage_name(ProcessingStage stage) {
  switch (stage) {
    case ProcessingStage::kRaw: return "raw";
    case ProcessingStage::kDarkSubtracted: return "dark_subtracted";
    case ProcessingStage::kDarkBlr: return "dark_blr";
    case ProcessingStage::kCorrected: return "corrected";
    case ProcessingStage::kThresholded: return "thresholded";
    case ProcessingStage::kCounted: return "counted";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Controlled burst capture.
// ---------------------------------------------------------------------------
struct BurstWriter::Impl {
  struct Buffer {
    void* device = nullptr;
    void* host = nullptr;
    cudaEvent_t ready = nullptr;
    size_t capacity = 0;
    size_t bytes = 0;
    bool reserved = false;
    BatchMetadata metadata;

    ~Buffer() {
      if (ready) { cudaEventDestroy(ready); }
      if (host) { cudaFreeHost(host); }
      if (device) { cudaFree(device); }
    }
  };

  struct ReceiverState {
    bool capturing = false;
    bool writing = false;
    bool done = false;
    uint64_t captures_started = 0;
    uint64_t next_expected_batch = 0;
    size_t next_buffer = 0;
    std::vector<Buffer*> active_buffers;
  };

  struct Job {
    uint32_t receiver_id = 0;
    uint64_t capture_index = 0;
    bool write = true;
    std::vector<Buffer*> buffers;
  };

  Impl(const BurstWriterConfig& value, uint32_t frame_height,
       uint32_t frame_width, uint32_t bucket_frames, uint32_t receivers,
       bool raw_float)
      : config(value),
        height(frame_height),
        width(frame_width),
        frames_per_bucket(bucket_frames),
        receiver_count(receivers),
        output_float(config.stage != ProcessingStage::kRaw || raw_float) {
    if (!config.enabled) { return; }
#ifndef STEM_DAQIRI_HAVE_HDF5
    throw std::runtime_error(
        "burst_writer.enabled=true requires HDF5 support in stem_daqiri_rx");
#else
    if (config.stage == ProcessingStage::kCounted) {
      throw std::runtime_error(
          "burst_writer.stage=counted is reserved for the future STEMPy-style "
          "event counter and is not implemented");
    }
    if (config.buckets_per_capture == 0) {
      throw std::runtime_error(
          "burst_writer.buckets_per_capture must be greater than zero");
    }
    if (receiver_count == 0) {
      throw std::runtime_error("BurstWriter requires at least one receiver");
    }
    if (receiver_count > 1 &&
        config.filepath_template.find("{receiver}") == std::string::npos) {
      throw std::runtime_error(
          "burst_writer.filepath_template must contain {receiver} when more "
          "than one receiver is configured");
    }
    const uint64_t elements = static_cast<uint64_t>(frames_per_bucket) *
                              height * width;
    const size_t bytes = static_cast<size_t>(elements) *
                         (output_float ? sizeof(float) : sizeof(uint16_t));
    buffers.resize(receiver_count);
    states.resize(receiver_count);
    for (auto& receiver_buffers : buffers) {
      receiver_buffers.reserve(config.buckets_per_capture);
      for (uint32_t index = 0; index < config.buckets_per_capture; ++index) {
        auto buffer = std::make_unique<Buffer>();
        buffer->capacity = bytes;
        check_cuda(cudaMalloc(&buffer->device, bytes),
                   "cudaMalloc burst device buffer");
        check_cuda(cudaMallocHost(&buffer->host, bytes),
                   "cudaMallocHost burst host buffer");
        check_cuda(cudaEventCreateWithFlags(&buffer->ready,
                                            cudaEventDisableTiming),
                   "cudaEventCreate burst buffer");
        receiver_buffers.push_back(std::move(buffer));
      }
    }
    worker = std::thread(&Impl::run, this);
#endif
  }

  ~Impl() {
    if (!config.enabled) { return; }
    drain();
    {
      std::lock_guard<std::mutex> lock(mu);
      stopping = true;
    }
    cv.notify_all();
    if (worker.joinable()) { worker.join(); }
  }

  bool capture_limit_reached(const ReceiverState& state) const {
    if (!config.rearm_after_write && state.captures_started >= 1) { return true; }
    return config.capture_count > 0 &&
           state.captures_started >= config.capture_count;
  }

  void queue_abort_locked(uint32_t receiver_id) {
    ReceiverState& state = states[receiver_id];
    Job job;
    job.receiver_id = receiver_id;
    job.write = false;
    for (size_t index = 0; index < state.next_buffer; ++index) {
      job.buffers.push_back(state.active_buffers[index]);
    }
    for (size_t index = state.next_buffer; index < state.active_buffers.size();
         ++index) {
      state.active_buffers[index]->reserved = false;
    }
    state.capturing = false;
    state.active_buffers.clear();
    state.next_buffer = 0;
    stats.aborted_captures++;
    if (job.buffers.empty()) {
      state.writing = false;
      return;
    }
    state.writing = true;
    jobs.push_back(std::move(job));
    cv.notify_one();
  }

  std::optional<BurstWriter::Reservation> reserve(
      const BatchMetadata& metadata) {
    if (!config.enabled) { return std::nullopt; }
    if (metadata.receiver_id >= receiver_count) {
      throw std::runtime_error("burst metadata receiver_id is out of range");
    }

    std::lock_guard<std::mutex> lock(mu);
    ReceiverState& state = states[metadata.receiver_id];
    if (state.done || capture_limit_reached(state)) {
      state.done = true;
      return std::nullopt;
    }
    if (state.writing) {
      stats.buckets_skipped_busy++;
      return std::nullopt;
    }
    if (state.capturing &&
        (metadata.batch_index != state.next_expected_batch ||
         (config.strict_complete && !metadata.complete))) {
      if (config.strict_complete && !metadata.complete) {
        stats.buckets_rejected_incomplete++;
      }
      queue_abort_locked(metadata.receiver_id);
      return std::nullopt;
    }
    if (!state.capturing) {
      if (config.strict_complete && !metadata.complete) {
        stats.buckets_rejected_incomplete++;
        return std::nullopt;
      }
      state.active_buffers.clear();
      for (auto& buffer : buffers[metadata.receiver_id]) {
        if (buffer->reserved) {
          throw std::runtime_error("burst buffer remained reserved after drain");
        }
        buffer->reserved = true;
        state.active_buffers.push_back(buffer.get());
      }
      state.capturing = true;
      state.next_buffer = 0;
      state.next_expected_batch = metadata.batch_index;
    }

    Buffer* buffer = state.active_buffers[state.next_buffer];
    buffer->metadata = metadata;
    const uint64_t elements = static_cast<uint64_t>(metadata.frames) * height *
                              width;
    buffer->bytes = static_cast<size_t>(elements) *
                    (output_float ? sizeof(float) : sizeof(uint16_t));
    if (buffer->bytes > buffer->capacity) {
      throw std::runtime_error("burst batch exceeds preallocated buffer size");
    }
    return BurstWriter::Reservation{buffer, buffer->device, buffer->bytes,
                                    output_float};
  }

  void finish_submission(Buffer* buffer) {
    std::lock_guard<std::mutex> lock(mu);
    ReceiverState& state = states[buffer->metadata.receiver_id];
    if (!state.capturing || state.active_buffers[state.next_buffer] != buffer) {
      throw std::runtime_error("burst reservation submitted out of order");
    }
    state.next_buffer++;
    state.next_expected_batch++;
    if (state.next_buffer != state.active_buffers.size()) { return; }

    Job job;
    job.receiver_id = buffer->metadata.receiver_id;
    job.capture_index = state.captures_started;
    job.buffers = state.active_buffers;
    state.captures_started++;
    state.capturing = false;
    state.writing = true;
    state.active_buffers.clear();
    state.next_buffer = 0;
    stats.captures_started++;
    stats.buckets_captured += job.buffers.size();
    jobs.push_back(std::move(job));
    cv.notify_one();
  }

  std::string output_path(const Job& job) const {
    std::string path = config.filepath_template;
    replace_all(path, "{receiver}", std::to_string(job.receiver_id));
    replace_all(path, "{capture}", std::to_string(job.capture_index));
    replace_all(path, "{stage}", processing_stage_name(config.stage));
    if (!job.buffers.empty()) {
      replace_all(path, "{first_frame}",
                  std::to_string(job.buffers.front()->metadata.first_frame));
    }
    return path;
  }

#ifdef STEM_DAQIRI_HAVE_HDF5
  void write_job(const Job& job) {
    const std::string path = output_path(job);
    const std::filesystem::path filesystem_path(path);
    if (filesystem_path.has_parent_path()) {
      std::filesystem::create_directories(filesystem_path.parent_path());
    }

    H5::H5File file(path, H5F_ACC_TRUNC);
    uint64_t total_frames = 0;
    for (const Buffer* buffer : job.buffers) {
      total_frames += buffer->metadata.frames;
    }
    hsize_t dimensions[3] = {total_frames, height, width};
    H5::DataSpace filespace(3, dimensions);
    H5::DSetCreatPropList properties;
    hsize_t chunk[3] = {1, height, width};
    properties.setChunk(3, chunk);
    const H5::DataType type = output_float ? H5::PredType::NATIVE_FLOAT
                                           : H5::PredType::NATIVE_UINT16;
    H5::DataSet dataset = file.createDataSet(
        normalized_dataset_path(config.dataset_name), type, filespace,
        properties);

    hsize_t frame_offset = 0;
    for (const Buffer* buffer : job.buffers) {
      hsize_t count[3] = {buffer->metadata.frames, height, width};
      hsize_t offset[3] = {frame_offset, 0, 0};
      H5::DataSpace selected = dataset.getSpace();
      selected.selectHyperslab(H5S_SELECT_SET, count, offset);
      H5::DataSpace memory(3, count);
      dataset.write(buffer->host, type, memory, selected);
      frame_offset += buffer->metadata.frames;
    }

    const BatchMetadata& first = job.buffers.front()->metadata;
    const BatchMetadata& last = job.buffers.back()->metadata;
    write_u32_attribute(dataset, "schema_version", 1);
    write_u32_attribute(dataset, "receiver_id", job.receiver_id);
    write_u64_attribute(dataset, "capture_index", job.capture_index);
    write_u64_attribute(dataset, "first_frame", first.first_frame);
    write_u64_attribute(dataset, "last_frame_exclusive",
                        last.first_frame + last.frames);
    write_u32_attribute(dataset, "buckets", job.buffers.size());
    write_u32_attribute(dataset, "frames_per_bucket", frames_per_bucket);
    write_string_attribute(dataset, "processing_stage",
                           processing_stage_name(config.stage));
    write_string_attribute(dataset, "interface_name", first.interface_name);
    write_string_attribute(dataset, "dtype",
                           output_float ? "float32" : "uint16");
  }
#endif

  void run() {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return stopping || !jobs.empty(); });
        if (stopping && jobs.empty()) { return; }
        job = std::move(jobs.front());
        jobs.pop_front();
        active_jobs++;
      }

      bool success = true;
      try {
        if (job.write) {
          for (Buffer* buffer : job.buffers) {
            check_cuda(cudaEventSynchronize(buffer->ready),
                       "cudaEventSynchronize burst buffer");
            check_cuda(cudaMemcpy(buffer->host, buffer->device, buffer->bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy burst device-to-host");
          }
#ifdef STEM_DAQIRI_HAVE_HDF5
          write_job(job);
#endif
        } else {
          for (Buffer* buffer : job.buffers) {
            check_cuda(cudaEventSynchronize(buffer->ready),
                       "cudaEventSynchronize aborted burst buffer");
          }
        }
      } catch (const std::exception& error) {
        std::fprintf(stderr, "BurstWriter failed: %s\n", error.what());
        success = false;
#ifdef STEM_DAQIRI_HAVE_HDF5
      } catch (const H5::Exception& error) {
        std::fprintf(stderr, "BurstWriter HDF5 failure: %s\n",
                     error.getCDetailMsg());
        success = false;
#endif
      }

      {
        std::lock_guard<std::mutex> lock(mu);
        for (Buffer* buffer : job.buffers) { buffer->reserved = false; }
        ReceiverState& state = states[job.receiver_id];
        state.writing = false;
        if (job.write && success) {
          stats.captures_written++;
        } else if (job.write) {
          stats.errors++;
        }
        if (capture_limit_reached(state)) { state.done = true; }
        active_jobs--;
      }
      cv.notify_all();
    }
  }

  void drain() {
    if (!config.enabled) { return; }
    std::unique_lock<std::mutex> lock(mu);
    for (uint32_t receiver = 0; receiver < states.size(); ++receiver) {
      if (states[receiver].capturing) { queue_abort_locked(receiver); }
    }
    cv.wait(lock, [&] {
      if (!jobs.empty() || active_jobs != 0) { return false; }
      return std::none_of(states.begin(), states.end(),
                          [](const ReceiverState& state) {
                            return state.capturing || state.writing;
                          });
    });
  }

  BurstWriterStats snapshot() const {
    std::lock_guard<std::mutex> lock(mu);
    return stats;
  }

  BurstWriterConfig config;
  uint32_t height = 0;
  uint32_t width = 0;
  uint32_t frames_per_bucket = 0;
  uint32_t receiver_count = 0;
  bool output_float = false;
  std::vector<std::vector<std::unique_ptr<Buffer>>> buffers;
  std::vector<ReceiverState> states;

  mutable std::mutex mu;
  std::condition_variable cv;
  std::deque<Job> jobs;
  size_t active_jobs = 0;
  bool stopping = false;
  std::thread worker;
  BurstWriterStats stats;
};

BurstWriter::BurstWriter(const BurstWriterConfig& config, uint32_t height,
                         uint32_t width, uint32_t frames_per_bucket,
                         uint32_t receiver_count, bool raw_input_float32)
    : impl_(std::make_unique<Impl>(config, height, width, frames_per_bucket,
                                   receiver_count, raw_input_float32)) {}

BurstWriter::~BurstWriter() = default;

bool BurstWriter::enabled() const { return impl_->config.enabled; }

ProcessingStage BurstWriter::stage() const { return impl_->config.stage; }

std::optional<BurstWriter::Reservation> BurstWriter::reserve(
    const BatchMetadata& metadata) {
  return impl_->reserve(metadata);
}

void BurstWriter::submit_copy(const Reservation& reservation,
                              const void* source, bool source_float32,
                              cudaStream_t stream) {
  if (!reservation) { return; }
  if (source_float32 != reservation.float32) {
    throw std::runtime_error("burst stage source type does not match output type");
  }
  auto* buffer = static_cast<Impl::Buffer*>(reservation.token);
  check_cuda(cudaMemcpyAsync(buffer->device, source, reservation.bytes,
                             cudaMemcpyDeviceToDevice, stream),
             "cudaMemcpyAsync burst device-to-device");
  check_cuda(cudaEventRecord(buffer->ready, stream),
             "cudaEventRecord burst buffer");
  impl_->finish_submission(buffer);
}

void BurstWriter::submit_direct(const Reservation& reservation,
                                cudaStream_t stream) {
  if (!reservation) { return; }
  auto* buffer = static_cast<Impl::Buffer*>(reservation.token);
  check_cuda(cudaEventRecord(buffer->ready, stream),
             "cudaEventRecord direct burst buffer");
  impl_->finish_submission(buffer);
}

void BurstWriter::drain() { impl_->drain(); }

BurstWriterStats BurstWriter::stats() const { return impl_->snapshot(); }

// ---------------------------------------------------------------------------
// Latest-only thinned PUB stream.
// ---------------------------------------------------------------------------
struct ThinnedStreamPublisher::Impl {
  enum class SlotState { kFree, kFilling, kQueued, kSending };

  struct Slot {
    float* representative_device = nullptr;
    float* sum_device = nullptr;
    float* representative_host = nullptr;
    float* sum_host = nullptr;
    cudaEvent_t ready = nullptr;
    SlotState state = SlotState::kFree;
    BatchMetadata metadata;

    ~Slot() {
      if (ready) { cudaEventDestroy(ready); }
      if (representative_host) { cudaFreeHost(representative_host); }
      if (sum_host) { cudaFreeHost(sum_host); }
      if (representative_device) { cudaFree(representative_device); }
      if (sum_device) { cudaFree(sum_device); }
    }
  };

  Impl(const ThinnedStreamConfig& value, uint32_t frame_height,
       uint32_t frame_width, uint32_t bucket_frames, uint32_t receivers)
      : config(value),
        height(frame_height),
        width(frame_width),
        frames_per_bucket(bucket_frames),
        receiver_count(receivers) {
    if (!config.enabled) { return; }
#ifndef STEM_DAQIRI_HAVE_ZMQ
    throw std::runtime_error(
        "thinned_stream.enabled=true requires ZeroMQ support in "
        "stem_daqiri_rx");
#else
    if (config.stage == ProcessingStage::kCounted) {
      throw std::runtime_error(
          "thinned_stream.stage=counted is reserved for the future "
          "STEMPy-style event counter and is not implemented");
    }
    if (!config.include_representative_frame && !config.include_bucket_sum) {
      throw std::runtime_error(
          "thinned_stream must include a representative frame, bucket sum, "
          "or both");
    }
    if (config.queue_depth == 0) {
      throw std::runtime_error("thinned_stream.queue_depth must be > 0");
    }
    if (config.representative_frame_index >= frames_per_bucket) {
      throw std::runtime_error(
          "thinned_stream.representative_frame_index is outside the bucket");
    }
    const size_t frame_bytes = static_cast<size_t>(height) * width *
                               sizeof(float);
    const uint32_t slot_count = std::max<uint32_t>(2, config.queue_depth + 1);
    slots.reserve(slot_count);
    for (uint32_t index = 0; index < slot_count; ++index) {
      auto slot = std::make_unique<Slot>();
      if (config.include_representative_frame) {
        check_cuda(cudaMalloc(&slot->representative_device, frame_bytes),
                   "cudaMalloc thinned representative device buffer");
        check_cuda(cudaMallocHost(&slot->representative_host, frame_bytes),
                   "cudaMallocHost thinned representative host buffer");
      }
      if (config.include_bucket_sum) {
        check_cuda(cudaMalloc(&slot->sum_device, frame_bytes),
                   "cudaMalloc thinned sum device buffer");
        check_cuda(cudaMallocHost(&slot->sum_host, frame_bytes),
                   "cudaMallocHost thinned sum host buffer");
      }
      check_cuda(cudaEventCreateWithFlags(&slot->ready,
                                          cudaEventDisableTiming),
                 "cudaEventCreate thinned slot");
      slots.push_back(std::move(slot));
    }
    next_due = std::chrono::steady_clock::now();
    worker = std::thread(&Impl::run, this);
    std::unique_lock<std::mutex> lock(mu);
    initialized_cv.wait(lock, [&] { return initialized; });
    if (!initialization_error.empty()) {
      lock.unlock();
      if (worker.joinable()) { worker.join(); }
      throw std::runtime_error(initialization_error);
    }
#endif
  }

  ~Impl() {
    if (!config.enabled) { return; }
    drain();
    {
      std::lock_guard<std::mutex> lock(mu);
      stopping = true;
    }
    cv.notify_all();
    if (worker.joinable()) { worker.join(); }
  }

  void advance_schedule(std::chrono::steady_clock::time_point now) {
    next_receiver = (next_receiver + 1) % receiver_count;
    if (config.total_refresh_hz > 0.0) {
      const auto period = std::chrono::duration<double>(
          1.0 / config.total_refresh_hz);
      next_due = now +
                 std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                     period);
    }
  }

  std::optional<ThinnedStreamPublisher::Reservation> reserve(
      const BatchMetadata& metadata) {
    if (!config.enabled) { return std::nullopt; }
    if (metadata.receiver_id >= receiver_count) {
      throw std::runtime_error("thinned metadata receiver_id is out of range");
    }
    std::lock_guard<std::mutex> lock(mu);
    const auto now = std::chrono::steady_clock::now();
    if (metadata.receiver_id != next_receiver ||
        (config.total_refresh_hz > 0.0 && now < next_due)) {
      return std::nullopt;
    }
    advance_schedule(now);

    auto available = std::find_if(slots.begin(), slots.end(), [](const auto& slot) {
      return slot->state == SlotState::kFree;
    });
    if (available == slots.end()) {
      stats.products_dropped_no_buffer++;
      return std::nullopt;
    }
    Slot* slot = available->get();
    slot->state = SlotState::kFilling;
    slot->metadata = metadata;
    return ThinnedStreamPublisher::Reservation{
        slot, slot->representative_device, slot->sum_device};
  }

  void submit(Slot* slot, cudaStream_t stream) {
    const size_t frame_bytes = static_cast<size_t>(height) * width *
                               sizeof(float);
    if (config.include_representative_frame) {
      check_cuda(cudaMemcpyAsync(slot->representative_host,
                                 slot->representative_device, frame_bytes,
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync thinned representative");
    }
    if (config.include_bucket_sum) {
      check_cuda(cudaMemcpyAsync(slot->sum_host, slot->sum_device, frame_bytes,
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync thinned sum");
    }
    check_cuda(cudaEventRecord(slot->ready, stream),
               "cudaEventRecord thinned slot");
    {
      std::lock_guard<std::mutex> lock(mu);
      if (slot->state != SlotState::kFilling) {
        throw std::runtime_error("thinned reservation submitted twice");
      }
      slot->state = SlotState::kQueued;
      queue.push_back(slot);
      stats.products_queued++;
    }
    cv.notify_one();
  }

  std::string topic(const Slot& slot) const {
    std::string prefix = config.topic_prefix;
    while (!prefix.empty() && prefix.back() == '/') { prefix.pop_back(); }
    return prefix + "/rx/" + std::to_string(slot.metadata.receiver_id) + "/" +
           processing_stage_name(config.stage);
  }

  std::string metadata_json(const Slot& slot) const {
    std::ostringstream json;
    json << "{\"schema\":\"stem.thinned.v1\",\"receiver_id\":"
         << slot.metadata.receiver_id << ",\"interface_name\":\""
         << json_escape(slot.metadata.interface_name) << "\",\"batch_index\":"
         << slot.metadata.batch_index << ",\"first_frame\":"
         << slot.metadata.first_frame << ",\"frames_in_bucket\":"
         << slot.metadata.frames << ",\"complete\":"
         << (slot.metadata.complete ? "true" : "false")
         << ",\"received_packets\":" << slot.metadata.received_packets
         << ",\"expected_packets\":" << slot.metadata.expected_packets
         << ",\"processing_stage\":\"" << processing_stage_name(config.stage)
         << "\",\"dtype\":\"float32\",\"byte_order\":\"little\""
         << ",\"height\":" << height << ",\"width\":" << width
         << ",\"representative_frame_index\":"
         << config.representative_frame_index << ",\"parts\":[\"metadata\"";
    if (config.include_representative_frame) { json << ",\"representative\""; }
    if (config.include_bucket_sum) { json << ",\"sum\""; }
    json << "]}";
    return json.str();
  }

#ifdef STEM_DAQIRI_HAVE_ZMQ
  bool send_part(void* socket, const void* data, size_t size, bool more) {
    const int flags = ZMQ_DONTWAIT | (more ? ZMQ_SNDMORE : 0);
    return zmq_send(socket, data, size, flags) >= 0;
  }

  bool publish(void* socket, const Slot& slot) {
    const std::string topic_value = topic(slot);
    const std::string json = metadata_json(slot);
    const size_t frame_bytes = static_cast<size_t>(height) * width *
                               sizeof(float);
    const uint32_t payload_parts =
        static_cast<uint32_t>(config.include_representative_frame) +
        static_cast<uint32_t>(config.include_bucket_sum);
    if (!send_part(socket, topic_value.data(), topic_value.size(), true) ||
        !send_part(socket, json.data(), json.size(), payload_parts > 0)) {
      return false;
    }
    uint32_t remaining = payload_parts;
    if (config.include_representative_frame) {
      remaining--;
      if (!send_part(socket, slot.representative_host, frame_bytes,
                     remaining > 0)) {
        return false;
      }
    }
    if (config.include_bucket_sum) {
      remaining--;
      if (!send_part(socket, slot.sum_host, frame_bytes, remaining > 0)) {
        return false;
      }
    }
    return true;
  }
#endif

  void mark_free(Slot* slot) {
    std::lock_guard<std::mutex> lock(mu);
    slot->state = SlotState::kFree;
  }

  void set_initialized(const std::string& error = {}) {
    {
      std::lock_guard<std::mutex> lock(mu);
      initialization_error = error;
      initialized = true;
    }
    initialized_cv.notify_all();
  }

  void run() {
#ifndef STEM_DAQIRI_HAVE_ZMQ
    set_initialized("ZeroMQ support is not compiled");
#else
    void* context = zmq_ctx_new();
    if (!context) {
      set_initialized("zmq_ctx_new failed");
      return;
    }
    void* socket = zmq_socket(context, ZMQ_PUB);
    if (!socket) {
      const std::string error = std::string("zmq_socket failed: ") +
                                zmq_strerror(zmq_errno());
      zmq_ctx_term(context);
      set_initialized(error);
      return;
    }
    const int linger = 0;
    const int high_water_mark = static_cast<int>(config.queue_depth);
    zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(socket, ZMQ_SNDHWM, &high_water_mark,
                   sizeof(high_water_mark));
    if (zmq_bind(socket, config.endpoint.c_str()) != 0) {
      const std::string error = std::string("ZeroMQ bind failed for ") +
                                config.endpoint + ": " +
                                zmq_strerror(zmq_errno());
      zmq_close(socket);
      zmq_ctx_term(context);
      set_initialized(error);
      return;
    }
    set_initialized();

    for (;;) {
      Slot* newest = nullptr;
      std::vector<Slot*> stale;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return stopping || !queue.empty(); });
        if (stopping && queue.empty()) { break; }
        while (queue.size() > 1) {
          stale.push_back(queue.front());
          queue.pop_front();
          stats.products_coalesced++;
        }
        newest = queue.front();
        queue.pop_front();
        newest->state = SlotState::kSending;
        sending = true;
      }

      for (Slot* slot : stale) {
        if (cudaEventSynchronize(slot->ready) != cudaSuccess) {
          std::lock_guard<std::mutex> lock(mu);
          stats.send_errors++;
        }
        mark_free(slot);
      }

      bool success = cudaEventSynchronize(newest->ready) == cudaSuccess;
      if (success) { success = publish(socket, *newest); }
      {
        std::lock_guard<std::mutex> lock(mu);
        if (success) {
          stats.products_published++;
        } else {
          stats.send_errors++;
        }
        newest->state = SlotState::kFree;
        sending = false;
      }
      cv.notify_all();
    }

    zmq_close(socket);
    zmq_ctx_term(context);
#endif
  }

  void drain() {
    if (!config.enabled) { return; }
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] {
      if (!queue.empty() || sending) { return false; }
      return std::none_of(slots.begin(), slots.end(), [](const auto& slot) {
        return slot->state == SlotState::kFilling ||
               slot->state == SlotState::kQueued ||
               slot->state == SlotState::kSending;
      });
    });
  }

  ThinnedStreamStats snapshot() const {
    std::lock_guard<std::mutex> lock(mu);
    return stats;
  }

  ThinnedStreamConfig config;
  uint32_t height = 0;
  uint32_t width = 0;
  uint32_t frames_per_bucket = 0;
  uint32_t receiver_count = 0;
  std::vector<std::unique_ptr<Slot>> slots;

  mutable std::mutex mu;
  std::condition_variable cv;
  std::condition_variable initialized_cv;
  std::deque<Slot*> queue;
  std::thread worker;
  bool stopping = false;
  bool sending = false;
  bool initialized = false;
  std::string initialization_error;
  uint32_t next_receiver = 0;
  std::chrono::steady_clock::time_point next_due;
  ThinnedStreamStats stats;
};

ThinnedStreamPublisher::ThinnedStreamPublisher(
    const ThinnedStreamConfig& config, uint32_t height, uint32_t width,
    uint32_t frames_per_bucket, uint32_t receiver_count)
    : impl_(std::make_unique<Impl>(config, height, width, frames_per_bucket,
                                   receiver_count)) {}

ThinnedStreamPublisher::~ThinnedStreamPublisher() = default;

bool ThinnedStreamPublisher::enabled() const { return impl_->config.enabled; }

ProcessingStage ThinnedStreamPublisher::stage() const {
  return impl_->config.stage;
}

std::optional<ThinnedStreamPublisher::Reservation>
ThinnedStreamPublisher::reserve(const BatchMetadata& metadata) {
  return impl_->reserve(metadata);
}

void ThinnedStreamPublisher::submit(const Reservation& reservation,
                                    cudaStream_t stream) {
  if (!reservation) { return; }
  impl_->submit(static_cast<Impl::Slot*>(reservation.token), stream);
}

void ThinnedStreamPublisher::drain() { impl_->drain(); }

ThinnedStreamStats ThinnedStreamPublisher::stats() const {
  return impl_->snapshot();
}

}  // namespace stem
