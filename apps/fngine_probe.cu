// fngine-probe: measures the machine facts the engine's cost models depend on and
// prints one JSON object: GPU, PCIe copy bandwidth with the link state sampled
// during the copy, CPU streaming-read bandwidth, GPU<->CPU doorbell latency and
// io_uring O_DIRECT random-read IOPS.
#include "cpu/cpu_isa.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <immintrin.h>
#include <liburing.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,                   \
                   cudaGetErrorString(e_));                                            \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string read_sysfs(const std::string& path) {
  std::ifstream f(path);
  std::string s;
  std::getline(f, s);
  return s;
}

std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

// CPUs in the order the engine assigns CPU expert workers: CCD0 cores 1-5, CCD1
// cores 7-11, then the engine (0) and I/O (6) cores. SMT siblings are unused.
constexpr int kCpuOrder[12] = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 0, 6};

void pin_to(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// ---------------------------------------------------------------- PCIe
struct LinkSample {
  std::string width, speed, port_width, port_speed;
};

struct PcieResult {
  double h2d_gbs = 0, d2h_gbs = 0;
  LinkSample during;
  std::string max_width, max_speed;
  std::string device_path;
};

PcieResult probe_pcie() {
  PcieResult r;
  char bus[32] = {};
  CK(cudaDeviceGetPCIBusId(bus, sizeof(bus), 0));
  std::string id(bus);
  std::transform(id.begin(), id.end(), id.begin(), ::tolower);
  std::string dev = "/sys/bus/pci/devices/" + id;
  char real[PATH_MAX] = {};
  std::string port;
  if (realpath(dev.c_str(), real)) {
    std::string rp(real);
    port = rp.substr(0, rp.rfind('/'));
    r.device_path = rp;
  }
  r.max_width = read_sysfs(dev + "/max_link_width");
  r.max_speed = read_sysfs(dev + "/max_link_speed");

  constexpr size_t kBytes = 256ull << 20;
  constexpr int kReps = 8;
  void* host = nullptr;
  void* devp = nullptr;
  CK(cudaHostAlloc(&host, kBytes, cudaHostAllocDefault));
  CK(cudaMalloc(&devp, kBytes));
  std::memset(host, 1, kBytes);
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));

  // Warm-up wakes the link from its idle power state.
  CK(cudaMemcpyAsync(devp, host, kBytes, cudaMemcpyHostToDevice, s));
  CK(cudaStreamSynchronize(s));

  std::atomic<bool> sampling{true};
  std::thread sampler([&] {
    // Sample the link while copies are in flight; keep the widest observation.
    int best = -1;
    while (sampling.load()) {
      LinkSample ls{read_sysfs(dev + "/current_link_width"),
                    read_sysfs(dev + "/current_link_speed"),
                    port.empty() ? "" : read_sysfs(port + "/current_link_width"),
                    port.empty() ? "" : read_sysfs(port + "/current_link_speed")};
      int w = std::atoi(ls.width.c_str());
      if (w > best) {
        best = w;
        r.during = ls;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });

  auto timed = [&](cudaMemcpyKind kind) {
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < kReps; ++i) {
      if (kind == cudaMemcpyHostToDevice)
        CK(cudaMemcpyAsync(devp, host, kBytes, kind, s));
      else
        CK(cudaMemcpyAsync(host, devp, kBytes, kind, s));
    }
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    return double(kBytes) * kReps / (ms * 1e-3) / 1e9;
  };
  r.h2d_gbs = timed(cudaMemcpyHostToDevice);
  r.d2h_gbs = timed(cudaMemcpyDeviceToHost);
  sampling = false;
  sampler.join();

  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  CK(cudaStreamDestroy(s));
  CK(cudaFree(devp));
  CK(cudaFreeHost(host));
  return r;
}

// ---------------------------------------------------------------- CPU bandwidth
struct CpuBw {
  int threads;
  double gbs;
};

std::vector<CpuBw> probe_cpu_bandwidth() {
  constexpr size_t kBytes = 4ull << 30;
  constexpr size_t kHuge = 2ull << 20;
  void* mem = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) {
    std::perror("mmap");
    std::exit(1);
  }
  madvise(mem, kBytes, MADV_HUGEPAGE);
  // First touch from the worker set so pages are populated before timing.
  {
    std::vector<std::thread> ts;
    for (int t = 0; t < 10; ++t)
      ts.emplace_back([=] {
        pin_to(kCpuOrder[t]);
        size_t per = kBytes / 10 / kHuge * kHuge;
        size_t beg = per * t, end = t == 9 ? kBytes : beg + per;
        std::memset(static_cast<char*>(mem) + beg, t + 1, end - beg);
      });
    for (auto& th : ts) th.join();
  }

  std::vector<CpuBw> out;
  for (int n : {1, 4, 8, 10, 12}) {
    double best = 0;
    for (int rep = 0; rep < 3; ++rep) {
      std::atomic<int> ready{0};
      std::atomic<bool> go{false};
      std::vector<std::thread> ts;
      std::vector<uint64_t> sink(n * 8);
      for (int t = 0; t < n; ++t)
        ts.emplace_back([&, t] {
          pin_to(kCpuOrder[t]);
          size_t per = kBytes / n / 64 * 64;
          ready.fetch_add(1);
          while (!go.load(std::memory_order_acquire)) _mm_pause();
#if FNGINE_CPU_BACKEND_AVX512
          const __m512i* p = reinterpret_cast<const __m512i*>(static_cast<char*>(mem) + per * t);
          size_t cnt = per / 64;
          __m512i acc0 = _mm512_setzero_si512(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
          size_t i = 0;
          for (; i + 4 <= cnt; i += 4) {
            acc0 = _mm512_add_epi64(acc0, _mm512_load_si512(p + i));
            acc1 = _mm512_add_epi64(acc1, _mm512_load_si512(p + i + 1));
            acc2 = _mm512_add_epi64(acc2, _mm512_load_si512(p + i + 2));
            acc3 = _mm512_add_epi64(acc3, _mm512_load_si512(p + i + 3));
          }
          acc0 = _mm512_add_epi64(_mm512_add_epi64(acc0, acc1), _mm512_add_epi64(acc2, acc3));
          sink[t * 8] = _mm512_reduce_add_epi64(acc0);
#else
          const __m256i* p = reinterpret_cast<const __m256i*>(static_cast<char*>(mem) + per * t);
          size_t cnt = per / sizeof(__m256i);
          __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
          size_t i = 0;
          for (; i + 4 <= cnt; i += 4) {
            acc0 = _mm256_add_epi64(acc0, _mm256_load_si256(p + i));
            acc1 = _mm256_add_epi64(acc1, _mm256_load_si256(p + i + 1));
            acc2 = _mm256_add_epi64(acc2, _mm256_load_si256(p + i + 2));
            acc3 = _mm256_add_epi64(acc3, _mm256_load_si256(p + i + 3));
          }
          acc0 = _mm256_add_epi64(_mm256_add_epi64(acc0, acc1), _mm256_add_epi64(acc2, acc3));
          __m128i sum128 = _mm_add_epi64(_mm256_castsi256_si128(acc0), _mm256_extracti128_si256(acc0, 1));
          sum128 = _mm_add_epi64(sum128, _mm_srli_si128(sum128, 8));
          sink[t * 8] = static_cast<uint64_t>(_mm_cvtsi128_si64(sum128));
#endif
        });
      while (ready.load() != n) _mm_pause();
      auto t0 = Clock::now();
      go.store(true, std::memory_order_release);
      for (auto& th : ts) th.join();
      double secs = seconds_since(t0);
      best = std::max(best, double(kBytes / n / 64 * 64) * n / secs / 1e9);
    }
    out.push_back({n, best});
  }
  munmap(mem, kBytes);
  return out;
}

// ---------------------------------------------------------------- doorbell
__device__ __forceinline__ void st_release_sys(volatile uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const volatile uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

struct alignas(64) Bell {
  uint32_t ready;
  uint32_t pad0[15];
  uint32_t done;
  uint32_t pad1[15];
};

__global__ void doorbell_kernel(Bell* bell, uint64_t* rtt_ns, int iters) {
  if (threadIdx.x != 0) return;
  for (int i = 1; i <= iters; ++i) {
    uint64_t t0 = globaltimer();
    __threadfence_system();
    st_release_sys(&bell->ready, uint32_t(i));
    while (ld_acquire_sys(&bell->done) != uint32_t(i)) __nanosleep(32);
    rtt_ns[i - 1] = globaltimer() - t0;
  }
}

double probe_doorbell_us() {
  constexpr int kIters = 10000;
  Bell* bell = nullptr;
  CK(cudaHostAlloc(reinterpret_cast<void**>(&bell), sizeof(Bell), cudaHostAllocMapped));
  std::memset(bell, 0, sizeof(Bell));
  Bell* dbell = nullptr;
  CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dbell), bell, 0));
  uint64_t* rtt = nullptr;
  CK(cudaMalloc(&rtt, kIters * sizeof(uint64_t)));

  std::atomic<bool> stop{false};
  std::thread cpu([&] {
    pin_to(kCpuOrder[0]);
    std::atomic_ref<uint32_t> ready(bell->ready), done(bell->done);
    uint32_t seen = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      uint32_t r = ready.load(std::memory_order_acquire);
      if (r != seen) {
        seen = r;
        done.store(r, std::memory_order_release);
      } else {
        _mm_pause();
      }
    }
  });
  doorbell_kernel<<<1, 32>>>(dbell, rtt, kIters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  stop = true;
  cpu.join();

  std::vector<uint64_t> h(kIters);
  CK(cudaMemcpy(h.data(), rtt, kIters * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  std::sort(h.begin(), h.end());
  CK(cudaFree(rtt));
  CK(cudaFreeHost(bell));
  return h[kIters / 2] / 1000.0;
}

// ---------------------------------------------------------------- io_uring
struct IoResult {
  bool ok = false;
  std::string error;
  double iops = 0, gbs = 0;
  uint64_t file_bytes = 0;
};

IoResult probe_io(const std::string& path) {
  IoResult r;
  int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    r.error = std::string("open: ") + std::strerror(errno);
    return r;
  }
  struct stat st {};
  fstat(fd, &st);
  r.file_bytes = st.st_size;
  const uint64_t pages = uint64_t(st.st_size) / 4096;
  if (pages < 1024) {
    r.error = "file too small";
    close(fd);
    return r;
  }
  constexpr int kQd = 64;
  io_uring ring;
  if (int e = io_uring_queue_init(kQd, &ring, 0); e < 0) {
    r.error = std::string("io_uring_queue_init: ") + std::strerror(-e);
    close(fd);
    return r;
  }
  std::vector<void*> bufs(kQd);
  for (auto& b : bufs)
    if (posix_memalign(&b, 4096, 4096)) std::abort();
  std::mt19937_64 rng(1234);
  auto submit = [&](int slot) {
    io_uring_sqe* sqe = io_uring_get_sqe(&ring);
    uint64_t off = (rng() % pages) * 4096;
    io_uring_prep_read(sqe, fd, bufs[slot], 4096, off);
    io_uring_sqe_set_data64(sqe, uint64_t(slot));
  };
  for (int i = 0; i < kQd; ++i) submit(i);
  io_uring_submit(&ring);
  uint64_t done = 0;
  auto t0 = Clock::now();
  double secs = 0;
  while ((secs = seconds_since(t0)) < 5.0) {
    io_uring_cqe* cqe = nullptr;
    if (int e = io_uring_wait_cqe(&ring, &cqe); e < 0) {
      r.error = std::string("wait_cqe: ") + std::strerror(-e);
      break;
    }
    unsigned head, n = 0;
    io_uring_for_each_cqe(&ring, head, cqe) {
      if (cqe->res != 4096) {
        r.error = "short or failed read: " + std::to_string(cqe->res);
      }
      submit(int(io_uring_cqe_get_data64(cqe)));
      ++n;
    }
    (void)head;
    io_uring_cq_advance(&ring, n);
    done += n;
    io_uring_submit(&ring);
  }
  io_uring_queue_exit(&ring);
  for (auto b : bufs) std::free(b);
  close(fd);
  r.ok = r.error.empty();
  r.iops = done / secs;
  r.gbs = r.iops * 4096 / 1e9;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string file;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--file" && i + 1 < argc) {
      file = argv[++i];
    } else if (a == "--help" || a == "-h") {
      std::printf("usage: fngine-probe [--file PATH]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }

  if (!fngine::cpu::cpu_backend_usable()) {
    std::fprintf(stderr, "CPU does not support the compiled expert backend (%s)\n",
                 fngine::cpu::cpu_backend_name());
    return 1;
  }

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));

  PcieResult pcie = probe_pcie();
  std::vector<CpuBw> cpu = probe_cpu_bandwidth();
  double bell_us = probe_doorbell_us();
  IoResult io;
  if (!file.empty()) io = probe_io(file);

  std::printf("{\n");
  std::printf("  \"cpu_backend\": %s,\n", json_str(fngine::cpu::cpu_backend_name()).c_str());
  std::printf("  \"gpu\": {\"name\": %s, \"sms\": %d, \"cc\": \"%d.%d\", \"free_mib\": %zu, \"total_mib\": %zu},\n",
              json_str(prop.name).c_str(), prop.multiProcessorCount, prop.major, prop.minor,
              free_b >> 20, total_b >> 20);
  std::printf("  \"pcie\": {\"h2d_gbs\": %.2f, \"d2h_gbs\": %.2f, \"device\": %s, "
              "\"link_width_during_copy\": %s, \"link_speed_during_copy\": %s, "
              "\"port_link_width_during_copy\": %s, \"port_link_speed_during_copy\": %s, "
              "\"max_link_width\": %s, \"max_link_speed\": %s},\n",
              pcie.h2d_gbs, pcie.d2h_gbs, json_str(pcie.device_path).c_str(),
              json_str(pcie.during.width).c_str(), json_str(pcie.during.speed).c_str(),
              json_str(pcie.during.port_width).c_str(), json_str(pcie.during.port_speed).c_str(),
              json_str(pcie.max_width).c_str(), json_str(pcie.max_speed).c_str());
  std::printf("  \"cpu_stream_read_gbs\": {");
  for (size_t i = 0; i < cpu.size(); ++i)
    std::printf("%s\"%d\": %.2f", i ? ", " : "", cpu[i].threads, cpu[i].gbs);
  std::printf("},\n");
  std::printf("  \"doorbell_rtt_us_median\": %.2f", bell_us);
  if (!file.empty()) {
    std::printf(",\n  \"io_uring_odirect_4k_qd64\": {\"file\": %s, \"ok\": %s, \"error\": %s, "
                "\"iops\": %.0f, \"gbs\": %.3f}",
                json_str(file).c_str(), io.ok ? "true" : "false", json_str(io.error).c_str(),
                io.iops, io.gbs);
  }
  std::printf("\n}\n");
  if (pcie.h2d_gbs < 20.0) {
    std::fprintf(stderr,
                 "warning: host->device bandwidth %.1f GB/s; the GPU link is narrower than x16 "
                 "(current width %s). Reseat the card or check the slot/BIOS lane settings.\n",
                 pcie.h2d_gbs, pcie.during.width.c_str());
  }
  return 0;
}
