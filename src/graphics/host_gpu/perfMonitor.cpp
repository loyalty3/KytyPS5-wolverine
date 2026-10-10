#include "graphics/host_gpu/perfMonitor.h"

#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::PerfMonitor {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kReportSeconds = 5.0;

bool            g_enabled  = false;
GraphicContext* g_graphics = nullptr;

// Nanoseconds, so plain integer atomics suffice.
std::atomic<int64_t>  g_idle_ns {0};
std::atomic<int64_t>  g_blocked_ns {0};
std::atomic<int64_t>  g_readback_ns {0};
std::atomic<int64_t>  g_readback_max_ns {0};
std::atomic<uint64_t> g_frames {0};

// Owned by the GPU thread (Tick).
Clock::time_point g_window_start {};
std::array<int64_t, static_cast<size_t>(Profiler::Counter::Total)> g_last_counters {};

int64_t ToNs(double seconds) {
	return static_cast<int64_t>(seconds * 1e9);
}

int64_t Counter(Profiler::Counter counter) {
	return Profiler::g_counter_slots[static_cast<uint32_t>(counter)].value.load(
	    std::memory_order_relaxed);
}

// The change of a counter since the last report. With a Tracy server running, its per-frame
// plots reset the counters themselves (PlotCounters), and the deltas are meaningless.
int64_t Delta(Profiler::Counter counter) {
	const auto index = static_cast<size_t>(counter);
	const auto now   = Counter(counter);
	const auto delta = now - g_last_counters[index];
	g_last_counters[index] = now;
	return delta;
}

} // namespace

void Initialize(GraphicContext* graphics) {
	const char* value = std::getenv("KYTY_PERF");
	g_enabled         = value != nullptr && value[0] != '\0' && value[0] != '0';
	if (!g_enabled) {
		return;
	}
	g_graphics = graphics;
	// The draw, dispatch, wait and readback counters (Profiler::Add) count only while this is on.
	Profiler::g_counters = true;
	g_window_start       = Clock::now();
	std::printf("Perf: KYTY_PERF on, a report every %.0f s\n", kReportSeconds);
	std::fflush(stdout);
}

bool Enabled() {
	return g_enabled;
}

void AddGpuThreadIdle(double seconds) {
	if (g_enabled) {
		g_idle_ns.fetch_add(ToNs(seconds), std::memory_order_relaxed);
	}
}

void AddGpuThreadBlocked(double seconds) {
	if (g_enabled) {
		g_blocked_ns.fetch_add(ToNs(seconds), std::memory_order_relaxed);
	}
}

void AddGameReadbackWait(double seconds) {
	if (!g_enabled) {
		return;
	}
	const auto ns = ToNs(seconds);
	g_readback_ns.fetch_add(ns, std::memory_order_relaxed);
	auto max = g_readback_max_ns.load(std::memory_order_relaxed);
	while (ns > max &&
	       !g_readback_max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
	}
}

void FramePresented() {
	if (g_enabled) {
		g_frames.fetch_add(1, std::memory_order_relaxed);
	}
}

void Tick() {
	if (!g_enabled) {
		return;
	}
	const auto now     = Clock::now();
	const auto elapsed = std::chrono::duration<double>(now - g_window_start).count();
	if (elapsed < kReportSeconds) {
		return;
	}
	g_window_start = now;

	const auto window_ns  = elapsed * 1e9;
	const auto idle       = static_cast<double>(g_idle_ns.exchange(0, std::memory_order_relaxed));
	const auto blocked    = static_cast<double>(g_blocked_ns.exchange(0, std::memory_order_relaxed));
	const auto readback   = static_cast<double>(g_readback_ns.exchange(0, std::memory_order_relaxed));
	const auto readback_max =
	    static_cast<double>(g_readback_max_ns.exchange(0, std::memory_order_relaxed));
	const auto frames     = g_frames.exchange(0, std::memory_order_relaxed);
	const double fps      = static_cast<double>(frames) / elapsed;
	const double per_frame_divisor = frames != 0 ? static_cast<double>(frames) : 1.0;

	const auto draws       = Delta(Profiler::Counter::Draws);
	const auto dispatches  = Delta(Profiler::Counter::Dispatches);
	const auto gpu_wait_us = Delta(Profiler::Counter::GpuWaitTime);
	const auto readbacks   = Delta(Profiler::Counter::Readbacks);
	const auto rb_bytes    = Delta(Profiler::Counter::ReadbackBytes);
	const auto up_bytes    = Delta(Profiler::Counter::UploadBytes);
	const auto shaders     = Delta(Profiler::Counter::ShadersCompiled);
	const auto shader_us   = Delta(Profiler::Counter::ShaderCompileTime);
	const auto pipelines   = Delta(Profiler::Counter::GraphicsPipelinesBuilt) +
	                       Delta(Profiler::Counter::ComputePipelinesBuilt);
	const auto pipeline_us = Delta(Profiler::Counter::PipelineBuildTime);
	const auto skipped_pipeline = Delta(Profiler::Counter::SkippedPipelinePending);
	const auto skipped_list     = Delta(Profiler::Counter::SkippedSkipList);
	const auto skipped_gave_up  = Delta(Profiler::Counter::SkippedGaveUp);
	const auto drains           = Delta(Profiler::Counter::RecordingDrains);
	const auto created          = Delta(Profiler::Counter::ImagesCreated);
	const auto destroyed        = Delta(Profiler::Counter::ImagesDestroyed);

	const double busy_ns   = std::max(0.0, window_ns - idle - blocked);
	const double busy_pct  = 100.0 * busy_ns / window_ns;
	const double gpu_wait_pct = 100.0 * static_cast<double>(gpu_wait_us) * 1000.0 / window_ns;
	const double work_items   = static_cast<double>(draws + dispatches);
	// The GPU thread's own time per draw or dispatch, host GPU waits taken out.
	const double us_per_item =
	    work_items > 0 ? (busy_ns - static_cast<double>(gpu_wait_us) * 1000.0) / 1000.0 / work_items
	                   : 0.0;

	uint64_t vram_used   = 0;
	uint64_t vram_budget = 0;
	if (g_graphics != nullptr && g_graphics->CanReportMemoryUsage()) {
		vram_used   = g_graphics->GetDeviceMemoryUsage() >> 20u;
		vram_budget = g_graphics->GetTotalMemoryBudget() >> 20u;
	}
	uint64_t commit_limit = 0;
	uint64_t commit_free  = 0;
	Common::VirtualMemory::QueryCommit(commit_limit, commit_free);

	std::printf(
	    "Perf: %.1f fps | GPU thread busy %.0f%% (host GPU waits %.0f%%), idle %.0f%%, blocked "
	    "%.0f%% | draws %.0f/frame, dispatches %.0f/frame, %.1f us each | game readback waits "
	    "%.0f ms/s (max %.1f ms), %" PRId64 " readbacks %.1f MB/s | uploads %.1f MB/s | VRAM %" PRIu64
	    "/%" PRIu64 " MB | commit free %" PRIu64 " MB | shaders %" PRId64 " (%.0f ms), pipelines %" PRId64
	    " (%.0f ms), draws skipped compiling %" PRId64 ", skip list %" PRId64 ", gave up %" PRId64
	    " | recording drains %" PRId64 " | images +%" PRId64 "/-%" PRId64 "%s\n",
	    fps, busy_pct, gpu_wait_pct, 100.0 * idle / window_ns, 100.0 * blocked / window_ns,
	    static_cast<double>(draws) / per_frame_divisor,
	    static_cast<double>(dispatches) / per_frame_divisor, us_per_item,
	    readback / 1e6 / elapsed, readback_max / 1e6, readbacks,
	    static_cast<double>(rb_bytes) / 1048576.0 / elapsed,
	    static_cast<double>(up_bytes) / 1048576.0 / elapsed, vram_used, vram_budget,
	    commit_free >> 20u, shaders, static_cast<double>(shader_us) / 1000.0, pipelines,
	    static_cast<double>(pipeline_us) / 1000.0, skipped_pipeline, skipped_list, skipped_gave_up,
	    drains, created, destroyed,
	    Profiler::Connected() ? " (Tracy connected: counters unreliable)" : "");
	std::fflush(stdout);
}

} // namespace Libs::Graphics::PerfMonitor
