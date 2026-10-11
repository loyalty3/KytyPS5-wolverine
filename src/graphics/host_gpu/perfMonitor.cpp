#include "graphics/host_gpu/perfMonitor.h"

#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#endif
#include <vk_mem_alloc.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

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

std::atomic<bool>  g_tex_due {false};
std::atomic<bool>  g_buf_due {false};
std::mutex         g_snapshot_mutex;
TextureSnapshot    g_tex_snapshot {};
BufferSnapshot     g_buf_snapshot {};
bool               g_tex_valid = false;
bool               g_buf_valid = false;

std::array<std::atomic<uint64_t>, static_cast<size_t>(TextureEvent::Count)> g_tex_bytes {};
std::array<std::atomic<uint64_t>, static_cast<size_t>(TextureEvent::Count)> g_tex_count {};
std::atomic<uint64_t> g_address_writes {0};

// Background scan of the large buffers' guest bytes: per buffer, how much of it (in 64 KB
// blocks) holds anything but zeros. Video memory a sparse buffer could leave unbacked.
constexpr double   kScanSeconds = 30.0;
constexpr uint64_t kScanBlock   = 64 * 1024;
struct ScanResult {
	uint64_t address  = 0;
	uint64_t size     = 0;
	uint64_t nonzero  = 0;
	bool     complete = false;
};
std::mutex                                           g_scan_mutex;
std::array<ScanResult, BufferSnapshot::MaxLarge>      g_scan {};
size_t                                               g_scan_count = 0;
std::atomic<bool>                                    g_scan_running {false};
Clock::time_point                                    g_scan_last {};

void ScanLargeBuffers(std::array<std::pair<uint64_t, uint64_t>, BufferSnapshot::MaxLarge> large,
                      size_t count) {
	std::vector<uint8_t> chunk(16 * kScanBlock);
	std::array<ScanResult, BufferSnapshot::MaxLarge> results {};
	for (size_t i = 0; i < count; i++) {
		const auto [address, size] = large[i];
		uint64_t nonzero = 0;
		bool     ok      = true;
		for (uint64_t offset = 0; offset < size && ok; offset += chunk.size()) {
			const auto bytes = std::min<uint64_t>(chunk.size(), size - offset);
			if (!Libs::LibKernel::Memory::TryReadBacking(address + offset, chunk.data(), bytes)) {
				// Unmapped in part: count the chunk as unknown (non-zero) and go on.
				nonzero += bytes;
				continue;
			}
			for (uint64_t block = 0; block < bytes; block += kScanBlock) {
				const auto  length = std::min<uint64_t>(kScanBlock, bytes - block);
				const auto* words  = reinterpret_cast<const uint64_t*>(chunk.data() + block);
				bool        any    = false;
				for (uint64_t w = 0; w < length / 8 && !any; w++) {
					any = words[w] != 0;
				}
				nonzero += any ? length : 0;
			}
		}
		results[i] = {address, size, nonzero, ok};
	}
	{
		std::scoped_lock lock(g_scan_mutex);
		g_scan       = results;
		g_scan_count = count;
	}
	g_scan_running.store(false, std::memory_order_release);
}

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
	std::printf("Perf: KYTY_PERF on (%s, JEEVES patch 6), a report every %.0f s\n", KYTY_BUILD_LABEL,
	            kReportSeconds);
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

bool SnapshotDueTexture() {
	return g_enabled && g_tex_due.exchange(false, std::memory_order_relaxed);
}

bool SnapshotDueBuffer() {
	return g_enabled && g_buf_due.exchange(false, std::memory_order_relaxed);
}

void SetTextureSnapshot(const TextureSnapshot& snapshot) {
	std::scoped_lock lock(g_snapshot_mutex);
	g_tex_snapshot = snapshot;
	g_tex_valid    = true;
}

void SetBufferSnapshot(const BufferSnapshot& snapshot) {
	std::scoped_lock lock(g_snapshot_mutex);
	g_buf_snapshot = snapshot;
	g_buf_valid    = true;
}

void AddTextureEvent(TextureEvent event, uint64_t bytes) {
	if (!g_enabled) {
		return;
	}
	const auto index = static_cast<size_t>(event);
	g_tex_bytes[index].fetch_add(bytes, std::memory_order_relaxed);
	g_tex_count[index].fetch_add(1, std::memory_order_relaxed);
}

void AddAddressWriteShader() {
	if (g_enabled) {
		g_address_writes.fetch_add(1, std::memory_order_relaxed);
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
	// Per heap: device-local (the card's own memory) against the budget Windows grants this
	// process, and what sits in system-memory heaps (spilled buffers and images: every GPU access
	// to them crosses PCIe).
	uint64_t card_used = 0, card_budget = 0, system_used = 0;
	// VMA's own view of the card: memory objects it holds (blocks and dedicated allocations)
	// against the bytes allocated in them. The difference is room inside blocks that only a
	// block's last allocation going would return.
	uint64_t card_blocks = 0, card_allocated = 0, card_block_count = 0, card_allocation_count = 0;
	if (g_graphics != nullptr && g_graphics->allocator != nullptr && g_graphics->CanReportMemoryUsage()) {
		VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
		vmaGetHeapBudgets(g_graphics->allocator, budgets);
		const auto& properties = g_graphics->GetPhysicalDeviceMemoryProperties();
		for (uint32_t heap = 0; heap < properties.memoryHeapCount; heap++) {
			const bool local = static_cast<bool>(properties.memoryHeaps[heap].flags &
			                                     vk::MemoryHeapFlagBits::eDeviceLocal);
			if (local) {
				card_used += budgets[heap].usage;
				card_budget += budgets[heap].budget;
				card_blocks += budgets[heap].statistics.blockBytes;
				card_allocated += budgets[heap].statistics.allocationBytes;
				card_block_count += budgets[heap].statistics.blockCount;
				card_allocation_count += budgets[heap].statistics.allocationCount;
			} else {
				system_used += budgets[heap].statistics.blockBytes;
			}
		}
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
	{
		std::scoped_lock lock(g_snapshot_mutex);
		if (g_tex_valid || g_buf_valid) {
			std::printf("Perf mem: card %" PRIu64 "/%" PRIu64 " MB (Windows budget), emulator data in system memory %" PRIu64
			            " MB | ",
			            card_used >> 20u, card_budget >> 20u, system_used >> 20u);
			std::printf("textures %" PRIu64 " MB in %" PRIu64 " images (heap-pinned %" PRIu64
			            " MB, targets %" PRIu64 " MB, GPU-written %" PRIu64
			            " MB; collector mark %" PRIu64 " MB) | buffers %" PRIu64 " MB in %" PRIu64
			            " (collector mark %" PRIu64 " MB) | page table %" PRIu64 " MB %s\n",
			            g_tex_snapshot.total_bytes >> 20u, g_tex_snapshot.images,
			            g_tex_snapshot.pinned_bytes >> 20u, g_tex_snapshot.target_bytes >> 20u,
			            g_tex_snapshot.gpu_bytes >> 20u, g_tex_snapshot.pressure_mark >> 20u,
			            g_buf_snapshot.total_bytes >> 20u, g_buf_snapshot.buffers,
			            g_buf_snapshot.trigger_mark >> 20u, g_buf_snapshot.pagetable_bytes >> 20u,
			            g_buf_snapshot.pagetable_sparse ? "(sparse)" : "(dense)");
			std::printf("Perf vram: VMA holds %" PRIu64 " MB in %" PRIu64 " memory objects, %" PRIu64
			            " MB of it allocated (%" PRIu64 " allocations; %" PRIu64
			            " MB free inside blocks) | images: %" PRIu64 " MB real, %" PRIu64
			            " MB duplicating guest bytes of another image, %" PRIu64 " with memory of their own (%" PRIu64
			            " MB)\n",
			            card_blocks >> 20u, card_block_count, card_allocated >> 20u,
			            card_allocation_count, (card_blocks - std::min(card_blocks, card_allocated)) >> 20u,
			            g_tex_snapshot.real_bytes >> 20u, g_tex_snapshot.duplicate_bytes >> 20u,
			            GraphicContext::DedicatedImageCount(), GraphicContext::DedicatedImageBytes() >> 20u);
		}
	}
	{
		std::array<uint64_t, static_cast<size_t>(TextureEvent::Count)> bytes {};
		std::array<uint64_t, static_cast<size_t>(TextureEvent::Count)> count {};
		for (size_t i = 0; i < bytes.size(); i++) {
			bytes[i] = g_tex_bytes[i].exchange(0, std::memory_order_relaxed) >> 20u;
			count[i] = g_tex_count[i].exchange(0, std::memory_order_relaxed);
		}
		const auto address_writes = g_address_writes.exchange(0, std::memory_order_relaxed);
		std::printf("Perf churn: textures made %" PRIu64 " (%" PRIu64 " MB); freed by the collector %" PRIu64
		            " (%" PRIu64 " MB), unsampled bindless %" PRIu64 " (%" PRIu64
		            " MB), guest dropped %" PRIu64 " (%" PRIu64 " MB), guest unmapped %" PRIu64 " (%" PRIu64
		            " MB), replaced %" PRIu64 " (%" PRIu64 " MB) | shaders storing through pointers: %" PRIu64
		            " runs\n",
		            count[0], bytes[0], count[1], bytes[1], count[2], bytes[2], count[3], bytes[3],
		            count[4], bytes[4], count[5], bytes[5], address_writes);
	}
	{
		std::scoped_lock lock(g_snapshot_mutex);
		if (g_buf_valid && g_buf_snapshot.large_count != 0) {
			std::scoped_lock scan_lock(g_scan_mutex);
			std::printf("Perf large buffers:");
			for (size_t i = 0; i < g_buf_snapshot.large_count; i++) {
				const auto [address, size] = g_buf_snapshot.large[i];
				std::printf(" %" PRIu64 " MB at 0x%010" PRIx64 ", %" PRIu64 " MB of video memory", size >> 20u,
				            address, g_buf_snapshot.large_resident[i] >> 20u);
				for (size_t j = 0; j < g_scan_count; j++) {
					if (g_scan[j].address == address && g_scan[j].size == size) {
						std::printf(" (non-zero guest data %" PRIu64 " MB)", g_scan[j].nonzero >> 20u);
					}
				}
				std::printf(i + 1 < g_buf_snapshot.large_count ? ";" : "");
			}
			std::printf(" | zero uploads left unbacked %" PRIu64 " MB\n",
			            g_buf_snapshot.sparse_skipped_bytes >> 20u);
			const bool due = std::chrono::duration<double>(now - g_scan_last).count() >= kScanSeconds;
			if (due && !g_scan_running.exchange(true, std::memory_order_acq_rel)) {
				g_scan_last = now;
				std::thread(ScanLargeBuffers, g_buf_snapshot.large, g_buf_snapshot.large_count).detach();
			}
		}
	}
	std::fflush(stdout);
	// Ask both caches for a fresh snapshot for the next report.
	g_tex_due.store(true, std::memory_order_relaxed);
	g_buf_due.store(true, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::PerfMonitor
