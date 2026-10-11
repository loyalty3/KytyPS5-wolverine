#ifndef KYTY_GRAPHICS_HOST_GPU_PERF_MONITOR_H_
#define KYTY_GRAPHICS_HOST_GPU_PERF_MONITOR_H_

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// KYTY_PERF=1: a one-line report every 5 seconds on stdout, whatever --printf-direction says, so
// a run without logging (logging itself costs frames) still says where the time goes:
//
//   Perf: 18.4 fps | GPU thread busy 97% (host GPU waits 6%), idle 1%, blocked 2% |
//         draws 2950/frame, dispatches 610/frame, 31 us/draw+dispatch |
//         game-thread readback waits 412 ms/s | VRAM 10840/10264 MB | ...
//
// - busy near 100 %: the GPU thread (the emulated command processor) sets the frame rate;
// - idle high: the GPU thread waits for the game, so the game's own threads are the limit
//   (readback waits on GPU-written memory are the usual reason);
// - host GPU waits high: the PC's graphics card is the limit;
// - VRAM over budget: video memory pages over PCIe and everything slows.
//
// The counters are the ones the Tracy plots use (Profiler::Counter); KYTY_PERF turns them on
// without a profiler. Off by default; nothing here costs anything when off.
namespace PerfMonitor {

void Initialize(GraphicContext* graphics);
bool Enabled();

// The GPU thread's time waiting for work (nothing queued) and waiting with work blocked behind
// a guest wait. Called by the GPU thread only.
void AddGpuThreadIdle(double seconds);
void AddGpuThreadBlocked(double seconds);

// A game thread waited this long for GPU-written memory to come back (any thread).
void AddGameReadbackWait(double seconds);

// A frame was presented (any thread).
void FramePresented();

// Prints the report when 5 seconds have passed. Called by the GPU thread once per loop.
void Tick();

// What the texture and buffer caches hold, for the next report: each report asks for a fresh
// snapshot (SnapshotDue returns true once per report and cache), the cache's collector answers.
struct TextureSnapshot {
	uint64_t images        = 0;
	uint64_t total_bytes   = 0; // every registered image
	uint64_t pinned_bytes  = 0; // bindless-pinned: the guest's texture heaps refer to them
	uint64_t target_bytes  = 0; // bound as a render or depth target
	uint64_t gpu_bytes     = 0; // written by the GPU (kept until written back)
	uint64_t pressure_mark = 0; // the collector's pressure threshold (collectable bytes)
};
struct BufferSnapshot {
	uint64_t buffers      = 0;
	uint64_t total_bytes  = 0;
	uint64_t trigger_mark = 0;
};
bool SnapshotDueTexture();
bool SnapshotDueBuffer();
void SetTextureSnapshot(const TextureSnapshot& snapshot);
void SetBufferSnapshot(const BufferSnapshot& snapshot);

} // namespace PerfMonitor

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_HOST_GPU_PERF_MONITOR_H_ */
