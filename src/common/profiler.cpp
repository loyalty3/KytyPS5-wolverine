#include "common/profiler.h"

#include "common/emulatorConfig.h"

#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <tracy/Tracy.hpp>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace Profiler {

namespace {

enum class CounterKind : uint8_t { Events, Bytes, TimeUs, Gauge };

struct CounterInfo {
	const char* name;
	CounterKind kind;
	uint32_t    color;
};

// Plot names are compared by pointer: each must stay this one string.
constexpr CounterInfo kCounterInfo[] = {
    {"Draws", CounterKind::Events, 0x64b5f6},
    {"Dispatches", CounterKind::Events, 0x4dd0e1},
    {"Skipped: shader gave up", CounterKind::Events, 0xff5252},
    {"Skipped: skip list", CounterKind::Events, 0x9e9e9e},
    {"Skipped: resources unreadable", CounterKind::Events, 0xff8a80},
    {"Skipped: pipeline compiling", CounterKind::Events, 0xffab40},
    {"Shaders compiled", CounterKind::Events, 0x81c784},
    {"Shaders gave up", CounterKind::Events, 0xff5252},
    {"Shader compile ms", CounterKind::TimeUs, 0xa5d6a7},
    {"Graphics pipelines built", CounterKind::Events, 0x66bb6a},
    {"Compute pipelines built", CounterKind::Events, 0x43a047},
    {"Pipeline build ms", CounterKind::TimeUs, 0xc8e6c9},
    {"Pipelines pending", CounterKind::Gauge, 0xffd54f},
    {"SRT refreshes", CounterKind::Events, 0xce93d8},
    {"SRT replayed", CounterKind::Events, 0xba68c8},
    {"SRT abandoned", CounterKind::Events, 0xff6e40},
    {"SRT recorded", CounterKind::Events, 0xf48fb1},
    {"GPU waits", CounterKind::Events, 0xe57373},
    {"GPU wait ms", CounterKind::TimeUs, 0xef5350},
    {"Readbacks", CounterKind::Events, 0xffb74d},
    {"Readback bytes", CounterKind::Bytes, 0xffa726},
    {"Command buffers submitted", CounterKind::Events, 0x90a4ae},
    {"Recording drains", CounterKind::Events, 0xff7043},
    {"Upload bytes", CounterKind::Bytes, 0x4fc3f7},
    {"Images created", CounterKind::Events, 0xaed581},
    {"Images destroyed", CounterKind::Events, 0xdce775},
    {"Bindless keys resolved", CounterKind::Events, 0x7986cb},
};
static_assert(std::size(kCounterInfo) == static_cast<size_t>(Counter::Total));

void ConfigurePlots() {
	for (const auto& info: kCounterInfo) {
		const auto format = info.kind == CounterKind::Bytes ? tracy::PlotFormatType::Memory
		                                                    : tracy::PlotFormatType::Number;
		tracy::Profiler::ConfigurePlot(info.name, format, true, info.kind != CounterKind::TimeUs,
		                               info.color);
	}
}

void PlotCounters() {
	for (size_t index = 0; index < std::size(kCounterInfo); index++) {
		const auto& info  = kCounterInfo[index];
		auto&       value = g_counter_slots[index].value;
		switch (info.kind) {
			case CounterKind::Gauge:
				tracy::Profiler::PlotData(info.name, value.load(std::memory_order_relaxed));
				break;
			case CounterKind::TimeUs:
				tracy::Profiler::PlotData(
				    info.name, static_cast<double>(value.exchange(0, std::memory_order_relaxed)) / 1000.0);
				break;
			default:
				tracy::Profiler::PlotData(info.name, value.exchange(0, std::memory_order_relaxed));
				break;
		}
	}
}

} // namespace

namespace {

// Scheduling for the emulator's own threads, by the name each gives itself at start.
// The GPU thread is the frame's critical path, and the recording, submission and draw-prep
// threads feed it; on a hybrid CPU (Core Ultra: 8 performance and 12 efficient cores) Windows may
// run any of them on an efficient core, or let the six shader translators and four pipeline
// compilers crowd them off the performance cores. Critical threads get a higher priority and
// are marked high-QoS (opted out of EcoQoS, which steers threads to efficient cores); the
// translators and compilers run below normal. KYTY_THREAD_PRIORITY=0 leaves all as they were.
enum class ThreadRole : uint8_t { Normal, Critical, Background };

[[maybe_unused]] ThreadRole RoleOf(const char* name) {
	static constexpr const char* kCritical[]   = {"Thread_Gpu", "GpuRecord", "GpuQueueSubmit",
	                                              "Thread_DrawPrep", "StagePrep"};
	static constexpr const char* kBackground[] = {"ShaderTranslator", "PipelineCompiler"};
	for (const char* candidate: kCritical) {
		if (std::strcmp(name, candidate) == 0) return ThreadRole::Critical;
	}
	for (const char* candidate: kBackground) {
		if (std::strcmp(name, candidate) == 0) return ThreadRole::Background;
	}
	return ThreadRole::Normal;
}

bool ThreadPriorityEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_THREAD_PRIORITY");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

void ApplyThreadRole(const char* name) {
	if (name == nullptr || !ThreadPriorityEnabled()) {
		return;
	}
#if defined(_WIN32)
	static const bool process_done = [] {
#if defined(PROCESS_POWER_THROTTLING_CURRENT_VERSION)
		// The whole process out of EcoQoS: Windows otherwise may throttle it when its window
		// is not in front, and EcoQoS threads are scheduled on the efficient cores.
		PROCESS_POWER_THROTTLING_STATE state {};
		state.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
		state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
		state.StateMask   = 0;
		(void)SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state,
		                            sizeof(state));
#endif
		return true;
	}();
	(void)process_done;
	const auto role = RoleOf(name);
	if (role == ThreadRole::Normal) {
		return;
	}
	const auto thread = GetCurrentThread();
	(void)SetThreadPriority(thread, role == ThreadRole::Critical ? THREAD_PRIORITY_HIGHEST
	                                                             : THREAD_PRIORITY_BELOW_NORMAL);
#if defined(THREAD_POWER_THROTTLING_CURRENT_VERSION)
	if (role == ThreadRole::Critical) {
		THREAD_POWER_THROTTLING_STATE state {};
		state.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
		state.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
		state.StateMask   = 0;
		(void)SetThreadInformation(thread, ThreadPowerThrottling, &state, sizeof(state));
	}
#endif
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1) < 16) {
		std::printf("Thread %s: %s priority\n", name,
		            role == ThreadRole::Critical ? "high (performance cores preferred)"
		                                         : "below normal");
	}
#endif
}

} // namespace

void SetThreadName(const char* name) {
	ApplyThreadRole(name);
	if (tracy::ProfilerAvailable() && name != nullptr) {
		tracy::SetThreadName(name);
	}
}

void MarkFrame() {
	if (tracy::ProfilerAvailable()) {
		FrameMark;
		if (g_counters) {
			PlotCounters();
		}
	}
}

bool Connected() {
	return tracy::ProfilerAvailable() && tracy::GetProfiler().IsConnected();
}

void Message(uint32_t color, const char* format, ...) {
	if (!Connected()) {
		return;
	}
	char    text[512];
	va_list args;
	va_start(args, format);
	const int size = std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	if (size <= 0) {
		return;
	}
	tracy::Profiler::LogString(tracy::MessageSourceType::User, tracy::MessageSeverity::Info, color,
	                           0, std::min(static_cast<size_t>(size), sizeof(text) - 1), text);
}

void Phases::SetText(const char* format, ...) {
	if (!Connected()) {
		m_text_size = 0;
		return;
	}
	va_list args;
	va_start(args, format);
	const int size = std::vsnprintf(m_text, sizeof(m_text), format, args);
	va_end(args);
	m_text_size = size <= 0 ? 0 : std::min(static_cast<size_t>(size), sizeof(m_text) - 1);
	if (m_zone && m_text_size != 0) {
		m_zone->Text(m_text, m_text_size);
	}
}

void Phases::Begin(const tracy::SourceLocationData* location) {
	m_zone.reset();
	if (!tracy::ProfilerAvailable()) {
		return;
	}
	m_zone.emplace(location, TRACY_CALLSTACK, true);
	if (m_text_size != 0) {
		m_zone->Text(m_text, m_text_size);
	}
}

namespace {
thread_local CommandZone* t_command_zone = nullptr;
} // namespace

CommandZone::CommandZone(const tracy::SourceLocationData* location) {
	if (Connected()) {
		m_zone.emplace(location, TRACY_CALLSTACK, true);
		m_previous     = t_command_zone;
		t_command_zone = this;
	}
}

CommandZone::~CommandZone() {
	if (m_zone) {
		t_command_zone = m_previous;
	}
}

void CommandZone::Annotate(const char* format, ...) {
	auto* zone = t_command_zone;
	if (zone == nullptr || !zone->m_zone) {
		return;
	}
	char    text[160];
	va_list args;
	va_start(args, format);
	const int size = std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	if (size > 0) {
		zone->m_zone->Text(text, std::min(static_cast<size_t>(size), sizeof(text) - 1));
	}
}

void Initialize() {
	if (const char* zones = std::getenv("KYTY_PROFILE_ZONES"); zones != nullptr && zones[0] == '0') {
		g_zones = false;
	}
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		TracySetProgramName("KytyPS5");
		ConfigurePlots();
		g_counters = true;
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
	}
}

void Shutdown() {
	g_counters = false;
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler

namespace Profiler {

PerfZoneStats* PerfZoneTable() {
	static PerfZoneStats table[kPerfZoneSlots] {};
	return table;
}

int64_t PerfZoneNow() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

static PerfZoneStats* FindSlot(const tracy::SourceLocationData* location) {
	auto*      table = PerfZoneTable();
	const auto hash  = (reinterpret_cast<uintptr_t>(location) >> 4u) * 0x9E3779B97F4A7C15ull;
	for (size_t probe = 0; probe < kPerfZoneSlots; probe++) {
		auto& slot = table[(hash + probe) & (kPerfZoneSlots - 1)];
		if (slot.location == location || slot.location == nullptr) {
			slot.location = location;
			return &slot;
		}
	}
	return nullptr;
}

static const char* PerfZoneLabel(const tracy::SourceLocationData* location) {
	return location->name != nullptr ? location->name : location->function;
}

void PerfZone::Finish() {
	const auto elapsed = PerfZoneNow() - m_start;
	s_top              = m_parent;
	if (m_parent != nullptr) {
		m_parent->m_child_ns += elapsed;
	}
	if (auto* slot = FindSlot(m_location); slot != nullptr) {
		slot->self_ns += elapsed - m_child_ns;
		slot->total_ns += elapsed;
		slot->calls++;
	}
	const char* name = PerfZoneLabel(m_location);
	if (name != nullptr && std::strncmp(name, "MasterSemaphore::Wait", 21) == 0) {
		// Charged to the nearest enclosing zone that is not itself a wait, nor the download
		// plumbing every readback goes through (which named only DownloadBufferCopies, not the
		// work that needed the bytes); the plumbing is kept when nothing else encloses it.
		const auto plumbing = [](const char* label) {
			return label != nullptr &&
			       (std::strstr(label, "Wait") != nullptr || std::strstr(label, "Download") != nullptr ||
			        std::strstr(label, "ReadMemory") != nullptr);
		};
		auto* owner = m_parent;
		while (owner != nullptr) {
			const char* owner_name = PerfZoneLabel(owner->m_location);
			if (owner_name == nullptr || std::strstr(owner_name, "Wait") == nullptr) {
				break;
			}
			owner = owner->m_parent;
		}
		for (auto* outer = owner; outer != nullptr; outer = outer->m_parent) {
			if (!plumbing(PerfZoneLabel(outer->m_location))) {
				owner = outer;
				break;
			}
		}
		if (owner != nullptr) {
			if (auto* slot = FindSlot(owner->m_location); slot != nullptr) {
				slot->gpu_wait_ns += elapsed;
				slot->gpu_wait_calls++;
			}
		}
	}
}

} // namespace Profiler
