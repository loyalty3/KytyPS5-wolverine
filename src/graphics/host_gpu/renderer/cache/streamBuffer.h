#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <string>
#include <vector>

VK_DEFINE_HANDLE(VmaAllocation)

namespace Libs::Graphics {

class CommandBuffer;
class CommandScheduler;
struct StreamBufferTestAccess;
struct GraphicContext;

enum class MemoryUsage : uint8_t {
	DeviceLocal,
	Upload,
	Download,
	Stream,
};

inline constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

inline constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

class Buffer {
public:
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	       uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size);
	// A sparse-residency buffer: address space only, no memory until MakeResident binds it.
	// Unbound ranges read as zero and drop writes (residencyNonResidentStrict). Needs
	// GraphicContext::sparse_buffer_enabled.
	struct SparseTag {};
	// chunk_min: the smallest piece bound at a time (rounded up to the device's sparse block).
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, SparseTag, uint64_t cpu_address,
	       vk::BufferUsageFlags flags, uint64_t size, uint64_t chunk_min);
	~Buffer();
	KYTY_CLASS_NO_COPY(Buffer);

	[[nodiscard]] bool IsSparse() const noexcept { return m_sparse; }
	// Sparse only (a dense buffer returns at once): binds memory under [offset, offset+size) and
	// zero-fills what it binds before the scheduler's next submission.
	// Binding waits for submitted work first (it may read these pages expecting zero) and
	// zeroes in a command buffer of its own; the one being recorded is not split.
	// GPU thread only. Memory stays bound until the buffer is destroyed.
	// cold: the range is backed only to be safe (a write binding over memory the scene has not
	// filled); its chunks go to system memory where the buffer takes it, and move to video memory
	// when data arrives (PromoteChunks).
	void MakeResident(uint64_t offset, uint64_t size, bool cold = false);
	// The same for several ranges (offset, size) at once: one wait, one bind.
	void MakeResidentRanges(std::span<const std::pair<uint64_t, uint64_t>> ranges, bool cold = false);
	// Why the binds made on this thread are made (MakeResident*), for the large-bind report and
	// the memory chosen. data: the chunks get bytes the scene uses (uploads, merges, the BDA page
	// table), so they stay in video memory. reported: the caller names large binds itself.
	// A bind of 64 MB or more that is not data goes to system memory: in the second half of
	// 2026-10-11 the geometry pool went from 1.2 GB to all 2.7 GB of video memory in one 689 ms
	// bind that no named path reported, and the card ran out.
	struct BindContext {
		const char* why      = nullptr;
		bool        data     = false;
		bool        reported = false;
	};
	class BindScope {
	public:
		BindScope(const char* why, bool data, bool reported): m_saved(s_bind) {
			s_bind = {why, data, reported};
		}
		~BindScope() { s_bind = m_saved; }
		BindScope(const BindScope&)            = delete;
		BindScope& operator=(const BindScope&) = delete;

	private:
		BindContext m_saved;
	};
	// Chunks in system memory (ChunkState ChunkHost) move to video memory, contents kept.
	void PromoteChunks(std::span<const uint64_t> chunks);
	static constexpr uint8_t ChunkUnbound = 0;
	static constexpr uint8_t ChunkDevice  = 1;
	static constexpr uint8_t ChunkHost    = 2;
	[[nodiscard]] uint8_t ChunkState(uint64_t offset) const noexcept {
		return m_sparse ? m_chunks[offset / m_chunk_size] : ChunkDevice;
	}
	// Sparse: whether the chunk holding this offset is bound (dense: always).
	[[nodiscard]] bool     IsResident(uint64_t offset) const noexcept {
		return !m_sparse || m_chunks[offset / m_chunk_size] != 0;
	}
	[[nodiscard]] uint64_t ChunkSize() const noexcept { return m_sparse ? m_chunk_size : m_size; }
	// Sparse: bytes of [offset, offset+size) whose chunks are not bound yet (whole chunks).
	[[nodiscard]] uint64_t UnboundBytes(uint64_t offset, uint64_t size) const noexcept {
		if (!m_sparse || size == 0 || offset >= m_size) {
			return 0;
		}
		const auto last  = (std::min(offset + size, m_size) - 1) / m_chunk_size;
		uint64_t   bytes = 0;
		for (auto chunk = offset / m_chunk_size; chunk <= last; chunk++) {
			bytes += m_chunks[chunk] == 0 ? m_chunk_size : 0;
		}
		return bytes;
	}
	// Bytes bound so far (sparse), or the whole size (dense).
	[[nodiscard]] uint64_t ResidentBytes() const noexcept; // in video memory
	[[nodiscard]] uint64_t HostResidentBytes() const noexcept; // sparse chunks in system memory
	// Whether cold chunks can live in system memory. NVIDIA on Windows offers sparse buffers video
	// memory only: the cold chunks of Patch 10 fell back to it, and the geometry pool filled the
	// card all the same (2026-10-11, the line said "system memory" and the pool showed 0 MB there).
	[[nodiscard]] bool     CanHoldHost() const noexcept { return m_host_type_bits != 0; }

	[[nodiscard]] vk::Buffer         Handle() const noexcept { return m_buffer; }
	[[nodiscard]] uint64_t           Size() const noexcept { return m_size; }
	[[nodiscard]] std::span<uint8_t> Mapped() const noexcept { return m_mapped; }
	[[nodiscard]] bool               IsCoherent() const noexcept { return m_coherent; }
	[[nodiscard]] MemoryUsage        Usage() const noexcept { return m_usage; }
	[[nodiscard]] uint64_t           CpuAddress() const noexcept { return m_cpu_address; }
	[[nodiscard]] vk::DeviceAddress BufferDeviceAddress() const noexcept;
	[[nodiscard]] bool HasDeviceAddress() const noexcept { return m_device_address != 0; }
	[[nodiscard]] uint64_t           Offset(uint64_t address) const noexcept {
		return address - m_cpu_address;
	}
	[[nodiscard]] bool IsInBounds(uint64_t address, uint64_t size) const noexcept;
	void               IncreaseStreamScore(int score) noexcept { stream_score += score; }
	[[nodiscard]] int  StreamScore() const noexcept { return stream_score; }
	void               Flush(uint64_t offset, uint64_t size);
	void               Invalidate(uint64_t offset, uint64_t size);
	void CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
	              uint64_t destination_offset, uint64_t size,
	              vk::AccessFlags source_before      = vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags destination_before = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags source_after       = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags destination_after  = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite);
	void Fill(uint64_t offset, uint64_t size, uint32_t value);

	// BufferCache state lives directly on the resource.
	bool   is_deleted   = false;
	// The last scheduler tick a draw or dispatch bound the buffer in (the buffer cache's).
	uint64_t last_use_tick = 0;
	int    stream_score = 0;
	size_t lru_id       = 0;

protected:
	[[nodiscard]] GraphicContext&   Graphics() const noexcept { return *m_graphics; }
	[[nodiscard]] CommandScheduler& Scheduler() const noexcept { return *m_scheduler; }

private:
	[[nodiscard]] vk::BufferMemoryBarrier Barrier(uint64_t offset, uint64_t size,
	                                              vk::AccessFlags source,
	                                              vk::AccessFlags destination) const;

	GraphicContext*               m_graphics    = nullptr;
	CommandScheduler*             m_scheduler   = nullptr;
	MemoryUsage                   m_usage       = MemoryUsage::DeviceLocal;
	uint64_t                      m_cpu_address = 0;
	vk::DeviceAddress             m_device_address = 0;
	vk::Buffer                    m_buffer     = nullptr;
	VmaAllocation                 m_allocation = nullptr;
	uint64_t                      m_size;
	bool                          m_coherent = false;
	std::span<uint8_t>            m_mapped;
	// Sparse: one allocation per bound chunk, null where nothing is bound.
	bool                          m_sparse          = false;
	uint64_t                      m_chunk_size      = 0;
	uint32_t                      m_memory_type_bits = 0;
	std::vector<uint8_t>          m_chunks; // ChunkUnbound, ChunkDevice or ChunkHost
	// Memory comes from 2 MB slabs, cut into chunks: one VMA allocation per 64 KB chunk made
	// 21,000 of them for the geometry pool, and past the budget VMA served each from a memory
	// object of its own (5,067 objects, 2026-10-11).
	struct Slab {
		VmaAllocation   allocation = nullptr;
		vk::DeviceMemory memory    = nullptr;
		uint64_t        offset     = 0;
		uint64_t        size       = 0;
		uint64_t        used       = 0;
	};
	static thread_local BindContext s_bind;
	std::vector<Slab>             m_slabs;      // video memory
	std::vector<Slab>             m_host_slabs; // system memory (cold chunks)
	uint32_t                      m_host_type_bits = 0;
	uint64_t                      m_host_bytes     = 0;
	struct ChunkMemory {
		vk::DeviceMemory memory = nullptr;
		uint64_t         offset = 0;
		bool             host   = false;
	};
	ChunkMemory AllocateChunk(bool cold);
	void        BindAndWait(std::span<const vk::SparseMemoryBind> binds);
	template <typename F>
	void RunOneShot(F&& fn);
	void WaitSubmitted();
	uint64_t                      m_resident_bytes  = 0;
	vk::CommandPool               m_pool            = nullptr;
};

class StreamBuffer final: public Buffer {
public:
	StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	             uint64_t size);

	[[nodiscard]] std::pair<uint8_t*, uint64_t> Map(uint64_t size, uint64_t alignment = 0,
	                                                bool allow_wait = true);
	void                                        Commit();
	[[nodiscard]] uint64_t Copy(const void* source, uint64_t size, uint64_t alignment = 0);

private:
	friend struct StreamBufferTestAccess;

	struct Watch {
		uint64_t tick        = 0;
		uint64_t upper_bound = 0;
	};

	[[nodiscard]] static bool NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
	                                               uint64_t& alignment);
	[[nodiscard]] bool        WaitPendingOperations(const std::vector<Watch>& watches,
	                                                std::optional<size_t>     invalidation_mark,
	                                                uint64_t requested_upper_bound, bool allow_wait,
	                                                size_t& wait_cursor, uint64_t& wait_bound);

	uint64_t              m_offset      = 0;
	uint64_t              m_mapped_size = 0;
	std::vector<Watch>    m_current_watches;
	size_t                m_current_watch_cursor = 0;
	std::optional<size_t> m_invalidation_mark;
	std::vector<Watch>    m_previous_watches;
	size_t                m_wait_cursor = 0;
	uint64_t              m_wait_bound  = 0;
};

// Device-fault report: whose buffer memory a faulting GPU address falls in.
std::string DescribeBufferGpuAddress(uint64_t address, uint64_t precision);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
