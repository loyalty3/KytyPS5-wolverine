#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"

#include <algorithm>
#include <atomic>
#include <span>
#include <utility>
#include <cinttypes>
#include <cstring>
#include <numeric>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

namespace {

constexpr size_t WATCHES_INITIAL_RESERVE = 0x4000;
constexpr size_t WATCHES_RESERVE_CHUNK   = 0x1000;

[[nodiscard]] VmaAllocationCreateFlags AllocationFlags(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::Upload:
		case MemoryUsage::Stream:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT |
			       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		case MemoryUsage::Download:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		case MemoryUsage::DeviceLocal: return {};
	}
	return {};
}

[[nodiscard]] VmaMemoryUsage AllocationUsage(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::DeviceLocal:
		case MemoryUsage::Stream: return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		case MemoryUsage::Upload:
		case MemoryUsage::Download: return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

[[nodiscard]] bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result) {
	if (alignment == 0) {
		result = value;
		return true;
	}
	const auto aligned = Common::AlignUp(value, alignment);
	if (aligned < value) {
		return false;
	}
	result = aligned;
	return true;
}

} // namespace

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(usage), m_cpu_address(cpu_address),
      m_size(size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(graphics.allocator == nullptr || size == 0);

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = flags;

	const bool with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
	const VmaAllocationCreateFlags bda_flag =
	    with_bda ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags =
	    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag | AllocationFlags(usage);
	allocation_info.usage = AllocationUsage(usage);
	allocation_info.preferredFlags = usage == MemoryUsage::DeviceLocal
	                                     ? VkMemoryPropertyFlags {}
	                                     : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VmaAllocationInfo allocation_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	const auto        create        = [&](const VmaAllocationCreateInfo& create_info) {
        return static_cast<vk::Result>(vmaCreateBuffer(
            graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info), &create_info,
            &native_buffer, &m_allocation, &allocation_result));
	};
	auto result = create(allocation_info);
	if (result != vk::Result::eSuccess) {
		// The heap it prefers is out of budget: video memory, on a 4-8 GB card, for the game's
		// large buffers. As images do (GraphicContext::CreateImage), take system memory and pay in
		// bandwidth rather than stop; then the preferred heap past its budget, which the driver
		// may still serve by paging. Only when both fail is the session out of memory.
		auto spill           = allocation_info;
		spill.usage          = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		spill.preferredFlags = 0;
		const auto first     = result;
		result               = create(spill);
		const char* where    = "system memory";
		if (result != vk::Result::eSuccess) {
			auto over  = allocation_info;
			over.flags &= ~static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
			result     = create(over);
			where      = "the preferred heap past its budget";
		}
		static std::atomic<uint32_t> spills {0};
		if (spills.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("Buffer spilled (%s): %" PRIu64 " MB, usage %d, first attempt %s -> %s\n", where,
			     size >> 20u, static_cast<int>(usage), vk::to_string(first).c_str(),
			     vk::to_string(result).c_str());
		}
	}
	if (result != vk::Result::eSuccess) {
		graphics.LogMemoryBudget();
		EXIT("Out of memory: a %" PRIu64 " MB GPU buffer fits neither in video memory nor in system "
		     "memory (%s). The VMA heap lines above give usage and budget. Close other programs; on "
		     "a card with little video memory lower the resolution; if Windows' commit limit is "
		     "reached, enlarge the page file.\n",
		     size >> 20u, vk::to_string(result).c_str());
	}

	m_buffer = native_buffer;
	if (with_bda) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
	}

	VkMemoryPropertyFlags properties = 0;
	vmaGetAllocationMemoryProperties(graphics.allocator, m_allocation, &properties);
	m_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	if (allocation_result.pMappedData != nullptr) {
		m_mapped = {static_cast<uint8_t*>(allocation_result.pMappedData),
		            static_cast<size_t>(size)};
	}
}

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, SparseTag,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size, uint64_t chunk_min)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(MemoryUsage::DeviceLocal),
      m_cpu_address(cpu_address), m_size(size), m_sparse(true) {
	EXIT_IF(!graphics.sparse_buffer_enabled || size == 0);
	vk::BufferCreateInfo buffer_info {};
	buffer_info.flags = vk::BufferCreateFlagBits::eSparseBinding |
	                    vk::BufferCreateFlagBits::eSparseResidency;
	buffer_info.size        = size;
	buffer_info.usage       = flags;
	buffer_info.sharingMode = vk::SharingMode::eExclusive;
	RequireVulkanSuccess(graphics.device.createBuffer(&buffer_info, nullptr, &m_buffer),
	                     "create sparse buffer");
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(m_buffer, &requirements);
	EXIT_IF(requirements.alignment == 0 || requirements.memoryTypeBits == 0);
	m_chunk_size       = Common::AlignUp(std::max<uint64_t>(chunk_min, requirements.alignment),
	                                     requirements.alignment);
	m_memory_type_bits = requirements.memoryTypeBits;
	m_chunks.assign(static_cast<size_t>((size + m_chunk_size - 1) / m_chunk_size), 0);
	if (static_cast<bool>(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress)) {
		// The address of the whole range, bound or not: unbound pages read as zero.
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
	}
}

Buffer::~Buffer() {
	if (m_sparse) {
		// Unbinding is implicit: the buffer goes first, then the memory it was bound to.
		if (m_buffer != nullptr) {
			m_graphics->device.destroyBuffer(m_buffer, nullptr);
		}
		if (m_pool != nullptr) {
			m_graphics->device.destroyCommandPool(m_pool, nullptr);
		}
		for (const auto& slab: m_slabs) {
			vmaFreeMemory(m_graphics->allocator, slab.allocation);
		}
		return;
	}
	if (m_buffer != nullptr) {
		vmaDestroyBuffer(m_graphics->allocator, m_buffer, m_allocation);
	}
}

uint64_t Buffer::ResidentBytes() const noexcept {
	return m_sparse ? m_resident_bytes : m_size;
}

void Buffer::MakeResident(uint64_t offset, uint64_t size) {
	const std::pair<uint64_t, uint64_t> range {offset, size};
	MakeResidentRanges({&range, 1});
}

void Buffer::MakeResidentRanges(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	if (!m_sparse) {
		return;
	}
	std::vector<uint64_t> missing;
	for (const auto& [offset, size]: ranges) {
		if (size == 0) {
			continue;
		}
		EXIT_IF(offset > m_size || size > m_size - offset);
		const auto first = offset / m_chunk_size;
		const auto last  = (offset + size - 1) / m_chunk_size;
		for (auto chunk = first; chunk <= last; ++chunk) {
			if (m_chunks[chunk] == 0) {
				missing.push_back(chunk);
			}
		}
	}
	if (missing.empty()) {
		return;
	}
	std::sort(missing.begin(), missing.end());
	missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
	KYTY_PROFILER_FUNCTION();
	auto& graphics = *m_graphics;
	// Newly bound memory holds garbage until it is zeroed, and submitted work may read these
	// pages (as unmapped, expecting zero): let it finish first. The command buffer being
	// recorded is left alone; it is submitted after the bind and the fill below.
	if (const auto tick = Scheduler().CurrentTick(); tick > 1) {
		Scheduler().Wait(tick - 1);
	}

	std::vector<vk::SparseMemoryBind> binds;
	binds.reserve(missing.size());
	for (const auto chunk: missing) {
		const auto start  = chunk * m_chunk_size;
		const auto length = std::min(m_chunk_size, m_size - start);
		if (m_slabs.empty() || m_slabs.back().used + m_chunk_size > m_slabs.back().size) {
			constexpr uint64_t SlabBytes = 2ull << 20u;
			const auto         slab_size = Common::AlignUp(std::max(SlabBytes, m_chunk_size), m_chunk_size);
			VkMemoryRequirements requirements {};
			requirements.size           = slab_size;
			requirements.alignment      = m_chunk_size;
			requirements.memoryTypeBits = m_memory_type_bits;
			VmaAllocationCreateInfo create {};
			create.flags         = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
			create.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			VmaAllocation     allocation = nullptr;
			VmaAllocationInfo info {};
			auto result =
			    vmaAllocateMemory(graphics.allocator, &requirements, &create, &allocation, &info);
			if (result != VK_SUCCESS) {
				// Over budget: the card past its budget (the driver may page), as dense buffers do.
				create.flags = 0;
				result = vmaAllocateMemory(graphics.allocator, &requirements, &create, &allocation, &info);
			}
			if (result != VK_SUCCESS) {
				// Then any memory type the buffer takes.
				create.requiredFlags  = 0;
				create.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
				result = vmaAllocateMemory(graphics.allocator, &requirements, &create, &allocation, &info);
			}
			if (result != VK_SUCCESS) {
				graphics.LogMemoryBudget();
				EXIT("Out of memory: %" PRIu64 " KB of video memory for a sparse buffer (%s).\n",
				     slab_size >> 10u, vk::to_string(static_cast<vk::Result>(result)).c_str());
			}
			m_slabs.push_back({allocation, info.deviceMemory, info.offset, slab_size, 0});
		}
		auto& slab = m_slabs.back();
		vk::SparseMemoryBind bind {};
		bind.resourceOffset = start;
		bind.size           = length;
		bind.memory         = slab.memory;
		bind.memoryOffset   = slab.offset + slab.used;
		slab.used += m_chunk_size;
		binds.push_back(bind);
		m_chunks[chunk] = 1;
		m_resident_bytes += length;
	}

	vk::SparseBufferMemoryBindInfo buffer_bind {};
	buffer_bind.buffer    = m_buffer;
	buffer_bind.bindCount = static_cast<uint32_t>(binds.size());
	buffer_bind.pBinds    = binds.data();
	vk::BindSparseInfo bind_info {};
	bind_info.bufferBindCount = 1;
	bind_info.pBufferBinds    = &buffer_bind;
	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence {};
	RequireVulkanSuccess(graphics.device.createFence(&fence_info, nullptr, &fence),
	                     "create sparse bind fence");
	{
		Common::LockGuard lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.bindSparse(1, &bind_info, fence), "vkQueueBindSparse");
	}
	RequireVulkanSuccess(graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX),
	                     "wait for sparse bind");
	RequireVulkanSuccess(graphics.device.resetFences(1, &fence), "reset sparse bind fence");

	// Newly bound memory is undefined; unmapped entries must read zero, as unbound ones did.
	// Zeroed in a command buffer of its own, done before the scheduler's next submission.
	if (m_pool == nullptr) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient |
		                             vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = graphics.queue_family;
		RequireVulkanSuccess(graphics.device.createCommandPool(&pool_info, nullptr, &m_pool),
		                     "create sparse fill pool");
	}
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::CommandBuffer command {};
	RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&allocate, &command),
	                     "allocate sparse fill command buffer");
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin), "begin sparse fill");
	for (size_t i = 0; i < binds.size();) {
		auto begin = binds[i].resourceOffset;
		auto end   = begin + binds[i].size;
		for (i++; i < binds.size() && binds[i].resourceOffset == end; i++) {
			end += binds[i].size;
		}
		command.fillBuffer(m_buffer, begin, end - begin, 0);
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 1, &barrier, 0, nullptr, 0,
	                        nullptr);
	RequireVulkanSuccess(command.end(), "end sparse fill");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.submit(1, &submit, fence), "submit sparse fill");
	}
	RequireVulkanSuccess(graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX),
	                     "wait for sparse fill");
	graphics.device.freeCommandBuffers(m_pool, 1, &command);
	graphics.device.destroyFence(fence, nullptr);
}

vk::DeviceAddress Buffer::BufferDeviceAddress() const noexcept {
	EXIT_IF(m_device_address == 0);
	return m_device_address;
}

bool Buffer::IsInBounds(uint64_t address, uint64_t size) const noexcept {
	return address >= m_cpu_address && size <= Size() && address - m_cpu_address <= Size() - size;
}

void Buffer::Flush(uint64_t offset, uint64_t size) {
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaFlushAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

void Buffer::Invalidate(uint64_t offset, uint64_t size) {
	EXIT_IF(m_usage != MemoryUsage::Download || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaInvalidateAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

vk::BufferMemoryBarrier Buffer::Barrier(uint64_t offset, uint64_t size, vk::AccessFlags source,
                                        vk::AccessFlags destination) const {
	if (Handle() == nullptr || size == 0 || offset > Size() || size > Size() - offset) {
		EXIT("Buffer: invalid DMA barrier, handle=%p offset=0x%016" PRIx64 " size=0x%016" PRIx64
		     " capacity=0x%016" PRIx64 "\n",
		     static_cast<const void*>(Handle()), offset, size, Size());
	}
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = source;
	barrier.dstAccessMask       = destination;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = Handle();
	barrier.offset              = offset;
	barrier.size                = size;
	return barrier;
}

void Buffer::CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
                      uint64_t destination_offset, uint64_t size, vk::AccessFlags source_before,
                      vk::AccessFlags destination_before, vk::AccessFlags source_after,
                      vk::AccessFlags destination_after) {
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	command.EndRendering();
	const vk::BufferMemoryBarrier before[] = {
	    source.Barrier(source_offset, size, source_before, vk::AccessFlagBits::eTransferRead),
	    Barrier(destination_offset, size, destination_before, vk::AccessFlagBits::eTransferWrite),
	};
	const auto host_access  = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite;
	auto       before_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_before | destination_before) & host_access)) {
		before_stage |= vk::PipelineStageFlagBits::eHost;
	}
	const auto native = command.Recorder();
	native.pipelineBarrier(before_stage, vk::PipelineStageFlagBits::eTransfer,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, before, 0, nullptr);
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	native.copyBuffer(source.Handle(), Handle(), 1, &copy);
	const vk::BufferMemoryBarrier after[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eTransferRead, source_after),
	    Barrier(destination_offset, size, vk::AccessFlagBits::eTransferWrite, destination_after),
	};
	auto after_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_after | destination_after) & host_access)) {
		after_stage |= vk::PipelineStageFlagBits::eHost;
	}
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, after_stage,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, after, 0, nullptr);
}

void Buffer::Fill(uint64_t offset, uint64_t size, uint32_t value) {
	if (((offset | size) & 3u) != 0) {
		EXIT("Buffer: fill range must be dword aligned\n");
	}
	auto& command = Scheduler().Current();
	command.EndRendering();
	const auto before =
	    Barrier(offset, size, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	            vk::AccessFlagBits::eTransferWrite);
	const auto native = command.Recorder();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.fillBuffer(Handle(), offset, size, value);
	const auto after = Barrier(offset, size, vk::AccessFlagBits::eTransferWrite,
	                           vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

StreamBuffer::StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                           uint64_t size)
    : Buffer(graphics, scheduler, usage, 0, AllFlags, size),
      m_current_watches(WATCHES_INITIAL_RESERVE), m_previous_watches(WATCHES_INITIAL_RESERVE) {}

bool StreamBuffer::NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
                                        uint64_t& alignment) {
	if (coherent) {
		return true;
	}
	if (!AlignUp(size, atom, size)) {
		return false;
	}
	const auto divisor = std::gcd(alignment, atom);
	if (alignment != 0 && alignment / divisor > UINT64_MAX / atom) {
		return false;
	}
	alignment = alignment == 0 ? atom : alignment / divisor * atom;
	return true;
}

std::pair<uint8_t*, uint64_t> StreamBuffer::Map(uint64_t size, uint64_t alignment,
                                                bool allow_wait) {
	if (Mapped().empty()) {
		return {nullptr, 0};
	}
	uint64_t   mapped_size = size;
	const auto atom        = Graphics().physical_device_properties.limits.nonCoherentAtomSize;
	if (!NormalizeReservation(IsCoherent(), atom, mapped_size, alignment)) {
		return {nullptr, 0};
	}
	if (mapped_size > Size()) {
		return {nullptr, 0};
	}

	uint64_t aligned_offset = 0;
	if (!AlignUp(m_offset, alignment, aligned_offset)) {
		return {nullptr, 0};
	}

	const bool wrap = aligned_offset > Size() - mapped_size;
	if (wrap) {
		aligned_offset = 0;
	}

	auto wait_cursor = wrap ? size_t {0} : m_wait_cursor;
	auto wait_bound  = wrap ? uint64_t {0} : m_wait_bound;
	auto invalidation_mark =
	    wrap ? std::optional<size_t> {m_current_watch_cursor} : m_invalidation_mark;
	auto& pending_watches = wrap ? m_current_watches : m_previous_watches;
	if (!WaitPendingOperations(pending_watches, invalidation_mark, aligned_offset + mapped_size,
	                           allow_wait, wait_cursor, wait_bound)) {
		return {nullptr, 0};
	}

	if (wrap) {
		m_invalidation_mark    = invalidation_mark;
		m_current_watch_cursor = 0;
		std::swap(m_previous_watches, m_current_watches);
	}
	m_wait_cursor = wait_cursor;
	m_wait_bound  = wait_bound;
	m_offset      = aligned_offset;
	m_mapped_size = mapped_size;
	return {Mapped().data() + m_offset, m_offset};
}

void StreamBuffer::Commit() {
	if (Usage() != MemoryUsage::Download && m_mapped_size != 0) {
		Flush(m_offset, m_mapped_size);
	}

	m_offset += m_mapped_size;
	const auto tick = Scheduler().CurrentTick();
	if (m_current_watch_cursor != 0 && m_current_watches[m_current_watch_cursor - 1].tick == tick) {
		m_current_watches[m_current_watch_cursor - 1].upper_bound = m_offset;
		return;
	}
	if (m_current_watch_cursor + 1 >= m_current_watches.size()) {
		m_current_watches.resize(m_current_watches.size() + WATCHES_RESERVE_CHUNK);
	}
	auto& watch       = m_current_watches[m_current_watch_cursor++];
	watch.upper_bound = m_offset;
	watch.tick        = tick;
}

uint64_t StreamBuffer::Copy(const void* source, uint64_t size, uint64_t alignment) {
	EXIT_IF(source == nullptr);
	const auto [data, offset] = Map(size, alignment);
	EXIT_IF(data == nullptr);
	std::memcpy(data, source, static_cast<size_t>(size));
	Commit();
	return offset;
}

bool StreamBuffer::WaitPendingOperations(const std::vector<Watch>& watches,
                                         std::optional<size_t>     invalidation_mark,
                                         uint64_t requested_upper_bound, bool allow_wait,
                                         size_t& wait_cursor, uint64_t& wait_bound) {
	if (!invalidation_mark.has_value()) {
		return true;
	}
	while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
		const auto& watch = watches[wait_cursor];
		if (!Scheduler().IsFree(watch.tick) && !allow_wait) {
			return false;
		}
		// Profiling-only zone: a stream-buffer ring wrap forcing the CPU to wait for the GPU to
		// finish with the memory it's about to reclaim. Scheduler().IsFree() above is the fast,
		// non-blocking path -- only reaching Wait() itself means the GPU genuinely hasn't caught
		// up yet.
		{
			KYTY_PROFILER_BLOCK("StreamBuffer::WaitPendingOperations (ring wrap wait)");
			Scheduler().Wait(watch.tick);
			if (Usage() == MemoryUsage::Download) {
				Scheduler().WaitPriorityOperations(watch.tick);
			}
		}
		wait_bound = watch.upper_bound;
		++wait_cursor;
	}
	return true;
}

} // namespace Libs::Graphics
