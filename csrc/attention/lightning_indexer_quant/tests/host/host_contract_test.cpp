#include "../../op_host/lightning_indexer_quant_host_contract.h"
#include "../../op_kernel/lightning_indexer_quant_resident_layout.h"
#include <cassert>
#include <limits>
int main()
{
    using namespace LIQHost;
    assert(ClassifyMode(false, false, 0) == Mode::Legacy);
    assert(ClassifyMode(true, true, 0) == Mode::Standalone);
    assert(ClassifyMode(false, true, 11) == Mode::Resident);
    assert(ClassifyMode(false, true, 0) == Mode::Invalid);
    assert(ClassifyMode(true, true, 11) == Mode::Invalid);
    for (unsigned count = 1; count < 11; ++count) {
        assert(ClassifyMode(false, true, count) == Mode::Invalid);
    }
    assert(FitsIndex(1) && FitsIndex(INT32_MAX));
    assert(!FitsIndex(0) && !FitsIndex(-1) && !FitsIndex(int64_t(INT32_MAX) + 1));
    size_t workspace = 0;
    assert(WorkspaceBytes(Mode::Resident, 7, 100, 0, 4, 512, workspace));
    assert(workspace == 36864);
    assert(WorkspaceBytes(Mode::Resident, 1, size_t(1) << 29, 0, 0, 0, workspace));
    assert(workspace == 8589938688ULL);
    for (size_t capacity : {size_t(2053), size_t(4099), size_t(1) << 30}) {
        for (size_t budget : {size_t(0), size_t(128), size_t(2048)}) {
            assert(WorkspaceBytes(Mode::Resident, 2, capacity, 0, budget, 0, workspace));
            const size_t rowBytes = static_cast<size_t>(LIQ950::ResidentLayout::RowElements(capacity, budget)) *
                                    sizeof(int32_t);
            assert(workspace == 2 * rowBytes + 512);
        }
    }
    assert(LIQ950::ResidentLayout::RowElements(1U << 30, 0) > 4ULL * (1ULL << 30));
    assert(WorkspaceBytes(Mode::Standalone, 7, 0, 33, 3, 0, workspace));
    assert(workspace == 7 * 512 + 512);
    assert(!WorkspaceBytes(Mode::Resident, std::numeric_limits<size_t>::max(), 100, 0, 0, 0, workspace));
    assert(!WorkspaceBytes(Mode::Resident, 1, 1, 0, 0, std::numeric_limits<size_t>::max(), workspace));
    assert(!WorkspaceBytes(Mode::Resident, 1, std::numeric_limits<size_t>::max(), 0, 0, 0, workspace));
    assert(!WorkspaceBytes(Mode::Resident, 1, 1, 0, std::numeric_limits<size_t>::max(), 0, workspace));
    assert(TensorSpan(8, 128, 4096));
    assert(!TensorSpan(INT32_MAX, INT32_MAX, 4096));
    assert(UbBytes(2048, 2048, 4096, false) <= 192 * 1024);
    assert(UbBytes(2048, 2048, 4096, false) == 192288);
    assert(UbBytes(2048, 2048, 4096, true) == 194368);
    assert(UbBytes(2048, 2048, 4096, true) <= 192 * 1024);
}
