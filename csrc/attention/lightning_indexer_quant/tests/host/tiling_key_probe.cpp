#include <iostream>
#include <set>
#include <utility>
#include <vector>

#include "lightning_indexer_quant_template_tiling_key.h"

int main()
{
#ifdef ASCENDC_DEBUG
    using Combination = std::vector<uint64_t>;
    std::set<Combination> actual;
    for (const auto &group : g_tilingSelectParams) {
        std::vector<Combination> product(1);
        for (const auto &parameter : group) {
            size_t first = 0;
            if (parameter.paramType == ASCENDC_TPL_UINT) {
                if (parameter.vals.empty() || parameter.vals.front() != ASCENDC_TPL_UI_LIST) {
                    std::cerr << "Expected an explicit UINT selection list\n";
                    return 1;
                }
                first = 1;
            }
            std::vector<Combination> next;
            for (const auto &prefix : product) {
                for (size_t index = first; index < parameter.vals.size(); ++index) {
                    Combination combination = prefix;
                    combination.push_back(parameter.vals[index]);
                    next.push_back(combination);
                }
            }
            product = std::move(next);
        }
        actual.insert(product.begin(), product.end());
    }
    const std::set<Combination> expected{
        {2, 2, 3, 1, 0, 2}, {2, 2, 3, 1, 1, 2},
        {2, 2, 3, 0, 0, 0}, {2, 2, 3, 0, 0, 1},
        {2, 2, 3, 0, 1, 0}, {2, 2, 3, 0, 1, 1},
        {36, 36, 3, 1, 1, 2},
    };
    if (actual != expected) {
        std::cerr << "INT8/FP8 template selection matrix mismatch\n";
        return 1;
    }
    std::cout << actual.size() << " supported combinations\n";
#else
    std::cout << GET_TPL_TILING_KEY(2, 2, 3, 1, 1, 2) << ' '
              << GET_TPL_TILING_KEY(36, 36, 3, 1, 1, 2) << '\n';
#endif
    return 0;
}
