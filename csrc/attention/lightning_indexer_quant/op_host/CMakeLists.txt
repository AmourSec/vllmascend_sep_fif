add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    # The Ascend 950 implementation reuses both the QLI-V2 vector/cube helpers
    # and the LightningIndexerV2 common headers.
    if (ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
        set(lightning_indexer_quant_depends
            attention/quant_lightning_indexer_v2
            attention/lightning_indexer_v2
            CACHE INTERNAL "Dependencies for lightning_indexer_quant"
        )
    endif()
    target_sources(op_host_aclnn PRIVATE
        lightning_indexer_quant_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME LightningIndexerQuant
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE lightning_indexer_quant ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
