def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "npu_smoke: fast fused QLI and planner hardware smoke test")
    config.addinivalue_line("markers", "npu_inline: inline UB planner scheduling and row-boundary tests")
    config.addinivalue_line("markers", "npu_stress: long-context or maximum-batch NPU stress test")
