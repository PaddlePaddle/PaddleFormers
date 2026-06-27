# Common Tests / 公共模块测试

Unit tests for top-level paddlefleet package initialization and metadata.
顶层 paddlefleet 包初始化与元数据的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_paddlefleet_init.py` | Tests for paddlefleet top-level package imports and exports / 测试 paddlefleet 顶层包的导入与导出 |

## Running Tests

```bash
# Run all common tests
python -m pytest fleet_tests/ai_edited_test/common/ -v
```

## Notes

- These tests verify that the top-level package is correctly structured and exports all expected symbols.
