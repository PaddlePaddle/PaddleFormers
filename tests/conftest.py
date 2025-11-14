# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# conftest.py
import os
import logging
import pytest
import sys
from datetime import datetime

def pytest_configure(config):
    """Configure pytest and setup logging for xdist workers"""
    os.environ.setdefault("DOWNLOAD_SOURCE", "aistudio")
    os.environ.setdefault("COVERAGE_SOURCE", "paddleformers")
    os.environ.setdefault("WAIT_UNTIL_DONE", "True")
    # Get worker ID for xdist parallel execution
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    
    # Create logs directory
    logs_dir = "unittest_logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Clear existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        f'%(asctime)s.%(msecs)03d | {worker_id:6} | %(levelname)-8s | %(test_case)-40s | %(name)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        f'%(asctime)s | {worker_id} | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    error_formatter = logging.Formatter(
        f'%(asctime)s | {worker_id} | %(levelname)-8s | %(test_case)-40s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - for real-time terminal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.DEBUG)
    
    # Global log file - all workers write to this file
    global_log_file = f"{logs_dir}/test_global.log"
    global_file_handler = logging.FileHandler(global_log_file, mode='a', encoding='utf-8')
    global_file_handler.setFormatter(detailed_formatter)
    global_file_handler.setLevel(logging.DEBUG)
    
    # Error log file - all ERROR and above logs from all workers
    error_log_file = f"{logs_dir}/test_errors.log"
    error_file_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
    error_file_handler.setFormatter(error_formatter)
    error_file_handler.setLevel(logging.ERROR)  # Only ERROR and CRITICAL
    
    # Worker-specific log file (only for worker processes)
    if worker_id != 'master':
        worker_log_file = f"{logs_dir}/test_worker_{worker_id}.log"
        worker_file_handler = logging.FileHandler(worker_log_file, mode='w', encoding='utf-8')
        worker_file_handler.setFormatter(detailed_formatter)
        worker_file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(worker_file_handler)
        print(f"Worker {worker_id} logging to {worker_log_file}")
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(global_file_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.setLevel(logging.DEBUG)
    
    # Force debug level for common loggers
    logging.getLogger('').setLevel(logging.DEBUG)
    for logger_name in ['tests', 'paddleformers', 'transformers', '__main__']:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    print(f"Global log: {global_log_file}")
    print(f"Error log: {error_log_file}")

def pytest_sessionstart(session):
    """Called after the Session object has been created and before performing collection"""
    # Ensure our logging configuration takes precedence over pytest's default
    session.config.option.log_cli = False

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Called before each test setup - add test case name to logger"""
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    test_name = item.name
    class_name = item.cls.__name__ if item.cls else 'Module'
    full_test_name = f"{class_name}.{test_name}"
    
    # Log test start with timestamp
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    logging.info(f"🚀 TEST START | {full_test_name}")
    
    # Store test info for use in fixtures
    item.full_test_name = full_test_name
    item.test_start_time = start_time

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Called after each test teardown - log test completion"""
    if hasattr(item, 'full_test_name'):
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        start_time = getattr(item, 'test_start_time', 'Unknown')
        
        # Calculate duration if start time is available
        try:
            start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
            end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f')
            duration = (end_dt - start_dt).total_seconds()
            duration_str = f"{duration:.3f}s"
        except:
            duration_str = "Unknown"
        
        logging.info(f"✅ TEST END | {item.full_test_name} | Duration: {duration_str}")

@pytest.fixture(autouse=True)
def test_case_logging(request):
    """Automatically add test case name to all log messages during test execution"""
    test_name = request.node.name
    class_name = request.node.cls.__name__ if request.node.cls else 'Module'
    full_test_name = f"{class_name}.{test_name}"
    
    # Create a custom filter to add test case name to log records
    class TestCaseFilter(logging.Filter):
        def __init__(self, test_case):
            super().__init__()
            self.test_case = test_case
            
        def filter(self, record):
            record.test_case = self.test_case
            return True
    
    # Add filter to all handlers temporarily for this test
    test_filter = TestCaseFilter(full_test_name)
    root_logger = logging.getLogger()
    
    for handler in root_logger.handlers:
        handler.addFilter(test_filter)
    
    yield
    
    # Remove the filter after test completion
    for handler in root_logger.handlers:
        handler.removeFilter(test_filter)

def pytest_collection_modifyitems(config, items):
    """Modify collected test items to ensure proper logging in deep directories"""
    for item in items:
        # Ensure test items have proper attributes for logging
        if not hasattr(item, 'full_test_name'):
            # Handle deep directory structures by using nodeid
            nodeid_parts = item.nodeid.split('::')
            if len(nodeid_parts) > 1:
                test_path = nodeid_parts[0]  # File path
                test_name = nodeid_parts[-1]  # Test name
                class_name = nodeid_parts[1] if len(nodeid_parts) > 2 else 'Module'
                item.full_test_name = f"{class_name}.{test_name}"
            else:
                item.full_test_name = item.nodeid

# Ensure real-time output flushing
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Ensure real-time output for test execution"""
    # Force stdout flushing for real-time output
    old_stdout = sys.stdout
    sys.stdout = type('FlushStdout', (), {
        'write': lambda self, x: (old_stdout.write(x), old_stdout.flush()),
        'flush': old_stdout.flush
    })()
    
    try:
        yield
    finally:
        sys.stdout = old_stdout

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test results and log failures to error log"""
    outcome = yield
    report = outcome.get_result()
    
    if report.failed:
        # Log test failure to error log
        error_msg = f"Test failed: {getattr(item, 'full_test_name', item.nodeid)} - {report.longreprtext}"
        logging.error(error_msg)

def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished"""
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    if worker_id == 'master':
        logging.info("All tests completed. Check logs in 'logs/' directory")
        print(f"\n📊 Log files created:")
        print(f"   - Global log: logs/test_global.log")
        print(f"   - Error log: logs/test_errors.log")
        if worker_id != 'master':
            print(f"   - Worker log: logs/test_worker_{worker_id}.log")