# Logging Guide for Green Network Training

## Overview

The project now uses Python's `logging` module instead of `print()` statements for better control, formatting, and debugging.

## Quick Start

### Basic Usage in Training

```python
from utils.logger import get_training_logger

# Get logger instance
logger = get_training_logger(log_dir="logs")

# Use different log levels
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical errors")
```

### Running Training with Logs

```bash
# Run training - logs will be saved to logs/training_TIMESTAMP.log
python src/green_network/train/trainer/train.py
```

---

## Logging Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| `DEBUG` | Detailed debugging info | Variable values, loop iterations |
| `INFO` | General progress updates | "Environment created", "Episode 10/1000" |
| `WARNING` | Potential issues | "File not found, using defaults" |
| `ERROR` | Errors that don't stop execution | "Failed to load file" |
| `CRITICAL` | Critical errors | "Training cannot continue" |

---

## Log Output Examples

### Console Output
```
2025-12-04 21:30:15 | GreenNetwork.Train | INFO | ============================================================
2025-12-04 21:30:15 | GreenNetwork.Train | INFO | Starting Green Network Training
2025-12-04 21:30:15 | GreenNetwork.Train | INFO | ============================================================
2025-12-04 21:30:15 | GreenNetwork.Train | INFO | Configuration loaded: 1000 episodes, device=cuda
2025-12-04 21:30:15 | GreenNetwork.Train | WARNING | Topology file 'topo.json' not found
2025-12-04 21:30:15 | GreenNetwork.Train | INFO | Using random topology generation
2025-12-04 21:30:15 | GreenNetwork.Train | INFO | Initializing SDN environment...
2025-12-04 21:30:15 | GreenNetwork.Train | INFO |   Target: 40 nodes, 61 edges
2025-12-04 21:30:15 | GreenNetwork.Train | INFO |   Method: random_tree_plus
2025-12-04 21:30:16 | GreenNetwork.Train | INFO | ✓ Environment created successfully
2025-12-04 21:30:16 | GreenNetwork.Train | INFO |   Actual nodes: 40
2025-12-04 21:30:16 | GreenNetwork.Train | INFO |   Actual edges: 61
2025-12-04 21:30:16 | GreenNetwork.Train | INFO | Episode 0/1000 | Avg Return: -12.345
2025-12-04 21:30:18 | GreenNetwork.Train | INFO | Episode 10/1000 | Avg Return: -8.234
```

### File Output
All console logs **plus** debug-level logs are saved to:
```
logs/training_20251204_213015.log
```

---

## Customizing Logging

### Change Log Level

```python
from utils.logger import setup_logger
import logging

# More verbose logging (includes DEBUG)
logger = setup_logger("MyLogger", level=logging.DEBUG)

# Less verbose logging (WARNING and above only)
logger = setup_logger("MyLogger", level=logging.WARNING)
```

### Disable Console Output

```python
# Only log to file, no console output
logger = setup_logger(
    "MyLogger",
    log_file="my_training.log",
    console=False
)
```

### Custom Log File

```python
from utils.logger import get_training_logger

# Specify custom log directory
logger = get_training_logger(log_dir="my_logs")

# This will create: my_logs/training_TIMESTAMP.log
```

---

## Using Logging in Different Modules

### In Environment Code

```python
# envs/env.py
from utils.logger import get_env_logger

class SDNEnv:
    def __init__(self, ...):
        self.logger = get_env_logger()
        self.logger.info("SDN Environment initialized")
    
    def reset(self):
        self.logger.debug("Resetting environment")
        # ... reset logic ...
        return obs
```

### In Agent Code

```python
# agents/agent.py
from utils.logger import get_agent_logger

class Agent:
    def __init__(self, ...):
        self.logger = get_agent_logger()
        self.logger.info("Agent initialized")
    
    def update(self):
        self.logger.debug("Updating policy")
        # ... update logic ...
```

### In Custom Modules

```python
from utils.logger import get_logger

# Get a logger for your module
logger = get_logger("GreenNetwork.MyModule")

logger.info("Starting custom process")
logger.warning("This might take a while")
```

---

## Advanced Features

### Logging with Exception Traceback

```python
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    # exc_info=True includes full stack trace
```

### Progress Logging

```python
# Log every N episodes
for ep in range(num_episodes):
    # ... training ...
    
    if ep % 10 == 0:
        logger.info(f"Progress: {ep}/{num_episodes} ({ep/num_episodes*100:.1f}%)")
```

### Structured Logging

```python
# Log structured data
logger.info(
    f"Episode {ep} | "
    f"Return: {avg_return:.3f} | "
    f"Loss: {loss:.4f} | "
    f"Time: {elapsed:.2f}s"
)
```

---

## File Locations

### Log Files Structure
```
Research-GreenNetwork/
├── logs/                                # Default log directory
│   ├── training_20251204_210000.log   # Training session 1
│   ├── training_20251204_213015.log   # Training session 2
│   └── training_20251204_220530.log   # Training session 3
└── src/green_network/train/
    └── utils/
        └── logger.py                   # Logging utilities
```

### Log File Naming
- Format: `training_YYYYMMDD_HHMMSS.log`
- Example: `training_20251204_213015.log`
- New file created for each training session

---

## Comparison: Print vs Logging

### Before (Print)
```python
print("Starting training...")
print(f"Episode {ep} avg_return={avg_return:.3f}")
```

**Problems:**
- ❌ No timestamps
- ❌ Can't filter by importance
- ❌ No file output
- ❌ Hard to debug
- ❌ Mixed with other output

### After (Logging)
```python
logger.info("Starting training...")
logger.info(f"Episode {ep} | Avg Return: {avg_return:.3f}")
```

**Benefits:**
- ✅ Automatic timestamps
- ✅ Filterable by level (DEBUG, INFO, WARNING, ERROR)
- ✅ Saves to file automatically
- ✅ Easy to debug with different verbosity levels
- ✅ Professional format
- ✅ Can be disabled/enabled per module

---

## Configuration Examples

### Minimal Logging (Production)
```python
logger = setup_logger("GreenNetwork", level=logging.WARNING)
# Only shows warnings and errors
```

### Verbose Logging (Development)
```python
logger = setup_logger("GreenNetwork", level=logging.DEBUG)
# Shows everything including debug messages
```

### Multiple Loggers
```python
# Different loggers for different components
train_logger = get_training_logger()
env_logger = get_env_logger()
agent_logger = get_agent_logger()

train_logger.info("Training started")
env_logger.debug("Environment step")
agent_logger.info("Agent updated")
```

---

## Troubleshooting

### Issue: No log file created
**Solution:** Check that the `log_dir` exists or will be created:
```python
from pathlib import Path
Path("logs").mkdir(parents=True, exist_ok=True)
```

### Issue: Too much output
**Solution:** Increase the logging level:
```python
logger.setLevel(logging.WARNING)  # Only warnings and errors
```

### Issue: Not enough detail
**Solution:** Decrease the logging level:
```python
logger.setLevel(logging.DEBUG)  # All messages including debug
```

### Issue: Duplicate log messages
**Solution:** Clear handlers before re-initializing:
```python
logger.handlers.clear()
```

---

## Best Practices

### 1. Use Appropriate Log Levels
```python
logger.debug("Variable x = 123")           # Debugging details
logger.info("Training epoch 5/100")        # Normal progress
logger.warning("Using default value")      # Potential issues
logger.error("Failed to load model")       # Errors
logger.critical("Out of memory")           # Critical failures
```

### 2. Include Context in Messages
```python
# ❌ Bad
logger.info("Failed")

# ✅ Good
logger.error(f"Failed to load model from {model_path}: {error}")
```

### 3. Use f-strings for Formatting
```python
# ✅ Readable and efficient
logger.info(f"Episode {ep}/{total} | Return: {ret:.3f}")
```

### 4. Log Exceptions Properly
```python
try:
    load_model()
except Exception as e:
    logger.error(f"Model loading failed: {e}", exc_info=True)
```

---

## Summary

✅ **Replaced all `print()` with logging**  
✅ **Automatic timestamps and formatting**  
✅ **Logs saved to `logs/training_TIMESTAMP.log`**  
✅ **Different severity levels (DEBUG, INFO, WARNING, ERROR)**  
✅ **Easy to customize and filter**  
✅ **Professional and maintainable**  

**Next steps:**
1. Run training to see new log format
2. Check `logs/` directory for saved logs
3. Adjust log level if needed (`logging.DEBUG` for more detail)
4. Use logging in your custom modules!
