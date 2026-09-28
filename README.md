# PCAP: Persona-based Conversational Adversarial Prompting

PCAP is a sophisticated red-teaming framework for testing language model safety through multi-turn, persona-based adversarial conversations. It uses multiple AI personas with distinct characteristics and strategies to systematically probe for vulnerabilities in target language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running Attacks](#running-attacks)
- [Analyzing Results](#analyzing-results)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

PCAP (Persona-based Conversational Adversarial Prompting) implements a tree-based attack strategy where:

- **Multiple personas** attack the target model
- **Each persona** has unique characteristics and manipulation strategies
- **Conversations branch** to explore different attack paths
- **Success is measured** by whether the model produces unsafe or policy-violating responses

### How It Works

1. **Persona Generation**: Creates diverse AI personas with specific traits and goals
2. **Strategy Assignment**: Assigns manipulation strategies (e.g., emotional appeal, authority) to each persona
3. **Tree Search**: Explores attack paths using branching factor and depth parameters
4. **Evaluation**: Judges responses to identify successful jailbreaks
5. **Analysis**: Aggregates results to identify vulnerabilities

## ✨ Key Features

- **Multi-persona attacks**: Test models with diverse attack styles
- **Tree-based exploration**: Systematically explore attack paths
- **Flexible model support**: Works with Ollama, Hugging Face, OpenAI, and more
- **Configurable strategies**: Customize attack approaches and parameters
- **Comprehensive analysis**: Built-in tools for result interpretation
- **Extensible framework**: Easy to add new strategies and evaluators

## 🚀 Installation

### Prerequisites

- Python 3.12.12
- GPU recommended (for local model inference)
- Ollama (optional, for local attacker models)

### Step 1: Create Virtual Environment

```bash
# Using uv (recommended)
uv venv --python 3.12.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using standard venv
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
uv pip install -r requirments_paper.txt

# Or using pip
pip install -r requirments_paper.txt
```

### Step 3: Install ARES Framework and Plugins

```bash
# Install the ARES framework
uv pip install .

# Install PCAP persona plugin
uv pip install plugins/ares-persona

# Install LiteLLM connector (for API-based models)
uv pip install plugins/ares-litellm-connector
```

### Step 4: Set Up Ollama (Optional)

If using Ollama for attacker models:

```bash
# Install Ollama from https://ollama.ai

# Pull a model (example)
ollama pull qwen3:8b
```

## 🎬 Quick Start

### 1. Run a Pre-configured Attack

```bash
# Run with Ollama models (fastest to set up)
ares evaluate example_configs/plugins/ares_persona/assets/persona_example_ollama_small.yaml

# Or run with mixed Ollama + Hugging Face
ares evaluate example_configs/plugins/ares_persona/assets/persona_example_ollama_and_huggingface_small.yaml
```

### 2. Monitor Progress

The attack will:
- Generate personas and assign strategies
- Create conversation trees
- Query the target model
- Evaluate responses for jailbreaks
- Save results to the configured output folder

### 3. Analyze Results

```bash
# Open the result analysis notebook
jupyter notebook result_reader.ipynb
```

## ⚙️ Configuration

### Using the Configuration Generator

The easiest way to create custom configurations is using the [`config_maker.ipynb`](config_maker.ipynb) notebook:

```bash
jupyter notebook config_maker.ipynb
```

**Key parameters to configure:**

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `N_CHARS` | Number of personas | 3-6 for testing, 6-10 for comprehensive |
| `DEPTH` | Max conversation turns | 5-8 for quick tests, 10-15 for thorough |
| `branching_factor` | Branches per node | 2-3 for balanced exploration |
| `STRAT_SET` | Strategies per persona | 2-3 for diverse behavior |
| `model_path` | Target model to test | Any Hugging Face model ID |
| `query_budget` | Max target-model queries per goal; `0` = unlimited | `0` for exhaustive runs; `50`–`200` to cap cost |
| `stop_at_first_jb` | Stop each goal's loop on the first successful jailbreak | `true` for efficiency; `false` to collect all jailbreaks within the budget |

### Pre-configured Examples

#### 1. Ollama-only Configuration
**File**: [`persona_example_ollama_small.yaml`](example_configs/plugins/ares_persona/assets/persona_example_ollama_small.yaml)

- **Attacker**: Ollama model (generates adversarial prompts)
- **Target**: Ollama model (being tested)
- **Evaluator**: Ollama model (judges responses)
- **Best for**: Quick local testing without GPU requirements

#### 2. Mixed Ollama + Hugging Face Configuration
**File**: [`persona_example_ollama_and_huggingface_small.yaml`](example_configs/plugins/ares_persona/assets/persona_example_ollama_and_huggingface_small.yaml)

- **Attacker**: Ollama model
- **Target**: Hugging Face model (local GPU inference)
- **Evaluator**: Hugging Face model
- **Best for**: Testing specific Hugging Face models with GPU acceleration

### Configuration File Structure

```yaml
persona-redteaming:
  strategy:
    persona-strategy:
      # Target model configuration
      target_model_config:
        connector: # Model connector settings
        local_target: false
        target_as_detector: false
      
      # Attacker model configuration
      attacker_model_config:
        connector: # Attacker model settings
      
      # Persona configuration
      persona_model_config:
        number_of_characters: 6
      
      # Tree search parameters
      tree_params:
        depth: 10
        width: 10
        branching_factor: 3
      
      # Strategy configuration
      strategy_config:
        use_fixed_strategies: false
        strategies_path: 'assets/strategies.yaml'
        n_strategies_per_character: 2
      
      # Output configuration
      prompts_folder: 'assets/output'

      # Budget / early-stop controls (optional)
      query_budget: 0          # max target-model queries per goal (0 = unlimited)
      stop_at_first_jb: true   # stop as soon as the first jailbreak is found;
                               # set to false to collect all jailbreaks within the budget
```

## 🎯 Running Attacks

### Basic Command

```bash
ares evaluate <path_to_config.yaml>
```

### With Custom Output Directory

Edit the `prompts_folder` field in your configuration file to specify where results should be saved.

### Understanding Output Structure

Results are saved in the following structure:

```
prompts_folder/
├── multi_character_N/          # N = number of personas
│   ├── 1/                      # First character
│   │   ├── goal_id_0.csv       # Results for goal 0
│   │   ├── goal_id_1.csv       # Results for goal 1
│   │   └── conv/               # Conversation logs
│   │       ├── <uuid>/
│   │       │   ├── characters.csv  # Persona definitions
│   │       │   └── intents.csv     # Conversation intents
│   ├── 2/                      # Second character
│   └── 3/                      # Third character
```

### Result Files Explained

- **`goal_id_X.csv`**: Contains attack results for each goal
  - `iteration`: Conversation turn number
  - `judge_score`: Success score (10 = successful jailbreak)
  - `adv_prompt`: The adversarial prompt used
  - `goal`: The attack goal
  - `char_queries`: Cumulative target-model queries used by this character
  - `global_queries`: Cumulative target-model queries across all characters
  - `first_jb_query`: Global query index at which this character first found a jailbreak (`null` if none)
  - `global_first_jb_query`: Global query index of the very first jailbreak across all characters (`null` if none)

- **`characters.csv`**: Persona definitions and traits
- **`intents.csv`**: Conversation intents and strategies used

## 📊 Analyzing Results

### Using the Result Reader Notebook

1. Open the notebook:
```bash
jupyter notebook result_reader.ipynb
```

2. Configure the analysis parameters:
```python
N = 3  # Number of runs to analyze
prompts_folder = 'assets/prompts_PCAP/'
```

3. Run all cells to generate:
   - Summary statistics
   - Success rates
   - Efficiency metrics
   - Successful jailbreak prompts

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `total_attacks` | Number of attack attempts | Higher = more comprehensive testing |
| `successful_goals` | Goals with ≥1 jailbreak | Higher = more vulnerabilities found |
| `avg_successful_attacks` | Mean jailbreaks per goal | Higher = easier to jailbreak |
| `avg_success_cost` | Jailbreaks per iteration | Higher = more efficient attacks |
| `avg_iterations` | Mean conversation depth | Shows how deep conversations went |
| `avg_num_queries` | Mean model queries | Resource usage indicator |


## 🔧 Advanced Usage

### Custom Attack Strategies

Edit [`assets/strategies.yaml`](assets/strategies.yaml) to add new manipulation strategies:

```yaml
strategies:
  - name: "emotional_appeal"
    description: "Use emotional manipulation"
  - name: "authority"
    description: "Claim authority or expertise"
  - name: "your_custom_strategy"
    description: "Your strategy description"
```

## 📁 Project Structure

```
PCAP/
├── assets/                          # Attack resources
│   ├── strategies.yaml              # Attack strategies
│   ├── attack_goals.json            # Attack goals
│   └── prompts_PCAP/                # Results directory
├── example_configs/                 # Example configurations
│   └── plugins/
│       └── ares_persona/
│           └── assets/              # Config templates
├── plugins/                         # ARES plugins
│   ├── ares-persona/                # PCAP persona plugin
│   └── ares-litellm-connector/      # LiteLLM connector
├── src/                             # ARES framework source
│   └── ares/
│       ├── connectors/              # Model connectors
│       ├── evals/                   # Evaluators
│       └── goals/                   # Attack goals
├── config_maker.ipynb               # Configuration generator
├── result_reader.ipynb              # Results analyzer
├── README.md                        # This file
└── requirments_paper.txt            # Dependencies
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Problem**: GPU runs out of memory during inference

**Solutions**:
- Reduce batch size in model config
- Use smaller models
- Use `dtype: "bfloat16"` or `"float16"`
- Reduce `branching_factor` or `DEPTH`

```yaml
model_config:
  dtype: "bfloat16"  # Use this instead of float32
```

#### 2. Ollama Connection Error

**Problem**: Cannot connect to Ollama server

**Solutions**:
- Ensure Ollama is running: `ollama serve`
- Check model is pulled: `ollama list`
- Verify endpoint in config: `endpoint-type: "ollama"`

#### 3. Slow Attack Generation

**Problem**: Attacks take too long to generate

**Solutions**:
- Reduce `DEPTH` (fewer conversation turns)
- Reduce `branching_factor` (fewer branches)
- Reduce `N_CHARS` (fewer personas)
- Use faster attacker models

#### 4. No Successful Jailbreaks

**Problem**: All attacks fail (judge_score < 10)

**Solutions**:
- Increase `DEPTH` for longer conversations
- Increase `N_CHARS` for more diverse personas
- Try different attack strategies
- Verify evaluator is working correctly
- Check if target model is too restrictive
- Set `stop_at_first_jb: false` and a generous `query_budget` to collect partial successes across the full run

#### 6. Attack Stops Earlier Than Expected

**Problem**: The attack loop terminates before reaching the configured `DEPTH`

**Solutions**:
- If a jailbreak was found and `stop_at_first_jb: true` (default), this is expected behaviour — set it to `false` to continue after the first hit
- If `query_budget` is set, the run stops when that many target queries have been made; increase the value or set it to `0` for unlimited queries

#### 5. Import Errors

**Problem**: Module not found errors

**Solutions**:
```bash
# Reinstall all components
uv pip install .
uv pip install plugins/ares-persona
uv pip install plugins/ares-litellm-connector

# Verify installation
python -c "import ares; print(ares.__version__)"
```
