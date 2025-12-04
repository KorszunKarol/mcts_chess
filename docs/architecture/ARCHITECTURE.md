# Chess Engine Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Contracts](#data-contracts)
4. [Data Flow](#data-flow)
5. [Module Reference](#module-reference)

---

## System Overview

This chess engine implements a **Cognitive Asymmetry** optimization system called "Coach Tal" that selects moves to maximize the gap between user ease and opponent confusion, subject to soundness constraints.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Points (bin/)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ UCI Engines  │  │ Runner Scripts│  │ Shell Wrappers│    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Coach Tal Selector (Core Logic)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Get MCTS candidates (or raw policy)              │  │
│  │ 2. Evaluate cognitive metrics for each candidate    │  │
│  │ 3. Re-rank using J(s') = λ·V + γ·(H_opp - E_user)   │  │
│  │ 4. Apply soundness constraint                        │  │
│  │ 5. Return selected move + analysis                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   MCTS       │  │  Evaluator   │  │   Agents     │
│  Controller  │  │ (Transformer)│  │ (Opp/User)   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Encoder     │  │  Neural Net  │  │   Metrics    │
│  (8x8x34)    │  │  (Hybrid)    │  │  (Entropy)   │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Lazy Loading**: Heavy models (neural networks) are loaded only when needed
3. **Perspective-Aware**: Model handles both White and Black perspectives without mirroring
4. **Extensible**: Easy to add new metrics, agents, or evaluation methods
5. **Performance**: MCTS uses shared memory IPC for high-throughput parallel search

---

## Component Architecture

### 1. Entry Points (`bin/`)

**Purpose**: UCI-compatible interfaces for chess GUIs and command-line tools.

| Script | Purpose | Output |
|--------|---------|--------|
| `coach_tal_uci.py` | UCI engine with Coach Tal re-ranking | UCI protocol |
| `coach_tal_mcts_uci.py` | UCI engine with MCTS + Coach Tal | UCI protocol |
| `uci_engine.py` | Basic UCI engine (legacy) | UCI protocol |
| `run_coach_tal.py` | Simple runner script | Move selection |
| `run_coach_tal_mcts.py` | Runner with MCTS | Move selection |

**Data Contract**: UCI protocol (text-based, stdin/stdout)

---

### 2. Coach Tal Core (`src/coach_tal/`)

#### 2.1 Selector (`selector.py`)

**Purpose**: Main move selection logic using cognitive asymmetry optimization.

**Key Classes**:
- `CoachTalConfig`: Configuration parameters
- `MoveCandidate`: Analysis of a single candidate move
- `SelectionResult`: Final selection with all candidate analyses
- `CoachTalSelector`: Main selector class

**Input**: 
- `chess.Board`: Current position
- `Dict[chess.Move, float]`: MCTS policy or raw policy logits

**Output**: 
- `SelectionResult`: Chosen move + full analysis

**Algorithm**:
```
1. Get top-k candidates from MCTS (or raw policy)
2. For each candidate move m:
   a. Evaluate resulting position: V_φ(s')
   b. Compute opponent entropy: H_opp(s')
   c. Compute user ease: E_user(s' after opponent reply)
   d. Calculate J(s') = λ·V + γ·(H_opp - E_user)
   e. Check soundness: V(s') ≥ V(s) - δ
3. Filter to sound moves
4. Select move with highest J(s')
5. If no sound moves, fall back to MCTS best
```

#### 2.2 Evaluator (`evaluator.py`)

**Purpose**: Lightweight wrapper for neural network inference.

**Key Class**: `TransformerEvaluator`

**Input**: 
- `chess.Board`: Position to evaluate

**Output**: 
- `Tuple[float, Dict[chess.Move, float]]`: (value, policy_dict)
  - `value`: Scalar in [-1, 1] from current player's perspective
  - `policy_dict`: Dict mapping legal moves to probabilities

**Features**:
- Lazy model loading (loads on first `evaluate()` call)
- Supports both TensorFlow/Keras and PyTorch backends
- Handles board encoding internally
- Masks illegal moves automatically
- Applies temperature scaling to policy

**Data Contract**:
- Input encoding: `(8, 8, 34)` numpy array (HWC format)
- Model output: `(value: float, policy_logits: np.ndarray[4672])`
- Policy normalization: Softmax over legal moves only

#### 2.3 Agents (`agents.py`)

**Purpose**: Proxy models for opponent and user perspectives.

**Key Classes**:
- `OpponentModel`: Computes opponent confusion (entropy)
- `UserModel`: Computes user ease (policy sharpness)
- `create_agent_pair()`: Factory function for matched pair

**Input**: 
- `chess.Board`: Position where it's the respective player's turn

**Output**:
- `OpponentModel.get_entropy()`: `float` (Shannon entropy in nats)
- `UserModel.get_ease()`: `float` (normalized ease in [0, 1])

**Key Insight**: Both use the same underlying `TransformerEvaluator`, but:
- Different temperature settings (opponent: 1.2, user: 1.0)
- Different positions evaluated (opponent-to-move vs user-to-move)
- Different interpretation (high entropy = bad for opponent, good for us)

#### 2.4 Metrics (`metrics.py`)

**Purpose**: Pure mathematical functions for cognitive metrics.

**Key Functions**:
- `entropy(policy) -> float`: Shannon entropy H(π)
- `user_ease(policy) -> float`: Normalized ease E_user
- `cognitive_asymmetry_score(...) -> float`: Combined J(s') score
- `passes_soundness_constraint(...) -> bool`: Soundness check

**Input Contracts**:
- `policy: Dict[chess.Move, float]`: Probabilities summing to ~1.0

**Output Contracts**:
- Entropy: Non-negative float (nats)
- User ease: Float in [0, 1]
- J score: Float (unbounded, higher is better)
- Soundness: Boolean

#### 2.5 Explainer (`explainer.py`)

**Purpose**: Generate natural language explanations for move choices.

**Key Classes**:
- `Explainer`: Main explanation generator
- `MoveAnalysis`: Structured explanation data

**Input**: 
- `SelectionResult`: Move selection result
- `chess.Board`: Current position

**Output**: 
- `MoveAnalysis`: Structured explanation with:
  - `move_san`: Move in SAN notation
  - `primary_reason`: Main explanation string
  - `supporting_reasons`: List of additional reasons
  - `metrics`: Dict of numerical metrics

---

### 3. Neural Network (`src/`)

#### 3.1 Encoder (`encoder.py`)

**Purpose**: Convert chess board to neural network input tensor.

**Key Class**: `Encoder`

**Input**: 
- `chess.Board`: Python-chess board object

**Output**: 
- `np.ndarray[8, 8, 34]`: Encoded position tensor

**Channel Mapping**:
```
Channels 0-11:   Piece positions (p, n, b, r, q, k, P, N, B, R, Q, K)
Channels 12-15:  Castling rights (white/black, kingside/queenside)
Channel 16:      Material score
Channel 17:      En passant target square
Channel 18:      Halfmove clock
Channel 19:      Fullmove number
Channel 20:      Piece mobility
Channels 21-23:  Pawn structure (doubled, isolated, passed)
Channel 24:      Center control
Channel 25:      Piece-square tables
Channels 26-29: Defended and vulnerable pieces
Channels 30-31:  Piece coordination
Channel 32:      Game phase (opening/middlegame/endgame)
Channel 33:      King safety
```

**Data Contract**:
- Format: `float32` numpy array
- Shape: `(8, 8, 34)` - Height × Width × Channels
- Orientation: Board flipped vertically (rank 0 = bottom)
- Normalization: Features normalized to reasonable ranges

#### 3.2 Transformer Models

**Purpose**: Neural network architecture for position evaluation.

**Files**:
- `transformer_model.py`: TensorFlow/Keras implementation
- `transformer_model_pytorch.py`: PyTorch implementation

**Architecture**:
```
Input:  (B, 34, 8, 8)  [PyTorch NCHW] or (B, 8, 8, 34) [Keras HWC]
  │
  ▼
CNN Stem: 34 → 128 → 128 → 256 → 256 channels
  │
  ▼
Flatten: (256, 8, 8) → (256, 64)
  │
  ▼
Transformer Body: 6 layers × (Self-Attention + FFN)
  │
  ├─► Value Head: (B, 3) → Win/Draw/Loss probabilities
  └─► Policy Head: (B, 4672) → Move logits
```

**Data Contracts**:
- **Input**: `(batch_size, channels, height, width)` or `(batch_size, height, width, channels)`
- **Value Output**: `(batch_size, 3)` → `[loss_prob, draw_prob, win_prob]`
- **Policy Output**: `(batch_size, 4672)` → Raw logits for each action
- **Action Space**: 4672 possible moves (73 move types × 64 squares)

**Training Note**: Model trained without mirroring - handles both perspectives directly.

#### 3.3 Move Mapping (`move_mapping.py`)

**Purpose**: Bidirectional mapping between chess moves and action indices.

**Key Functions**:
- `move_to_index(move, board) -> Optional[int]`: Convert move to index [0, 4671]
- `index_to_move(index) -> chess.Move`: Convert index to move

**Action Space Structure**:
```
For each square (64 squares):
  - 56 queen-like moves (8 directions × 7 distances)
  - 8 knight moves
  - 9 underpromotions (3 pieces × 3 directions)
  Total: 73 move types per square × 64 squares = 4672
```

**Data Contract**:
- Index range: `[0, 4671]`
- Invalid moves: Return `None` from `move_to_index()`
- Move representation: Uses `chess.Move` objects

---

### 4. MCTS (`src/mcts/`)

#### 4.1 Controller (`controller.py`)

**Purpose**: High-performance parallel MCTS search using shared memory IPC.

**Key Classes**:
- `MCTSController`: Main controller
- `MCTSResult`: Search result with policy and Q-value
- `SharedMemoryConfig`: IPC buffer configuration

**Architecture**:
```
MCTSController
  │
  ├─► EvaluationManager (1 process)
  │   └─► Batches neural network evaluations
  │
  └─► SearchWorkers (N processes)
      └─► Run MCTS simulations in parallel
```

**IPC Design**:
- **Shared Memory**: NumPy arrays (zero-copy transfer)
  - Input buffer: `(8, 8, 34)` float32 = 8,704 bytes
  - Output buffer: `(3 + 4672)` float32 = 18,700 bytes
  - Total per buffer: 27,404 bytes
- **Queues**: Lightweight coordination (task dispatch, results)
- **Buffer Pool**: 256 buffers by default (configurable)

**Input**: 
- `fen: str`: Position in FEN notation
- `num_simulations: int`: Total simulations across all workers

**Output**: 
- `MCTSResult`: Aggregated policy + Q-value
  - `policy: Dict[chess.Move, float]`: Visit count proportions
  - `q_value: float`: Average Q-value across all simulations
  - `error: Optional[str]`: Error message if search failed

**Data Contract**:
- FEN format: Standard chess FEN string
- Policy: Normalized visit counts (sum to 1.0)
- Q-value: Average value estimate in [-1, 1]

#### 4.2 Worker (`worker.py`)

**Purpose**: Individual MCTS search worker process.

**Key Classes**:
- `SearchWorker`: Worker process
- `SearchTask`: Task specification
- `SearchResult`: Worker's search result

**Algorithm**: Standard MCTS (Selection → Expansion → Simulation → Backpropagation)

**Data Contract**:
- Communicates via shared memory + queues
- Returns visit counts and Q-values for aggregation

#### 4.3 Manager (`manager.py`)

**Purpose**: Batches neural network evaluation requests.

**Key Class**: `EvaluationManager`

**Features**:
- Batches multiple evaluation requests
- Waits up to `max_wait_time_ms` for batching
- Loads model once and reuses across requests
- Returns results via shared memory

**Data Contract**:
- Input: Batched encoded positions via shared memory
- Output: Batched value + policy outputs via shared memory

---

### 5. Data Pipeline (`src/data/`)

#### 5.1 Replay Buffer (`replay_buffer.py`)

**Purpose**: High-performance storage for MCTS self-play training data.

**Key Classes**:
- `TalExperience`: Single training sample
- `HDF5ReplayBuffer`: Sharded HDF5 storage
- `TalDataset`: PyTorch IterableDataset

**Data Schema**:
```python
TalExperience:
  state: np.ndarray[34, 8, 8]  # NCHW format, float16
  policy: np.ndarray[4672]      # float16
  value: float                   # float32
  outcome: float                 # float32 (game result)
```

**Storage Format**:
- **Sharding**: ~5000 positions per HDF5 file
- **Compression**: LZF (fast decompression)
- **Type**: float16 for states/policies (2x space savings)
- **Layout**: Pre-transposed to NCHW (avoids runtime transpose)

**Data Contract**:
- Input: List of `TalExperience` objects
- Output: PyTorch `IterableDataset` compatible with `DataLoader`
- File format: HDF5 with datasets `states`, `policies`, `values`, `outcomes`

---

## Data Contracts

### Core Data Types

#### Board Representation

**Type**: `chess.Board` (python-chess library)

**Properties**:
- Standard chess board state
- Turn tracking (white/black)
- Move history
- Game state (checkmate, stalemate, etc.)

**Usage**: Primary input/output for all position-based operations

---

#### Encoded Position

**Type**: `np.ndarray[8, 8, 34]`, dtype=`float32`

**Format**: Height × Width × Channels (HWC)

**Channels**: See [Encoder Channel Mapping](#channel-mapping) above

**Normalization**: Features normalized to reasonable ranges (typically [-1, 1] or [0, 1])

**Orientation**: Board flipped vertically (rank 0 = bottom, rank 7 = top)

---

#### Policy Distribution

**Type**: `Dict[chess.Move, float]`

**Properties**:
- Keys: Legal moves (`chess.Move` objects)
- Values: Probabilities (should sum to ~1.0)
- Normalization: Softmax over legal moves only

**Usage**: 
- Neural network output (after masking illegal moves)
- MCTS visit count proportions
- Move selection probabilities

---

#### Value Estimate

**Type**: `float` in range `[-1, 1]`

**Semantics**:
- `+1.0`: Current player is winning decisively
- `0.0`: Draw or equal position
- `-1.0`: Current player is losing decisively

**Perspective**: Always from current player's perspective (no negation needed)

---

#### Action Index

**Type**: `int` in range `[0, 4671]`

**Mapping**: See [Move Mapping](#move-mapping) section

**Invalid Moves**: Represented as `None` or masked with `-inf` in logits

---

### Interface Contracts

#### TransformerEvaluator Interface

```python
class TransformerEvaluator:
    def evaluate(board: chess.Board) -> Tuple[float, Dict[chess.Move, float]]:
        """
        Returns:
            value: float in [-1, 1]
            policy: Dict[chess.Move, float] (normalized probabilities)
        """
```

**Preconditions**:
- `board` is a valid `chess.Board` object
- Model weights exist at `weights_path`

**Postconditions**:
- Policy contains only legal moves
- Policy probabilities sum to ~1.0
- Value is from current player's perspective

---

#### MCTSController Interface

```python
class MCTSController:
    def run_search(fen: str, num_simulations: int) -> MCTSResult:
        """
        Returns:
            MCTSResult with:
                policy: Dict[chess.Move, float] (visit proportions)
                q_value: float (average Q-value)
                error: Optional[str] (if search failed)
        """
```

**Preconditions**:
- `fen` is valid FEN string
- `num_simulations > 0`
- Controller is started (`start()` called)

**Postconditions**:
- Policy contains only legal moves
- Policy probabilities sum to ~1.0
- Q-value in [-1, 1]

---

#### CoachTalSelector Interface

```python
class CoachTalSelector:
    def select_from_board(board: chess.Board) -> SelectionResult:
        """
        Returns:
            SelectionResult with:
                chosen_move: chess.Move
                chosen_analysis: MoveCandidate
                all_candidates: List[MoveCandidate]
                root_value: float
                fallback_used: bool
        """
```

**Preconditions**:
- `board` is valid and not terminal
- Config is valid (`weights_path` exists)

**Postconditions**:
- `chosen_move` is legal
- All candidates analyzed with cognitive metrics
- Soundness constraint applied

---

### Tal-RL PPO Training Data Contracts

#### VectorizedChessEnv Interface

```python
class VectorizedChessEnv:
    def reset(key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        """
        Returns:
            obs: (B, 34, 8, 8) initial observations
            state: pgx.State for stepping
        """
    
    def step(state: Any, actions: jnp.ndarray) -> Tuple[StepResult, Any]:
        """
        Args:
            state: pgx.State
            actions: (B,) action indices in [0, 4671]
        
        Returns:
            result: StepResult with obs, rewards, dones, etc.
            new_state: Updated pgx.State
        """
```

**Preconditions**:
- `key` is valid JAX random key
- `actions` are legal (enforced by legal_mask)
- `state` is valid pgx.State

**Postconditions**:
- `obs` shape: `(B, 34, 8, 8)` in NCHW format
- `rewards` shape: `(B,)` in [-1, 1]
- `dones` shape: `(B,)` boolean
- `current_player` shape: `(B,)` (0=Agent/White, 1=Victim/Black)

---

#### TalModelJAX Interface

```python
class TalModelJAX:
    def apply(params: Dict, obs: jnp.ndarray, train: bool) -> ModelOutput:
        """
        Args:
            params: Flax parameter dictionary
            obs: (B, 34, 8, 8) or (B, 8, 8, 34) observations
            train: Training mode flag
        
        Returns:
            ModelOutput with:
                value: (B, 3) W/D/L probabilities
                policy_logits: (B, 4672) raw action logits
        """
```

**Preconditions**:
- `params` contains all required model parameters
- `obs` is valid observation tensor
- Model initialized with correct architecture

**Postconditions**:
- `value` probabilities sum to 1.0 per sample
- `policy_logits` are unnormalized (softmax not applied)
- Outputs are JAX arrays on same device as input

---

#### BatchedMCTS Interface

```python
class BatchedMCTS:
    def search(
        params: Dict,
        key: jax.random.PRNGKey,
        obs: jnp.ndarray,
        legal_mask: jnp.ndarray,
        env_step_fn: Callable,
    ) -> MCTSOutput:
        """
        Args:
            params: Model parameters
            key: Random key
            obs: (B, C, H, W) observations
            legal_mask: (B, A) boolean mask
            env_step_fn: (state, action) -> (next_state, reward, done)
        
        Returns:
            MCTSOutput with:
                policy: (B, A) visit count proportions
                q_value: (B,) root Q-values
                action: (B,) selected actions
        """
```

**Preconditions**:
- `env_step_fn` is provided (raises ValueError if None)
- `legal_mask` matches observation batch size
- `obs` is valid observation tensor

**Postconditions**:
- `policy` sums to 1.0 per sample
- `q_value` in [-1, 1] range
- `action` indices are legal (within legal_mask)

---

#### TalRewardEngine Interface

```python
class TalRewardEngine:
    def compute_rewards(
        q_truth: jnp.ndarray,
        v_victim: jnp.ndarray,
        pi_victim: jnp.ndarray,
        game_outcomes: jnp.ndarray,
        sound_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            q_truth: (B,) MCTS Q-values
            v_victim: (B,) Victim network values
            pi_victim: (B, A) Victim policy
            game_outcomes: (B,) Terminal rewards
            sound_mask: (B, A) Sound move mask
        
        Returns:
            rewards: (B,) Composite Tal rewards
        """
```

**Preconditions**:
- All inputs have matching batch size B
- `pi_victim` probabilities sum to ~1.0 per sample
- `sound_mask` is boolean

**Postconditions**:
- `rewards` shape: `(B,)`
- Rewards normalized if `normalize_rewards=True`
- Rewards clipped if `reward_clip` is set

---

#### RolloutBuffer Interface

```python
class RolloutBuffer:
    def add(
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        q_truth: Optional[torch.Tensor],
        victim_entropy: Optional[torch.Tensor],
    ) -> None:
        """Add single timestep of experience."""
    
    def compute_returns_and_advantages(
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and returns."""
    
    def get_batches(batch_size: int) -> Iterator[Batch]:
        """Yield minibatches for PPO update."""
```

**Preconditions**:
- All tensors on same device
- Batch sizes match `num_envs`
- `add()` called exactly `num_steps` times before `compute_returns_and_advantages()`

**Postconditions**:
- Advantages normalized once over entire rollout (not per minibatch)
- Returns = advantages + values
- Minibatches are shuffled if `shuffle=True`

---

#### PPOTrainer Interface

```python
class PPOTrainer:
    def update(buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Args:
            buffer: RolloutBuffer with computed advantages
        
        Returns:
            Dictionary of training metrics
        """
```

**Preconditions**:
- `buffer.ready == True` (GAE computed)
- Model parameters require gradients
- Optimizer initialized

**Postconditions**:
- Model parameters updated
- Metrics dictionary contains: policy_loss, value_loss, entropy, kl, clip_fraction, explained_variance

---

## Data Flow

### Move Selection Flow

```
1. User/GUI requests move for position
   │
   ▼
2. CoachTalSelector.select_from_board(board)
   │
   ├─► Option A: MCTS Path
   │   │
   │   ├─► MCTSController.run_search(fen, num_sims)
   │   │   │
   │   │   ├─► Distribute simulations to workers
   │   │   │   │
   │   │   │   ├─► Worker: Selection → Expansion → Simulation
   │   │   │   │   │
   │   │   │   │   └─► EvaluationManager.evaluate(encoded_state)
   │   │   │   │       │
   │   │   │   │       ├─► Encoder.encode(board) → (8,8,34)
   │   │   │   │       │
   │   │   │   │       └─► Neural Network → (value, policy_logits)
   │   │   │   │
   │   │   └─► Aggregate visit counts → policy distribution
   │   │
   │   └─► Get top-k candidates from MCTS policy
   │
   ├─► Option B: Direct Policy Path
   │   │
   │   └─► TransformerEvaluator.evaluate(board)
   │       │
   │       ├─► Encoder.encode(board) → (8,8,34)
   │       │
   │       └─► Neural Network → (value, policy_logits)
   │           │
   │           └─► Mask illegal moves → policy distribution
   │
   ▼
3. For each candidate move:
   │
   ├─► Make move: board.push(move)
   │
   ├─► Evaluate resulting position:
   │   │
   │   ├─► TransformerEvaluator.evaluate(board_after_move)
   │   │   └─► Get value: V_φ(s')
   │   │
   │   ├─► OpponentModel.get_entropy(board_after_move)
   │   │   └─► Get entropy: H_opp(s')
   │   │
   │   └─► Simulate opponent reply → UserModel.get_ease(board_after_reply)
   │       └─► Get ease: E_user(s'')
   │
   ├─► Compute cognitive asymmetry score:
   │   │
   │   └─► J(s') = λ·V + γ·(H_opp - E_user)
   │
   └─► Check soundness: V(s') ≥ V(s) - δ
   │
   ▼
4. Filter to sound moves
   │
   ▼
5. Select move with highest J(s')
   │
   ▼
6. Return SelectionResult with chosen move + analysis
```

### Training Data Flow

```
1. MCTS Self-Play generates games
   │
   ├─► For each position in game:
   │   │
   │   ├─► MCTS search → policy distribution
   │   │
   │   ├─► Game outcome (win/draw/loss)
   │   │
   │   └─► Create TalExperience:
   │       │
   │       ├─► state: Encoder.encode(board) → (34,8,8) NCHW
   │       ├─► policy: MCTS visit proportions → (4672,)
   │       ├─► value: Game outcome → float
   │       └─► outcome: Game result → float
   │
   ▼
2. HDF5ReplayBuffer.add_game(experiences)
   │
   ├─► Buffer in memory (up to ~5000 positions)
   │
   └─► When threshold reached → flush to HDF5 shard
       │
       ├─► Convert to float16
       │
       ├─► Apply LZF compression
       │
       └─► Write to disk: replay_data/shard_<uuid>.h5
   │
   ▼
3. Training loop:
   │
   ├─► TalDataset loads from HDF5 shards
   │
   ├─► DataLoader batches samples
   │
   └─► Model training:
       │
       ├─► Forward: states → (values, policies)
       │
       ├─► Loss: MSE(value) + CrossEntropy(policy)
       │
       └─► Backward: Update weights
```

### Tal-RL PPO Training Flow

```
1. Initialize Training Pipeline
   │
   ├─► VectorizedChessEnv(num_envs=4096)
   ├─► TalModelJAX (Flax) for MCTS/Victim
   ├─► HybridChessModel (PyTorch) for PPO updates
   ├─► BatchedMCTS(num_simulations=50)
   ├─► VictimModel(temperature=1.5)
   ├─► TalRewardEngine(alpha=0.3, beta=0.2)
   ├─► RolloutBuffer(num_steps=128, num_envs=4096)
   └─► PPOTrainer(model, config)
   │
   ▼
2. Reset Environment
   │
   └─► env.reset(key) → (obs: [B, 34, 8, 8], state: pgx.State)
   │
   ▼
3. Rollout Phase (for num_steps iterations)
   │
   ├─► For each step:
   │   │
   │   ├─► Get legal action mask: env.get_legal_actions(state)
   │   │
   │   ├─► Check whose turn: env.is_agent_turn(state)
   │   │
   │   ├─► If Agent's turn (White):
   │   │   │
   │   │   ├─► BatchedMCTS.search(params, key, obs, legal_mask, env_step_fn)
   │   │   │   │
   │   │   │   ├─► mctx.gumbel_muzero_policy(...)
   │   │   │   │   │
   │   │   │   │   ├─► Recurrent function uses env_step_fn for state transitions
   │   │   │   │   │
   │   │   │   │   └─► Returns: policy [B, A], q_value [B], action [B]
   │   │   │   │
   │   │   └─► Sample action from MCTS policy
   │   │
   │   ├─► If Victim's turn (Black):
   │   │   │
   │   │   ├─► VictimModel(obs) → (value, policy, entropy)
   │   │   │
   │   │   └─► Sample action from raw policy (T=1.5)
   │   │
   │   ├─► Step environment: env.step(state, actions)
   │   │   │
   │   │   └─► Returns: (StepResult, new_state)
   │   │       │
   │   │       ├─► obs: [B, 34, 8, 8]
   │   │       ├─► rewards: [B] (game outcomes)
   │   │       ├─► dones: [B] (termination flags)
   │   │       └─► current_player: [B] (0=Agent, 1=Victim)
   │   │
   │   ├─► Compute Tal Rewards:
   │   │   │
   │   │   ├─► TalRewardEngine.compute_rewards(
   │   │   │       q_truth=mcts_q_value,
   │   │   │       v_victim=victim_value,
   │   │   │       pi_victim=victim_policy,
   │   │   │       game_outcomes=rewards,
   │   │   │       sound_mask=legal_mask,
   │   │   │   )
   │   │   │
   │   │   └─► Returns: [B] composite rewards
   │   │       │
   │   │       └─► R = R_outcome + α·(1 - M_surv) + β·Gap
   │   │
   │   ├─► Convert JAX → PyTorch (DLPack bridge):
   │   │   │
   │   │   ├─► obs_torch = jax_to_torch(obs)
   │   │   ├─► rewards_torch = jax_to_torch(tal_rewards)
   │   │   └─► q_truth_torch = jax_to_torch(q_truth)
   │   │
   │   ├─► Get PyTorch model outputs:
   │   │   │
   │   │   ├─► value_probs, policy_logits = pytorch_model(obs_torch)
   │   │   ├─► value = value_probs[:, 2] - value_probs[:, 0]
   │   │   └─► log_prob = log_softmax(policy_logits)[actions]
   │   │
   │   └─► Add to RolloutBuffer:
   │       │
   │       └─► buffer.add(obs, action, reward, done, value, log_prob, q_truth, victim_entropy)
   │
   ▼
4. Compute Returns and Advantages
   │
   ├─► Get last value: pytorch_model(last_obs)
   │
   └─► buffer.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)
       │
       ├─► Compute GAE (Generalized Advantage Estimation)
       │   │
       │   └─► advantages = δ_t + γ·λ·advantages_{t+1}
       │       where δ_t = r_t + γ·V_{t+1} - V_t
       │
       ├─► returns = advantages + values
       │
       └─► Normalize advantages ONCE over entire rollout
           │
           └─► advantages = (advantages - mean) / (std + 1e-8)
   │
   ▼
5. PPO Update Phase
   │
   ├─► For ppo_epochs iterations:
   │   │
   │   ├─► Shuffle rollout data
   │   │
   │   └─► For each minibatch:
   │       │
   │       ├─► Forward pass: value, policy_logits = model(batch.obs)
   │       │
   │       ├─► Compute new log_probs
   │       │
       │       ├─► Policy Loss (Clipped):
       │       │   │
       │       │   ├─► ratio = exp(new_log_prob - old_log_prob)
       │       │   │
       │       │   └─► loss = -min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
       │       │
       │       ├─► Value Loss:
       │       │   │
       │       │   └─► loss = MSE(value_pred, returns)
       │       │
       │       ├─► Entropy Bonus:
       │       │   │
       │       │   └─► entropy = -sum(p·log(p))
       │       │
       │       ├─► Total Loss:
       │       │   │
       │       │   └─► loss = policy_loss + value_coef·value_loss - entropy_coef·entropy
       │       │
       │       └─► Backward + Optimizer Step
   │
   ▼
6. Logging and Checkpointing
   │
   ├─► TalMetricsLogger.log_iteration(iteration, ppo_metrics, tal_metrics)
   │   │
   │   └─► Logs to WandB:
   │       │
   │       ├─► PPO metrics: policy_loss, value_loss, entropy, kl
   │       ├─► Tal metrics: survival_mass, value_gap
   │       └─► Environment metrics: reward_mean, done_rate
   │
   └─► Save checkpoint (every N iterations)
   │
   ▼
7. Reset and Repeat
   │
   └─► buffer.reset() → Clear for next rollout
   │
   └─► Continue from step 2 until total_timesteps reached
```

**Key Differences from Traditional RL**:
- **Asymmetric Play**: Agent uses MCTS (System 2), Victim uses raw policy (System 1)
- **Tal Reward**: Composite reward includes cognitive asymmetry components
- **Hybrid Backend**: JAX for environment/MCTS, PyTorch for PPO updates
- **Batched Operations**: All 4096 games processed simultaneously

---

## Module Reference

### `src/coach_tal/`

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `selector.py` | Move selection with cognitive asymmetry | `CoachTalSelector`, `CoachTalConfig`, `SelectionResult` |
| `evaluator.py` | Neural network inference wrapper | `TransformerEvaluator` |
| `agents.py` | Opponent/user model proxies | `OpponentModel`, `UserModel` |
| `metrics.py` | Cognitive metric functions | `entropy`, `user_ease`, `cognitive_asymmetry_score` |
| `explainer.py` | Natural language explanations | `Explainer`, `MoveAnalysis` |

### `src/`

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `encoder.py` | Board → tensor encoding | `Encoder` |
| `transformer_model.py` | Keras model architecture | `create_model()` |
| `transformer_model_pytorch.py` | PyTorch model architecture | `HybridChessModel` |
| `move_mapping.py` | Move ↔ index conversion | `move_to_index`, `index_to_move`, `ACTION_SPACE_SIZE` |
| `feature_engineering.py` | Advanced chess features | Various feature functions |
| `utils.py` | Utility functions | `unmirror_policy` (deprecated) |

### `src/mcts/`

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `controller.py` | MCTS controller | `MCTSController`, `MCTSResult` |
| `worker.py` | MCTS worker process | `SearchWorker`, `SearchTask`, `SearchResult` |
| `manager.py` | Evaluation batching | `EvaluationManager` |
| `config.py` | MCTS configuration | MCTS constants |

### `src/data/`

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `replay_buffer.py` | Training data storage | `HDF5ReplayBuffer`, `TalExperience`, `TalDataset` |
| `dataset.py` | Dataset utilities | Dataset helpers |

### `src/training/`

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `losses.py` | Loss functions | Training loss implementations |

### `src/training_ppo/` (Tal-RL PPO Training Pipeline)

**Purpose**: Asynchronous PPO training pipeline for learning cognitive asymmetry through reinforcement learning.

**Architecture Overview**:
```
JAX Environment (pgx, 4096 games)
        │
        ├──► Agent Turn: mctx batched MCTS (50 sims) → Q_truth, π_mcts
        │
        └──► Victim Turn: Raw policy (T=1.5) → π_victim, M_surv
                │
                ▼
        Tal Reward: R = R_outcome + α(1-M_surv) + β·Gap
                │
                ▼
        PPO Learner (PyTorch) → Update Agent weights
```

#### 6.1 Configuration (`config.py`)

**Purpose**: Centralized configuration management for PPO training.

**Key Classes**:
- `PPOTalConfig`: Complete training configuration
- `EnvConfig`: Environment settings (num_envs, max_steps)
- `AgentConfig`: Agent model configuration
- `VictimConfig`: Victim model settings (temperature)
- `MCTSConfig`: Batched MCTS parameters
- `TalRewardConfig`: Cognitive asymmetry reward hyperparameters
- `PPOConfig`: PPO algorithm hyperparameters
- `TrainingConfig`: Training schedule and logging

**Data Contract**:
- YAML serializable via `from_yaml()` / `to_yaml()`
- Nested dataclass structure
- Type-safe with Pydantic-style validation

#### 6.2 Environment (`env/`)

**Purpose**: Vectorized chess environment using JAX/pgx for parallel game simulation.

**Key Classes**:
- `VectorizedChessEnv`: Wraps pgx.make("chess") for batched operations
- `StepResult`: NamedTuple with obs, rewards, dones, current_player, legal_mask

**Key Functions**:
- `pgx_to_tal_encoding()`: Converts pgx observations to 34-channel Tal format
- `tal_to_pgx_action()`: Converts Tal action indices to pgx format

**Data Contracts**:
- **Input**: `jax.random.PRNGKey` for initialization
- **Output**: `(obs: jnp.ndarray[B, 34, 8, 8], state: pgx.State)`
- **Actions**: `jnp.ndarray[B]` action indices in range [0, 4671]
- **Observations**: `(B, 34, 8, 8)` in NCHW format (matches Tal model)

**Features**:
- Runs 4096 parallel games simultaneously
- JIT-compiled step function for GPU acceleration
- Automatic reset of terminated environments
- Asymmetric turn handling (Agent=White, Victim=Black)

#### 6.3 Models (`models/`)

**Purpose**: JAX/Flax model implementations and PyTorch bridges.

**Key Classes**:
- `TalModelJAX`: Flax port of HybridChessModel
  - Architecture: CNN Stem → Transformer → Dual Heads
  - Input: `(B, 8, 8, 34)` or `(B, 34, 8, 8)`
  - Output: `ModelOutput(value: [B, 3], policy_logits: [B, 4672])`
  - Weight loading: `from_pytorch()` converts PyTorch checkpoints
  
- `VictimModel`: Frozen model with elevated temperature
  - Wraps TalModelJAX with `requires_grad=False`
  - Temperature scaling: `logits / T` where T=1.5
  - Returns: `VictimOutput(value, policy, entropy)`

- `jax_bridge.py`: PyTorch ↔ JAX conversion utilities
  - `torch_to_jax()`: DLPack zero-copy transfer
  - `jax_to_torch()`: Reverse conversion
  - `pytorch_state_dict_to_flax()`: Weight format conversion

**Data Contracts**:
- **Model Input**: `jnp.ndarray[B, C, H, W]` or `jnp.ndarray[B, H, W, C]`
- **Model Output**: `ModelOutput` with:
  - `value`: `(B, 3)` W/D/L probabilities
  - `policy_logits`: `(B, 4672)` raw action logits
- **Scalar Value**: `value[:, 2] - value[:, 0]` (Win - Loss) in [-1, 1]

#### 6.4 MCTS (`mcts/`)

**Purpose**: Vectorized Monte Carlo Tree Search using DeepMind's mctx library.

**Key Classes**:
- `BatchedMCTS`: Vectorized MCTS for parallel search
  - Uses `mctx.gumbel_muzero_policy` for batched search
  - Processes all positions simultaneously (4096 games)
  - Returns: `MCTSOutput(policy, q_value, action, search_tree)`

**Key Methods**:
- `search()`: Run batched MCTS on observation batch
- `_make_recurrent_fn()`: Creates dynamics function for mctx
  - **Requires**: `env_step_fn` for state transitions (raises error if missing)

**Data Contracts**:
- **Input**: 
  - `obs`: `(B, C, H, W)` observations
  - `legal_mask`: `(B, A)` boolean mask
  - `env_step_fn`: `(state, action) -> (next_state, reward, done)`
- **Output**:
  - `policy`: `(B, A)` visit count proportions
  - `q_value`: `(B,)` root Q-values (V_truth)
  - `action`: `(B,)` selected actions

**Configuration**:
- `num_simulations`: 50 (fast) to 200 (accurate)
- `max_num_considered_actions`: 16 (prune for speed)
- `use_gumbel`: True (recommended for exploration)

#### 6.5 Rewards (`rewards/`)

**Purpose**: Tal reward computation for cognitive asymmetry training.

**Key Classes**:
- `TalRewardEngine`: Computes composite cognitive asymmetry rewards
  - Formula: `R = R_outcome + α·(1 - M_surv) + β·Gap`
  - Components:
    - `R_outcome`: Game result (+1/-1/0)
    - `M_surv`: Survival mass (victim probability on sound moves)
    - `Gap`: Value gap (Q_truth - V_victim)
  
- `RunningMeanStd`: Running statistics for reward normalization
  - Uses Welford's online algorithm
  - Critical for training stability

- `RewardScaler`: Multi-component reward scaling
  - Tracks separate stats for each component
  - Normalizes composite reward

**Data Contracts**:
- **Input**:
  - `q_truth`: `(B,)` MCTS Q-values
  - `v_victim`: `(B,)` Victim network values
  - `pi_victim`: `(B, A)` Victim policy distribution
  - `game_outcomes`: `(B,)` Terminal rewards
  - `sound_mask`: `(B, A)` Boolean mask of sound moves
- **Output**: `(B,)` Composite Tal rewards (normalized if enabled)

**Key Metrics**:
- `survival_mass`: Probability victim places on sound moves [0, 1]
  - Lower = better (more traps)
- `value_gap`: Deception metric (Q_truth - V_victim)
  - Higher = better (opponent underestimates danger)

#### 6.6 Storage (`storage/`)

**Purpose**: On-policy rollout buffer for PPO training.

**Key Classes**:
- `RolloutBuffer`: PPO-style experience storage
  - Stores: obs, actions, rewards, dones, values, log_probs
  - Tal-specific: q_truth, victim_entropy
  - Computes GAE (Generalized Advantage Estimation)
  - Generates minibatches for PPO updates

**Data Contracts**:
- **Storage Shape**: `(T, B, ...)` where T=num_steps, B=num_envs
- **GAE Computation**: 
  - `advantages = compute_gae(trajectory, last_value, gamma, gae_lambda)`
  - Normalized once over entire rollout (not per minibatch)
- **Minibatch Output**: `Batch` NamedTuple with:
  - `obs`: `(batch_size, C, H, W)`
  - `actions`: `(batch_size,)`
  - `advantages`: `(batch_size,)` (normalized)
  - `returns`: `(batch_size,)` (advantages + values)
  - `old_log_probs`: `(batch_size,)`
  - `values`: `(batch_size,)`
  - `q_truth`: `(batch_size,)`
  - `victim_entropy`: `(batch_size,)`

**Lifecycle**:
1. `add()`: Collect experience during rollout
2. `compute_returns_and_advantages()`: Compute GAE after rollout
3. `get_batches()`: Yield minibatches for PPO update
4. `reset()`: Clear for next rollout

#### 6.7 Trainer (`trainer/`)

**Purpose**: PPO algorithm implementation with clipped objective.

**Key Classes**:
- `PPOTrainer`: PPO with clipped surrogate objective
  - Policy loss: `L^CLIP = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]`
  - Value loss: MSE or clipped MSE
  - Entropy bonus: Encourages exploration
  - Gradient clipping: Prevents exploding gradients

**Data Contracts**:
- **Input**: `RolloutBuffer` with computed advantages
- **Output**: `Dict[str, float]` with metrics:
  - `policy_loss`, `value_loss`, `entropy`
  - `kl`: Approximate KL divergence
  - `clip_fraction`: How often clipping was active
  - `explained_variance`: Value prediction quality

**Configuration**:
- `lr`: 3e-4 (learning rate)
- `clip_range`: 0.2 (PPO clip epsilon)
- `entropy_coef`: 0.01 (entropy bonus weight)
- `value_coef`: 0.5 (value loss weight)
- `ppo_epochs`: 4 (updates per rollout)
- `minibatch_size`: 256

#### 6.8 Metrics (`metrics/`)

**Purpose**: WandB integration and comprehensive metrics tracking for verifying Tal-style play.

**Key Classes**:
- `TalMetricsLogger`: WandB logger with multi-category metrics
  - Aggregates: step, episode, style, safety, and iteration metrics
  - Logs: PPO, Tal, Style, Safety, and Performance metrics
  - Tracks: VRAM usage for 8GB GPU constraints

- `MetricAggregator`: Running statistics aggregator
  - Computes: mean, std, min, max, count
  - Supports: reset after logging

**Metric Categories**:

| Category | Prefix | Purpose |
|----------|--------|---------|
| Cognitive | `tal/*` | Deception metrics (survival mass, value gap) |
| Style | `style/*` | Tal personality verification (material, chaos) |
| Safety | `safety/*` | Hope chess prevention (suicide rate) |
| PPO Health | `ppo/*` | Algorithm stability (entropy, KL, clip fraction) |
| Environment | `env/*` | Game outcomes (win rate, game length) |
| Performance | `perf/*` | System health (SPS, VRAM usage) |

**Key Metrics**:

*Cognitive Asymmetry (Target: increase deception)*
- `tal/survival_mass_mean`: Victim probability on sound moves (target: ↘)
- `tal/value_gap_mean`: Q_truth - V_victim (target: ↗)

*Style Verification (Target: Tal-like play)*
- `style/material_imbalance_mean`: White - Black material (target: low/negative)
- `style/chaos_index_mean`: Opponent sound moves available (target: ↘)

*Safety Checks (Target: avoid gambling)*
- `safety/agent_suicide_rate`: Moves with Q < -0.5 (target: < 0.05)

*PPO Health (Target: stable training)*
- `ppo/entropy`: Exploration health (target: stable ~1.0)
- `ppo/explained_variance`: Value prediction quality (target: > 0.5)
- `ppo/clip_fraction`: PPO clipping rate (target: < 0.2)

*Environment (Target: winning games)*
- `env/win_rate`: Win percentage (target: ↗)
- `env/game_length_mean`: Moves per game (target: 40-60)

*Performance (Target: efficient training)*
- `perf/steps_per_second`: Training throughput
- `perf/vram_gb`: GPU memory usage (target: < 7.5GB)

#### 6.10 Style Metrics (`metrics/style_metrics.py`)

**Purpose**: Pure functions for computing Tal personality verification metrics.

**Key Functions**:
- `compute_material_imbalance(obs)`: Extract material from piece channels
  - Channels 0-5: Black pieces, 6-11: White pieces
  - Weights: P=1, N=3, B=3, R=5, Q=9, K=0
  - Returns: White - Black (negative = sacrificing, Tal-style!)

- `compute_chaos_index(legal_mask, sound_mask)`: Count opponent's good moves
  - Low value = opponent under pressure ("only moves")
  - Target: decrease during training

- `detect_agent_suicide(q_truth, threshold=-0.5)`: Flag losing moves
  - Prevents "hope chess" gambling
  - Target: < 5% of moves

**JAX Versions**: `*_jax` variants for use in JIT-compiled loops

#### 6.9 Specifications (`specs.py`)

**Purpose**: Centralized constants to avoid magic numbers.

**Key Constants**:
- `ACTION_SPACE_SIZE = 4672`
- `INPUT_CHANNELS = 34`
- `BOARD_SIZE = 8`
- `AGENT = 0` (White)
- `VICTIM = 1` (Black)

---

## Module Reference

### `src/training_ppo/` (Tal-RL PPO Training Pipeline)

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `config.py` | Configuration schemas | `PPOTalConfig`, `PPOConfig`, `TalRewardConfig`, `EnvConfig`, `AgentConfig`, `VictimConfig`, `MCTSConfig`, `TrainingConfig` |
| `specs.py` | Centralized constants | `ACTION_SPACE_SIZE`, `INPUT_CHANNELS`, `BOARD_SIZE`, `AGENT`, `VICTIM` |
| `env/chess_env.py` | Vectorized chess environment | `VectorizedChessEnv`, `StepResult`, `create_env` |
| `env/encoding.py` | pgx ↔ Tal encoding adapter | `pgx_to_tal_encoding`, `tal_to_pgx_action`, `pgx_to_tal_action` |
| `models/tal_jax.py` | Flax/JAX model implementation | `TalModelJAX`, `ModelOutput`, `create_model`, `forward_inference`, `forward_training` |
| `models/victim.py` | Frozen victim model | `VictimModel`, `VictimOutput`, `create_victim` |
| `models/jax_bridge.py` | PyTorch ↔ JAX bridge | `torch_to_jax`, `jax_to_torch`, `pytorch_state_dict_to_flax`, `load_pytorch_weights_to_flax` |
| `mcts/batched_mcts.py` | Vectorized MCTS | `BatchedMCTS`, `MCTSOutput`, `SimplifiedMCTS`, `create_mcts` |
| `rewards/tal_reward.py` | Cognitive asymmetry rewards | `TalRewardEngine`, `TalRewardConfig`, `TalRewardEngineJIT`, `create_reward_engine` |
| `rewards/normalizer.py` | Reward normalization | `RunningMeanStd`, `RewardScaler`, `RunningMeanStdJAX` |
| `storage/rollout_buffer.py` | PPO rollout buffer | `RolloutBuffer`, `Batch`, `EpisodeStats`, `create_buffer` |
| `trainer/ppo.py` | PPO algorithm | `PPOTrainer`, `PPOConfig`, `create_trainer` |
| `metrics/logger.py` | WandB logging | `TalMetricsLogger`, `MetricAggregator`, `create_logger` |
| `metrics/style_metrics.py` | Tal personality metrics | `compute_material_imbalance`, `compute_chaos_index`, `detect_agent_suicide` |

### `scripts/`

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `train_ppo.py` | Main PPO training entry point | `main()`, `parse_args()`, `load_config()`, `set_seeds()` |
| `test_ppo_pipeline.py` | Pipeline test suite | `run_all_tests()`, individual test functions |

### `configs/`

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `ppo_tal.yaml` | PPO training configuration | YAML config for Tal-RL training (all hyperparameters) |

### `requirements_ppo.txt`

| Purpose | Contents |
|---------|----------|
| PPO pipeline dependencies | JAX, pgx, mctx, wandb, flax, pydantic, etc. |

---

## Version History

- **v1.0**: Initial architecture with Coach Tal cognitive asymmetry
- **v1.1**: Added MCTS support with shared memory IPC
- **v1.2**: Removed mirroring from inference (model handles both perspectives)
- **v1.3**: Reorganized project structure (bin/, docs/, scripts/ subdirectories)
- **v2.0**: Tal-RL PPO Training Pipeline
  - Added vectorized chess environment (JAX/pgx)
  - Implemented Flax/JAX model port with PyTorch weight loading
  - Added batched MCTS using mctx library
  - Implemented Tal reward engine (cognitive asymmetry)
  - Added PPO trainer with clipped objective
  - Created rollout buffer with GAE computation
  - Added WandB logging with Tal-specific metrics
  - Fixed MCTS recurrence bug (require env_step_fn)
  - Centralized magic numbers in specs.py
- **v2.1**: Tal Personality Metrics
  - Added style_metrics.py with material imbalance, chaos index, suicide detection
  - Extended RolloutBuffer with EpisodeStats tracking
  - Added episode-level statistics (win_rate, game_length, material_in_wins)
  - Extended TalMetricsLogger with style/*, safety/*, perf/* metric categories
  - Added VRAM tracking for 8GB GPU constraints
  - Multi-line console logging with PPO health, Tal metrics, game stats

---

## Future Extensions

### Planned Components

1. **Opening Book**: Database of opening moves
2. **Endgame Tablebase**: Perfect endgame play
3. **Time Management**: Adaptive time allocation
4. **Multi-PV Analysis**: Multiple principal variations
5. **Pondering**: Think during opponent's time

### Extension Points

- **Custom Metrics**: Add new functions to `metrics.py`
- **Custom Agents**: Implement new agent types in `agents.py`
- **Custom Evaluators**: Implement `Evaluator` interface
- **Custom Encoders**: Implement `Encoder` interface

---

## References

- [python-chess Documentation](https://python-chess.readthedocs.io/)
- [UCI Protocol Specification](http://wbec-ridderkerk.nl/html/UCIProtocol.html)
- [MCTS Algorithm](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)


