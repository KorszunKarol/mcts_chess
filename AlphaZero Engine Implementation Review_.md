

# **Technical Implementation Guide for an AlphaZero-Style Chess Engine**

## **I. The High-Performance Asynchronous MCTS Engine**

This section details the definitive architecture for the core search component. The design choices here are foundational and dictate the performance ceiling of the entire system. The primary challenge is to overcome the Python Global Interpreter Lock (GIL) to achieve true parallelism for the CPU-bound Monte Carlo Tree Search (MCTS) and to establish a high-throughput communication channel between the CPU-based search processes and the GPU-based neural network evaluation process.

### **1.1. Concurrency Strategy: multiprocessing as the Unavoidable Choice**

A critical analysis of Python's concurrency models, threading and multiprocessing, reveals that only one is suitable for the CPU-intensive workload of MCTS.

#### **Analysis of the Global Interpreter Lock (GIL)**

The Python GIL is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecode at the same time within a single process.1 For I/O-bound tasks, where threads spend time waiting for external resources, the GIL is released, allowing for effective concurrency. However, the MCTS search—which involves continuous tree traversal, state updates, and PUCT score calculations—is a purely CPU-bound task. In a

threading-based model, even with multiple search threads on a multi-core CPU, the GIL would permit only one thread to execute its Python code at any given moment. The other threads would be idle, waiting for the lock. This effectively serializes the work of all search threads, completely negating the benefit of having multiple CPU cores and failing to achieve true parallelism.2

#### **multiprocessing as the Definitive Solution**

The multiprocessing module circumvents the GIL by creating separate operating system processes, each with its own independent Python interpreter, memory space, and GIL.1 Each MCTS

SearchWorker can be instantiated as a separate process, allowing it to run on a dedicated CPU core without being blocked by the GIL of other workers. This process-based parallelism is the only viable method in Python to fully leverage the computational power of a multi-core CPU for the search algorithm.1

#### **The Architectural Trade-off**

While multiprocessing solves the GIL problem, it introduces a new set of architectural challenges. Unlike threads, which share memory by default, processes have isolated memory spaces.2 This decision has two primary consequences:

1. **Higher Memory Footprint:** Each process loads its own copy of necessary modules and data, increasing overall RAM usage.  
2. **Inter-Process Communication (IPC) Overhead:** Any data that needs to be exchanged between processes (e.g., a board state sent for evaluation) must be explicitly passed through an IPC mechanism. This involves serialization, data transfer, and deserialization, which can become a significant performance bottleneck if not designed carefully.4

The choice of multiprocessing is therefore a foundational architectural decision. It trades the GIL bottleneck for an IPC bottleneck. The remainder of this section is dedicated to designing a system that minimizes this new IPC overhead to unlock the full potential of the parallel architecture.

### **1.2. Inter-Process Communication: A High-Throughput Hybrid Architecture**

The communication between the numerous CPU-based SearchWorker processes and the single GPU-based EvaluationManager process is the most critical performance pathway in the system. A SearchWorker must send a board state representation (a potentially large NumPy array) to the manager and receive back a policy vector and a value estimate (another NumPy array and a float). This must happen thousands of times per second.

#### **Inadequacy of multiprocessing.Queue for Bulk Data**

The standard multiprocessing.Queue is a straightforward IPC tool. However, it operates by pickling (serializing) Python objects into a byte stream, sending these bytes through an underlying OS pipe, and then unpickling them in the receiving process.2 While efficient for small, simple objects, this process is prohibitively slow for large objects like NumPy arrays, which represent our board states and policy vectors. The overhead of serializing and deserializing these arrays for every single evaluation request would create a massive bottleneck, starving the GPU and leaving it idle for long periods while it waits for data.4

#### **The multiprocessing.shared\_memory Advantage**

Introduced in Python 3.8, the multiprocessing.shared\_memory module provides a mechanism for processes to access a common block of raw memory without any serialization or data copying.5 This is the ideal solution for transferring large, raw data structures like NumPy arrays. A NumPy array can be instantiated such that its underlying data buffer points directly to a shared memory segment.6 When one process modifies the array, the changes are instantly visible to all other processes that have access to that same shared memory block. This offers a dramatic performance improvement over

Queue for the actual neural network inputs and outputs, as it reduces the transfer cost to nearly zero.5

#### **Proposed Hybrid IPC Architecture**

A purely Queue-based system is too slow, and a purely shared\_memory-based system is complex to coordinate. The optimal architecture is a hybrid model that separates the high-bandwidth data plane from the low-bandwidth control plane.

* **Data Plane (shared\_memory):** Upon startup, the main process creates a pool of large, pre-allocated shared memory blocks. This pool acts as a set of reusable buffers. For example, one could create 256 buffers, each large enough to hold one NN input (the encoded board state) and one NN output (the policy vector and value). NumPy arrays within the SearchWorkers and the EvaluationManager will be created to view these buffers directly.  
* **Control Plane (multiprocessing.Queue):** Two lightweight, high-performance queues are used for signaling and metadata exchange.  
  * request\_queue: When a SearchWorker has prepared a board state in a shared buffer, it places a simple message, such as the integer index of that buffer (e.g., 127), onto the request\_queue. This message tells the EvaluationManager: "The data in shared buffer \#127 is ready for you to process."  
  * response\_queue: After the EvaluationManager has processed a batch and written the results (policy and value) back into the corresponding shared buffers, it places the buffer indices onto the response\_queue. A waiting SearchWorker can then retrieve its index and know that the results in its designated buffer are ready to be read.

This hybrid approach leverages each technology for its intended purpose. shared\_memory handles the high-volume, latency-sensitive transfer of NumPy arrays with maximum efficiency, while the simple and robust multiprocessing.Queue handles the low-volume, lightweight task of coordinating which buffers are ready.11 This design minimizes IPC overhead and is crucial for keeping the GPU saturated with work.

### **1.3. MCTS Tree Management: The Case for Independent, Per-Move Trees**

The question of how multiple SearchWorker processes should share and update the MCTS game tree is critical.

#### **The Anti-Pattern: Centralized Tree with multiprocessing.Manager**

One might consider using a multiprocessing.Manager to host a shared tree object (e.g., a dictionary mapping node hashes to node objects).11 This approach is fundamentally flawed for high-performance MCTS. Every access to a managed proxy object requires a network-like round-trip communication to the central manager process, which is protected by locks to ensure consistency.13 An MCTS search involves thousands of rapid node traversals and updates per second. Funneling all of this activity through a single manager process would create a catastrophic contention bottleneck, making this pattern entirely non-viable.13

#### **The Optimal Pattern: No Shared Tree**

The key realization is that in the AlphaZero self-play loop, the MCTS tree has an ephemeral lifecycle. A search is conducted to select a single move. Once that move is played, the game state advances, and the *entire search tree from the previous move is discarded*. The search for the next move begins from scratch with a new root node.14 This lifecycle makes a persistent, shared tree unnecessary.

The recommended and most efficient implementation is therefore as follows:

1. **Task Distribution:** For each move decision in a self-play game, the main orchestrator process sends the current board state (as a FEN string or other compact representation) to all idle SearchWorker processes. This state becomes the root of the search for this turn.  
2. **Independent Tree Construction:** Each SearchWorker process builds its own complete, independent MCTS tree in its own private process memory. They are all working on the same search problem (from the same root), but their tree data structures are not shared.  
3. **Result Aggregation:** After the allocated thinking time (or simulation count) is reached, each worker reports its final root node statistics (the visit counts for each child move) back to the main process. The main process aggregates these statistics to form the final policy π for the move.  
4. **Move Selection & Reset:** A move is selected based on the aggregated policy π, the game state advances, and all the temporary, per-process MCTS trees are implicitly destroyed as the workers are assigned the next search task.

This "no shared tree" architecture completely eliminates the immense complexity, overhead, and contention issues associated with synchronizing a complex data structure across multiple processes. It is simple, robust, and scales linearly with the number of CPU cores, as there is no central point of contention for the tree itself. This pattern is not a compromise; it is the most efficient design because it perfectly aligns with the fundamental, per-move nature of the self-play search loop.

### **1.4. Code Skeletons: Core Engine Components**

The following Python code skeletons illustrate the architectural principles described above. They provide a structural foundation for the asynchronous MCTS engine.

#### **MCTS Node Data Structure**

The node must store MCTS statistics and additional state for the Search-Contempt algorithm.

Python

import numpy as np  
import math

class MCTSNode:  
    def \_\_init\_\_(self, parent=None, prior\_p=1.0, depth=0):  
        self.parent \= parent  
        self.children \= {}  \# A map from action to MCTSNode  
          
        \# Core MCTS statistics  
        self.N \= 0  \# Visit count  
        self.Q \= 0.0  \# Mean action value  
        self.P \= prior\_p  \# Prior probability from policy head

        \# State for Search-Contempt MCTS  
        self.depth \= depth  
        self.is\_frozen \= False  
        self.frozen\_visit\_counts \= None

    def expand(self, policy\_output):  
        """Expand the node by creating children for all legal actions."""  
        for action, prob in policy\_output.items():  
            if action not in self.children:  
                self.children\[action\] \= MCTSNode(parent=self, prior\_p=prob, depth=self.depth \+ 1)

    def select\_child(self, c\_puct, n\_scl):  
        """Select a child node using PUCT or Search-Contempt logic."""  
        \# \--- Implementation of Search-Contempt logic goes here \---  
        \# For now, a standard PUCT implementation  
        best\_score \= \-float('inf')  
        best\_action \= \-1  
        best\_child \= None

        for action, child in self.children.items():  
            score \= child.Q \+ c\_puct \* child.P \* (math.sqrt(self.N) / (1 \+ child.N))  
            if score \> best\_score:  
                best\_score \= score  
                best\_action \= action  
                best\_child \= child  
          
        return best\_action, best\_child

    def update(self, value):  
        """Backpropagate the value from a simulation."""  
        self.N \+= 1  
        self.Q \+= (value \- self.Q) / self.N  
        if self.parent:  
            \# The value is from the perspective of the current player.  
            \# For the parent, it's the opposite perspective.  
            self.parent.update(-value)

    def is\_leaf(self):  
        return len(self.children) \== 0

#### **Evaluation Manager (GPU Process)**

This process manages the GPU and performs batched NN inference.

Python

import tensorflow as tf  
from multiprocessing import Process, Queue  
from multiprocessing.shared\_memory import SharedMemory

class EvaluationManager(Process):  
    def \_\_init\_\_(self, request\_q, response\_q, shm\_pool\_config, batch\_size, model\_path):  
        super().\_\_init\_\_()  
        self.request\_q \= request\_q  
        self.response\_q \= response\_q  
        self.shm\_pool\_config \= shm\_pool\_config  
        self.batch\_size \= batch\_size  
        self.model\_path \= model\_path  
        self.model \= None  
        self.shm\_pool \= {} \# To hold SharedMemory objects

    def setup\_gpu(self):  
        """Configure TensorFlow to not pre-allocate all GPU memory."""  
        gpus \= tf.config.list\_physical\_devices('GPU')  
        if gpus:  
            try:  
                for gpu in gpus:  
                    tf.config.experimental.set\_memory\_growth(gpu, True)  
                print(f"GPU memory growth enabled for {len(gpus)} GPU(s).")  
            except RuntimeError as e:  
                print(e)

    def run(self):  
        """Main loop to consume requests and produce evaluations."""  
        self.setup\_gpu()  
        self.model \= tf.keras.models.load\_model(self.model\_path)  
          
        \# Attach to all shared memory blocks  
        for name, spec in self.shm\_pool\_config.items():  
            self.shm\_pool\[name\] \= SharedMemory(name=spec\['name'\])

        while True:  
            requests\_batch \=  
            \# Wait for the first request to start forming a batch  
            requests\_batch.append(self.request\_q.get())  
              
            \# Gather more requests up to batch\_size, with a small timeout  
            while not self.request\_q.empty() and len(requests\_batch) \< self.batch\_size:  
                try:  
                    requests\_batch.append(self.request\_q.get\_nowait())  
                except queue.Empty:  
                    break \# Should not happen due to the outer while loop condition

            \# Prepare batch for inference  
            batch\_indices \= \[req\['buffer\_index'\] for req in requests\_batch\]  
              
            \# Create a numpy view of the input buffers  
            input\_arrays \= \[  
                np.ndarray(self.shm\_pool\_config\['input'\]\['shape'\], dtype=self.shm\_pool\_config\['input'\]\['dtype'\], buffer=self.shm\_pool\[f"input\_{i}"\].buf)  
                for i in batch\_indices  
            \]  
              
            input\_tensor \= np.stack(input\_arrays, axis=0)

            \# Perform NN inference  
            policy\_logits, value \= self.model(input\_tensor, training=False)

            \# Write results back to shared memory  
            for i, buffer\_index in enumerate(batch\_indices):  
                \# Get views of the output buffers  
                policy\_out\_arr \= np.ndarray(self.shm\_pool\_config\['policy'\]\['shape'\], dtype=self.shm\_pool\_config\['policy'\]\['dtype'\], buffer=self.shm\_pool\[f"policy\_{buffer\_index}"\].buf)  
                value\_out\_arr \= np.ndarray(self.shm\_pool\_config\['value'\]\['shape'\], dtype=self.shm\_pool\_config\['value'\]\['dtype'\], buffer=self.shm\_pool\[f"value\_{buffer\_index}"\].buf)  
                  
                \# Copy data  
                policy\_out\_arr\[:\] \= policy\_logits\[i\].numpy()  
                value\_out\_arr \= value\[i\].numpy()

            \# Signal that results are ready  
            for req in requests\_batch:  
                self.response\_q.put({'worker\_id': req\['worker\_id'\], 'buffer\_index': req\['buffer\_index'\]})

#### **Search Worker (CPU Process)**

This process runs the MCTS loop, offloading evaluations to the EvaluationManager.

Python

from multiprocessing import Process, Queue  
from multiprocessing.shared\_memory import SharedMemory  
import chess

class SearchWorker(Process):  
    def \_\_init\_\_(self, worker\_id, task\_q, request\_q, response\_q, shm\_pool\_config):  
        super().\_\_init\_\_()  
        self.worker\_id \= worker\_id  
        self.task\_q \= task\_q          \# Receives root states to search  
        self.request\_q \= request\_q    \# Sends evaluation requests  
        self.response\_q \= response\_q  \# Receives evaluation completions  
        self.shm\_pool\_config \= shm\_pool\_config  
        self.shm\_pool \= {}

    def run(self):  
        \# Attach to all shared memory blocks  
        for name, spec in self.shm\_pool\_config.items():  
            self.shm\_pool\[name\] \= SharedMemory(name=spec\['name'\])

        while True:  
            task \= self.task\_q.get()  
            if task is None: \# Sentinel for termination  
                break  
              
            root\_fen \= task\['fen'\]  
            num\_simulations \= task\['simulations'\]  
            buffer\_index \= task\['buffer\_index'\] \# Pre-assigned buffer for this worker

            board \= chess.Board(root\_fen)  
            root \= MCTSNode()

            for \_ in range(num\_simulations):  
                node \= root  
                search\_path \= \[node\]

                \# 1\. Selection  
                while not node.is\_leaf():  
                    action, node \= node.select\_child(c\_puct=1.0, n\_scl=100) \# Placeholder params  
                    board.push(action)  
                    search\_path.append(node)

                \# 2\. Expansion & Evaluation  
                if not board.is\_game\_over():  
                    \# Prepare board state for NN  
                    encoded\_state \= self.encode\_board(board)  
                      
                    \# Get a view of the input buffer and copy data  
                    input\_arr \= np.ndarray(self.shm\_pool\_config\['input'\]\['shape'\], dtype=self.shm\_pool\_config\['input'\]\['dtype'\], buffer=self.shm\_pool\[f"input\_{buffer\_index}"\].buf)  
                    input\_arr\[:\] \= encoded\_state

                    \# Send evaluation request  
                    self.request\_q.put({'worker\_id': self.worker\_id, 'buffer\_index': buffer\_index})  
                      
                    \# Wait for the response for this specific worker  
                    \# A more advanced implementation might use a per-worker response queue or futures  
                    while True:  
                        response \= self.response\_q.get()  
                        if response\['worker\_id'\] \== self.worker\_id:  
                            break  
                        else: \# Not for me, put it back  
                            self.response\_q.put(response)  
                      
                    \# Get views of the output buffers and read data  
                    policy\_arr \= np.ndarray(self.shm\_pool\_config\['policy'\]\['shape'\], dtype=self.shm\_pool\_config\['policy'\]\['dtype'\], buffer=self.shm\_pool\[f"policy\_{buffer\_index}"\].buf)  
                    value\_arr \= np.ndarray(self.shm\_pool\_config\['value'\]\['shape'\], dtype=self.shm\_pool\_config\['value'\]\['dtype'\], buffer=self.shm\_pool\[f"value\_{buffer\_index}"\].buf)  
                      
                    policy \= self.create\_policy\_dict(policy\_arr, board.legal\_moves)  
                    value \= value\_arr  
                      
                    node.expand(policy)  
                else:  
                    value \= self.get\_game\_outcome(board)

                \# 3\. Backpropagation  
                node.update(value)

            \# Report results (e.g., back to a results queue, not shown)  
            \#...

    def encode\_board(self, board):  
        \# Implementation for converting a python-chess board to a NumPy array  
        pass

    def create\_policy\_dict(self, policy\_array, legal\_moves):  
        \# Implementation to map the policy vector to a dictionary of legal moves and their probabilities  
        pass

    def get\_game\_outcome(self, board):  
        \# Implementation to get game result (+1, \-1, 0\)  
        pass

## **II. Implementing Search-Contempt MCTS for Superior Data Generation**

To move beyond a simple AlphaZero replication and generate higher-quality training data, the implementation of Search-Contempt MCTS is critical. This algorithm modifies the standard PUCT search to force exploration into more "challenging" or "unusual" positions, which is key to exposing the neural network's weaknesses and accelerating learning.15

### **2.1. Algorithm Deep Dive: Asymmetric Search**

The core principle of Search-Contempt is an asymmetric search strategy. The selection logic within the MCTS tree changes depending on whether it is the player-to-move's turn or the opponent's turn, relative to the root of the search.15

* **Player-to-Move Nodes (Even Depth):** At the root node (depth 0\) and all its descendants at even depths (2, 4,...), the engine uses the standard **PUCT algorithm** to select the next move. This ensures the engine is always trying to find the objectively best line for itself, maximizing its own expected outcome based on current knowledge.15 The selection formula is:U(s,a)=Q(s,a)+cpuct​⋅P(s,a)⋅1+N(s,a)∑b​N(s,b)​​  
* **Opponent Nodes (Odd Depth):** At nodes of odd depth (1, 3, 5,...), which represent the opponent's possible replies, the search strategy is more complex and designed to promote diversity.

### **2.2. The Nscl Parameter and Thompson Sampling**

The behavior at opponent nodes is governed by a new hyperparameter, Nscl (Search-Contempt-Node-Limit), which defines a visit count threshold.15

The search logic at an opponent node s is as follows:

1. **Initial PUCT Phase:** As long as the total number of simulations that have passed through node s is less than or equal to Nscl (i.e., ∑b​N(s,b)≤Nscl​), the search continues to use the standard PUCT algorithm to select the opponent's move.  
2. **Distribution Freezing:** The moment the total visit count at node s exceeds Nscl, the current visit distribution across its children is "frozen". The visit counts for each child action a, denoted Nfrozen​(s,a), are stored permanently for that node for the remainder of the current search.  
3. **Thompson Sampling Phase:** For all subsequent simulations that pass through this now-frozen node, the opponent's move is no longer selected by PUCT. Instead, it is chosen by **sampling proportionally from the frozen visit distribution**.15 This is a form of Thompson Sampling.

This mechanism prevents the engine from focusing all its search effort on refuting a single "best" reply from the opponent. By forcing the search to consider a wider distribution of plausible, but not necessarily optimal, opponent moves, it pushes the game into more complex and varied positions. This is invaluable for generating a rich dataset that covers more of the game's state space, leading to a more robust and rapidly improving neural network.15

### **2.3. Integrating into the SearchWorker**

Implementing Search-Contempt requires modifying both the MCTSNode data structure and the child selection logic within the SearchWorker.

#### **MCTSNode State Augmentation**

The algorithm's logic is stateful and depends on information at each node. Therefore, the MCTSNode class must be extended to track:

* depth: An integer representing the node's depth in the tree relative to the root (root is depth 0). This is needed to distinguish between player and opponent nodes.  
* is\_frozen: A boolean flag, initially False, set to True once the Nscl threshold is crossed for an opponent node.  
* frozen\_visit\_counts: A data structure (e.g., a NumPy array or dictionary) to store the visit counts of the children at the moment of freezing.

#### **Pseudocode for select\_child Method**

The core logic must be embedded within the child selection function of the SearchWorker.

Python

def select\_child(self, node, c\_puct, n\_scl):  
    """  
    Selects a child based on Search-Contempt logic.  
    This method would be part of the SearchWorker's MCTS implementation.  
    """  
    \# Check if it's an opponent node (odd depth)  
    if node.depth % 2\!= 0:  
        \# If the node is already frozen, use Thompson Sampling  
        if node.is\_frozen:  
            \# Sample an action based on the frozen distribution  
            actions \= list(node.frozen\_visit\_counts.keys())  
            counts \= np.array(list(node.frozen\_visit\_counts.values()), dtype=np.float32)  
            probabilities \= counts / counts.sum()  
            chosen\_action \= np.random.choice(actions, p=probabilities)  
            return chosen\_action, node.children\[chosen\_action\]

        \# Check if we should freeze the node now  
        current\_total\_visits \= sum(child.N for child in node.children.values())  
        if current\_total\_visits \> n\_scl:  
            \# Freeze the distribution  
            node.frozen\_visit\_counts \= {action: child.N for action, child in node.children.items()}  
            node.is\_frozen \= True  
            \# Perform the first sample  
            actions \= list(node.frozen\_visit\_counts.keys())  
            counts \= np.array(list(node.frozen\_visit\_counts.values()), dtype=np.float32)  
            probabilities \= counts / counts.sum()  
            chosen\_action \= np.random.choice(actions, p=probabilities)  
            return chosen\_action, node.children\[chosen\_action\]

    \# Default case: Player-to-move node OR unfrozen opponent node. Use standard PUCT.  
    return self.find\_best\_child\_puct(node, c\_puct)

def find\_best\_child\_puct(self, node, c\_puct):  
    """Standard PUCT selection."""  
    best\_score \= \-float('inf')  
    best\_action \= \-1  
    best\_child \= None  
      
    sqrt\_total\_visits \= math.sqrt(node.N)  
    for action, child in node.children.items():  
        score \= child.Q \+ c\_puct \* child.P \* (sqrt\_total\_visits / (1 \+ child.N))  
        if score \> best\_score:  
            best\_score \= score  
            best\_action \= action  
            best\_child \= child  
              
    return best\_action, best\_child

This implementation demonstrates that Search-Contempt requires state to be maintained within the tree itself. The selection logic in the SearchWorker must read this state (depth, is\_frozen) from each node to correctly decide whether to apply the PUCT formula or the Thompson Sampling procedure based on the frozen visit counts.

## **III. Bootstrapping the Network: A "Warm Start" Data Pipeline**

Before initiating the self-play loop, it is highly advantageous to "warm start" the neural network by pre-training it on a large corpus of human expert games. This provides the network with a foundational understanding of chess, significantly accelerating the initial stages of self-play. The Lichess Elite Database is an excellent source for this data.

### **3.1. PGN Parsing and Filtering with python-chess**

The primary challenge with large PGN databases is their size, often many gigabytes. Loading the entire file into memory is not feasible. The python-chess library provides an efficient way to handle this.

* **Iterative Parsing:** The recommended approach is to open the PGN file and read it game by game in a loop using chess.pgn.read\_game(). This function reads from the file handle until it finds a complete game, parses it, and returns a Game object, or None if the end of the file is reached. This method has a minimal memory footprint, as only one game is held in memory at a time.19  
* **Header-Based Filtering:** To ensure high-quality training data, it is essential to filter the games. After reading a game, its metadata can be accessed via the game.headers dictionary-like object.19 Key filtering criteria include:  
  * **Player Strength:** Select games where both players are highly rated. This can be done by checking the WhiteElo and BlackElo tags. A minimum threshold of 2400 for both players is a reasonable starting point.  
  * **Game Termination:** Exclude games that did not end decisively or through normal play. The Termination tag can be checked to filter out games that ended due to Time forfeit or Abandoned. The Result tag (1-0, 0-1, 1/2-1/2) provides the final outcome.19  
* **Training Data Extraction:** For each game that passes the filters, iterate through its mainline moves to generate training samples. For each position s\_t in the game, extract the following triplet:  
  1. **State (s\_t):** The board position, which will be encoded into the NN's input format.  
  2. **Policy Target (π\_t):** The move that was actually played from s\_t. This will be encoded as a one-hot vector across the action space.  
  3. **Value Target (z):** The final outcome of the game from the perspective of the player at s\_t. This is a single scalar: \+1 for a win, \-1 for a loss, and 0 for a draw.

### **3.2. Building an Efficient tf.data Pipeline**

The tf.data API is the standard for building high-performance input pipelines in TensorFlow. The most effective way to integrate the iterative PGN parser is by using a Python generator.

* **The Generator Bridge:** Create a Python generator function that encapsulates the PGN parsing and filtering logic from the previous step. This function will loop through the PGN file and yield a single (encoded\_state, encoded\_policy, value\_target) tuple for each position in each valid game. This approach creates a lazy, memory-efficient data stream.22  
* **The tf.data Pipeline:** The pipeline is constructed by chaining transformations:  
  1. tf.data.Dataset.from\_generator(): This is the key function that wraps the Python generator. It's crucial to provide the output\_signature argument, which is a nest of tf.TensorSpec objects defining the shape and data type of each yielded element. This allows TensorFlow to build an efficient, static graph for the pipeline.22  
  2. .shuffle(buffer\_size): Randomly shuffles the elements. The buffer\_size should be large enough to ensure good randomization (e.g., 100,000).  
  3. .batch(BATCH\_SIZE): Groups the individual training examples into batches for the model.  
  4. .prefetch(tf.data.AUTOTUNE): This is a critical performance optimization. It allows the data pipeline (running on the CPU) to prepare the next batch of data while the current batch is being processed by the model (on the GPU), overlapping the two and minimizing GPU idle time.23

This generator-based pattern is the canonical way to handle large, custom-formatted datasets in TensorFlow, providing both scalability and performance.

### **3.3. The Dual-Head Loss Function**

The model has two outputs (value and policy), so the loss function must combine the errors from both heads. The total loss is a weighted sum, allowing for tuning of the relative importance of each task.14

Ltotal​=α⋅Lvalue​+(1−α)⋅Lpolicy​

* **Value Loss (Lvalue​):** The standard loss for the value head is the Mean Squared Error (MSE) between the network's predicted value v and the ground-truth game outcome z. This is implemented with tf.keras.losses.MeanSquaredError.  
* **Policy Loss (Lpolicy​):** The standard loss for the policy head is the Categorical Cross-Entropy between the network's raw output logits p and the target policy π (a one-hot vector representing the move played). This is implemented with tf.keras.losses.CategoricalCrossentropy(from\_logits=True). The from\_logits=True argument is important for numerical stability.

The following code provides a clean implementation of this combined, weighted loss function within the Keras API.24

Python

import tensorflow as tf

def create\_combined\_loss(alpha=0.5):  
    """  
    Creates a custom loss function that is a weighted sum of value and policy losses.  
    The value 'alpha' weights the value\_loss, and (1-alpha) weights the policy\_loss.  
    """  
    mse \= tf.keras.losses.MeanSquaredError()  
    cce \= tf.keras.losses.CategoricalCrossentropy(from\_logits=True)

    def combined\_loss(y\_true, y\_pred):  
        """  
        Calculates the combined loss.  
        Assumes y\_true and y\_pred are tuples of (value, policy).  
        """  
        \# Unpack the true values and predictions  
        \# y\_true is a tuple of (game\_outcome, move\_policy)  
        \# y\_pred is a tuple of (predicted\_value, policy\_logits)  
        value\_true, policy\_true \= y\_true  
        value\_pred, policy\_pred \= y\_pred

        \# Calculate individual losses  
        value\_loss \= mse(y\_true=value\_true, y\_pred=value\_pred)  
        policy\_loss \= cce(y\_true=policy\_true, y\_pred=policy\_pred)

        \# Apply weighting  
        total\_loss \= alpha \* value\_loss \+ (1 \- alpha) \* policy\_loss  
          
        return total\_loss

    return combined\_loss

\# Example usage with model.compile()  
\# Assume a model with two outputs: 'value\_head' and 'policy\_head'  
\# And a tf.data pipeline yielding a dictionary of labels  
\# model.compile(  
\#     optimizer='adam',  
\#     loss={  
\#         'value\_head': tf.keras.losses.MeanSquaredError(),  
\#         'policy\_head': tf.keras.losses.CategoricalCrossentropy(from\_logits=True)  
\#     },  
\#     loss\_weights={'value\_head': 0.5, 'policy\_head': 0.5} \# This is the Keras-native way to weight losses  
\# )

*Note: A more idiomatic Keras approach for models with named outputs is to provide separate loss functions and use the loss\_weights argument in model.compile(), as shown in the commented example. The custom function approach is more flexible if the true and predicted values are packed into single tensors.*

## **IV. High-Throughput Data Storage for the Replay Buffer**

The self-play phase will generate millions of training examples of the form (state, policy\_vector, game\_outcome). The choice of storage format for this replay buffer is critical, as it directly impacts I/O speed during training, which can easily become a bottleneck.

### **4.1. Format Comparison**

We compare three common formats for storing large numerical datasets in Python.

| Format | Storage Efficiency | I/O Read Speed (for Training) | Ease of Integration with tf.data | Splittability/Parallelism | Best Use Case |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **numpy.savez\_compressed** | Good (compression) | Poor | Low (requires custom loading logic) | No (monolithic file) | Small datasets, archiving, simple persistence. |
| **HDF5 (.h5)** | Excellent (chunking, compression) | Good | Medium (requires h5py and custom generator) | Yes (chunk-based) | Large, structured scientific datasets; array-like data.26 |
| **TFRecord** | Good (protocol buffer) | Excellent | Native (designed for tf.data) | Yes (natively sharded) | High-throughput ML training pipelines in TensorFlow.27 |

* **numpy.savez\_compressed**: While simple for saving NumPy arrays, it creates a single archive file. To read any data, the entire file must be accessed, which is inefficient for sampling from a large replay buffer.  
* **HDF5**: A powerful format for scientific data. It supports hierarchical data, chunking, and compression, which allows for efficient slicing and random access.26 However, it is not a native TensorFlow format and requires a custom data loader (like a Python generator using the  
  h5py library). Furthermore, the underlying HDF5 C library's Python bindings can have concurrency limitations, potentially bottlenecking parallel data loading.26  
* **TFRecord**: A simple binary format for storing a sequence of protocol buffer records.29 Its primary advantage is its seamless and highly optimized integration with the  
  tf.data API. tf.data.TFRecordDataset can read from multiple TFRecord files in parallel, interleave records from them, and prefetch data, creating an extremely high-throughput input pipeline that keeps the GPU fully utilized.27

### **4.2. Recommendation: TFRecord for Optimal Training Performance**

For the specific task of feeding a large replay buffer to a TensorFlow training loop, **TFRecord is the unequivocally recommended format.**

The justification is singular and critical: **performance of the training input pipeline.** While HDF5 is a more feature-rich format in general, TFRecord is purpose-built for exactly this use case. The native integration with tf.data allows for the most efficient parallel data loading, which is essential to prevent the training loop from becoming I/O-bound. In a system where GPU cycles are the most valuable resource, ensuring the GPU is never waiting for data is paramount. TFRecord provides the most direct and optimized path to achieving this goal.27

### **4.3. Implementation Snippet: Writing and Reading a TFRecord**

Here are the core functions for writing a single training example to a TFRecord file and for parsing it back within a tf.data pipeline.

#### **Writing a Training Example**

Python

import tensorflow as tf  
import numpy as np

def \_bytes\_feature(value):  
    """Returns a bytes\_list from a string / byte."""  
    if isinstance(value, type(tf.constant(0))):  
        value \= value.numpy()  
    return tf.train.Feature(bytes\_list=tf.train.BytesList(value=\[value\]))

def \_float\_feature(value):  
    """Returns a float\_list from a float / double."""  
    return tf.train.Feature(float\_list=tf.train.FloatList(value=\[value\]))

def serialize\_example(state\_array, policy\_array, outcome\_scalar):  
    """  
    Creates a tf.train.Example message ready to be written to a file.  
    """  
    feature \= {  
        'state': \_bytes\_feature(state\_array.tobytes()),  
        'policy': \_bytes\_feature(policy\_array.tobytes()),  
        'outcome': \_float\_feature(outcome\_scalar),  
    }  
    example\_proto \= tf.train.Example(features=tf.train.Features(feature=feature))  
    return example\_proto.SerializeToString()

\# \--- Example Usage \---  
\# state \= np.random.rand(8, 8, 119).astype(np.float32)  
\# policy \= np.random.rand(4672).astype(np.float32)  
\# outcome \= 1.0  
\#  
\# with tf.io.TFRecordWriter("training\_data.tfrecord") as writer:  
\#     serialized\_sample \= serialize\_example(state, policy, outcome)  
\#     writer.write(serialized\_sample)

#### **Reading and Parsing within tf.data**

Python

def parse\_tfrecord\_fn(example\_proto, state\_shape, policy\_shape):  
    """  
    Parses a single example proto.  
    """  
    feature\_description \= {  
        'state': tf.io.FixedLenFeature(, tf.string),  
        'policy': tf.io.FixedLenFeature(, tf.string),  
        'outcome': tf.io.FixedLenFeature(, tf.float32),  
    }  
      
    parsed\_features \= tf.io.parse\_single\_example(example\_proto, feature\_description)  
      
    \# Decode the raw byte strings back into tensors  
    state \= tf.io.decode\_raw(parsed\_features\['state'\], tf.float32)  
    state \= tf.reshape(state, state\_shape)  
      
    policy \= tf.io.decode\_raw(parsed\_features\['policy'\], tf.float32)  
    policy \= tf.reshape(policy, policy\_shape)  
      
    outcome \= parsed\_features\['outcome'\]  
      
    \# The model expects a tuple of (inputs, outputs)  
    \# Inputs is the state, outputs are a tuple of (value, policy)  
    return state, (outcome, policy)

\# \--- Example Usage \---  
\# STATE\_SHAPE \= (8, 8, 119\)  
\# POLICY\_SHAPE \= (4672,)  
\#  
\# raw\_dataset \= tf.data.TFRecordDataset(\["training\_data.tfrecord"\])  
\# parsed\_dataset \= raw\_dataset.map(  
\#     lambda x: parse\_tfrecord\_fn(x, STATE\_SHAPE, POLICY\_SHAPE),  
\#     num\_parallel\_calls=tf.data.AUTOTUNE  
\# )  
\#  
\# for state, (outcome, policy) in parsed\_dataset.take(1):  
\#     print("State shape:", state.shape)  
\#     print("Outcome:", outcome.numpy())  
\#     print("Policy shape:", policy.shape)

## **V. Comprehensive Training & Performance Monitoring with TensorBoard**

Effective monitoring is not an afterthought; it is essential for debugging, tuning, and understanding the complex dynamics of the self-play training loop. TensorBoard is the standard visualization toolkit for TensorFlow and provides the necessary tools.31 A comprehensive monitoring setup should track metrics from the model, the self-play process, and the MCTS engine itself.

### **5.1. Essential Metrics and Visualizations**

Metrics should be logged to timestamped directories to allow for easy comparison between different training runs.32

#### **Model-Specific Metrics (Logged during model.fit)**

These metrics are crucial for diagnosing the health of the neural network. They should be logged via the tf.keras.callbacks.TensorBoard callback.

* **Losses (Scalars):** Log total\_loss, policy\_loss, and value\_loss for both the training and validation datasets. The divergence between training and validation loss is the primary indicator of overfitting.34  
* **Weight and Bias Distributions (Histograms):** Configure the TensorBoard callback with histogram\_freq=1. This will log histograms of the weights and biases for every layer of the network at the end of each epoch. This is invaluable for diagnosing issues like vanishing or exploding gradients, or "dying ReLU" neurons where weights are stuck at zero.32  
* **Policy Head Entropy (Custom Scalar):** The entropy of the policy head's output distribution, calculated as H(p)=−∑i​pi​logpi​, is a measure of its "uncertainty". A healthy network should maintain some level of entropy. If the entropy collapses towards zero, it indicates the network is becoming overconfident and is no longer exploring diverse moves, which can harm the MCTS search. This must be logged as a custom scalar using tf.summary.  
* **Value Head Predictions (Histogram):** Log a histogram of the network's value predictions (v) across a validation batch. This visualization quickly reveals if the model is developing a bias (e.g., always predicting values close to 0\) or if it is failing to predict extreme outcomes (-1 or \+1).

#### **Self-Play Statistics (Logged after each generation of games)**

These metrics track the macroscopic behavior of the self-play agents.

* **Average Game Length (Scalar):** A sudden, dramatic change in average game length can signal a significant shift in the agent's playing style or a potential bug in the engine.  
* **Win/Loss/Draw Rates (Scalars):** Track the win, loss, and draw percentages for White and Black separately over training generations. This helps identify any color-based biases the agent might be developing.  
* **Opening Move Distribution (Histogram):** Log a histogram of the first N (e.g., 5\) moves played in the self-play games. This is a critical metric for monitoring exploration. If the distribution collapses to only a few openings, it means the agent is no longer generating diverse data, and the training may stagnate.14

#### **MCTS Engine Performance (Logged per move or averaged per game)**

These metrics monitor the health and throughput of the search engine itself.

* **Nodes Per Second (NPS) (Scalar):** This is the primary throughput metric for the MCTS engine. Log the total NPS across all workers and the average NPS per worker. A drop in NPS can indicate a performance regression or a system bottleneck.  
* **Average Search Depth (Scalar):** Track the average depth reached by the MCTS search for each move. This indicates how far ahead the engine is "thinking".  
* **Root Node Visit Entropy (Scalar):** Similar to policy entropy, the entropy of the MCTS-improved visit counts (π) at the root node measures how much the search explored different moves versus focusing heavily on a single "best" move. A higher entropy suggests a more exploratory search.36

### **5.2. Visualizing True Progress: ELO Rating Over Time**

Loss curves are useful for debugging, but the ultimate measure of progress is playing strength. This must be tracked using the Elo rating system. Plotting Elo over training generations is the single most important graph for assessing the project's success.

This is a derived metric that requires a multi-step process:

1. **Evaluation Protocol:** A dedicated Evaluator process must periodically stage a head-to-head match of a statistically significant number of games (e.g., 400\) between the latest network checkpoint and the current reigning "best" model.14  
2. ELO Calculation: Based on the match result (wins, losses, draws), the ELO difference (ΔElo) between the two models can be calculated using the formula:  
   ΔElo=−400⋅log10​(S1​−1)

   where S is the score (win rate) of the new model against the old. The new model's Elo is then Elo\_old \+ \\Delta Elo.37  
3. **Custom Scalar Logging:** This calculated Elo value is not automatically logged. It must be explicitly written to a TensorBoard log file using the tf.summary API. The step argument should be the current training generation number.  
   Python  
   \# In the Evaluator process after a match  
   summary\_writer \= tf.summary.create\_file\_writer("logs/elo\_over\_time")  
   with summary\_writer.as\_default():  
       tf.summary.scalar("ELO\_Rating", new\_elo, step=generation\_number)

4. **Plotting with Confidence Intervals:** A single Elo number can be noisy. To visualize the statistical significance of an Elo gain, calculate the 95% confidence interval for the win rate S. Then, use the Elo formula to convert the lower and upper bounds of this win rate into lower and upper Elo bounds. Log these as separate scalars (ELO\_lower\_bound, ELO\_upper\_bound). When plotted on the same TensorBoard graph, these three series will create a shaded confidence band around the main Elo rating, making it clear whether an observed improvement is statistically significant or just random noise.

This disciplined process of evaluation, calculation, and custom logging is the only way to generate the authoritative plot of playing strength versus training time, which is the ultimate benchmark of the entire system's progress.

#### **Cytowane prace**

1. multiprocessing — Process-based parallelism — Python 3.13.5 documentation, otwierano: czerwca 13, 2025, [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)  
2. Python's multiprocessing performance problem, otwierano: czerwca 13, 2025, [https://pythonspeed.com/articles/faster-multiprocessing-pickle/](https://pythonspeed.com/articles/faster-multiprocessing-pickle/)  
3. Python Multiprocessing: A Guide to Threads and Processes \- DataCamp, otwierano: czerwca 13, 2025, [https://www.datacamp.com/tutorial/python-multiprocessing-tutorial](https://www.datacamp.com/tutorial/python-multiprocessing-tutorial)  
4. Why are Multiprocessing Queues Slow when Sharing Large Objects in Python? \- Mindee, otwierano: czerwca 13, 2025, [https://www.mindee.com/blog/why-are-multiprocessing-queues-slow-when-sharing-large-objects-in-python](https://www.mindee.com/blog/why-are-multiprocessing-queues-slow-when-sharing-large-objects-in-python)  
5. multiprocessing.shared\_memory — Shared memory for direct access across processes — Python 3.13.5 documentation, otwierano: czerwca 13, 2025, [https://docs.python.org/3/library/multiprocessing.shared\_memory.html](https://docs.python.org/3/library/multiprocessing.shared_memory.html)  
6. Multiprocessing with Queue.queue in Python for numpy arrays \- Stack Overflow, otwierano: czerwca 13, 2025, [https://stackoverflow.com/questions/73267885/multiprocessing-with-queue-queue-in-python-for-numpy-arrays](https://stackoverflow.com/questions/73267885/multiprocessing-with-queue-queue-in-python-for-numpy-arrays)  
7. How to use read-only, shared memory (as NumPy arrays) in multiprocessing, otwierano: czerwca 13, 2025, [https://stackoverflow.com/questions/75357968/how-to-use-read-only-shared-memory-as-numpy-arrays-in-multiprocessing](https://stackoverflow.com/questions/75357968/how-to-use-read-only-shared-memory-as-numpy-arrays-in-multiprocessing)  
8. Demo of multiprocessing with shared numpy arrays \- GitHub Gist, otwierano: czerwca 13, 2025, [https://gist.github.com/dkirkby/6f5e07f6bc950b1c42739b54e8b3d046](https://gist.github.com/dkirkby/6f5e07f6bc950b1c42739b54e8b3d046)  
9. Python multiprocessing Queue vs Pipe vs SharedMemory \- Stack Overflow, otwierano: czerwca 13, 2025, [https://stackoverflow.com/questions/71331820/python-multiprocessing-queue-vs-pipe-vs-sharedmemory](https://stackoverflow.com/questions/71331820/python-multiprocessing-queue-vs-pipe-vs-sharedmemory)  
10. dillonalaird/shared\_numpy: A simple library for creating shared memory numpy arrays \- GitHub, otwierano: czerwca 13, 2025, [https://github.com/dillonalaird/shared\_numpy](https://github.com/dillonalaird/shared_numpy)  
11. Python multiprocessing.Queue vs multiprocessing.manager().Queue() \- GeeksforGeeks, otwierano: czerwca 13, 2025, [https://www.geeksforgeeks.org/python-multiprocessing-queue-vs-multiprocessing-manager-queue/](https://www.geeksforgeeks.org/python-multiprocessing-queue-vs-multiprocessing-manager-queue/)  
12. Shared Memory in python \- Omid Sadeghnezhad, otwierano: czerwca 13, 2025, [https://sadeghnezhad.me/blog/2024/shared-memory-python/](https://sadeghnezhad.me/blog/2024/shared-memory-python/)  
13. How to overcome overhead in python multiprocessing? \- Stack Overflow, otwierano: czerwca 13, 2025, [https://stackoverflow.com/questions/68892839/how-to-overcome-overhead-in-python-multiprocessing](https://stackoverflow.com/questions/68892839/how-to-overcome-overhead-in-python-multiprocessing)  
14. Asynchronous MCTS Implementation Plan  
15. Search-contempt: a hybrid MCTS algorithm for training AlphaZero ..., otwierano: czerwca 13, 2025, [https://arxiv.org/pdf/2504.07757?](https://arxiv.org/pdf/2504.07757)  
16. Search-contempt: a hybrid MCTS algorithm for training AlphaZero-like engines with better computational efficiency \- arXiv, otwierano: czerwca 13, 2025, [https://arxiv.org/html/2504.07757v1](https://arxiv.org/html/2504.07757v1)  
17. \[Literature Review\] Search-contempt: a hybrid MCTS algorithm for training AlphaZero-like engines with better computational efficiency \- Moonlight | AI Colleague for Research Papers, otwierano: czerwca 13, 2025, [https://www.themoonlight.io/en/review/search-contempt-a-hybrid-mcts-algorithm-for-training-alphazero-like-engines-with-better-computational-efficiency](https://www.themoonlight.io/en/review/search-contempt-a-hybrid-mcts-algorithm-for-training-alphazero-like-engines-with-better-computational-efficiency)  
18. Search-contempt: a hybrid MCTS algorithm for training AlphaZero-like engines with better computational efficiency \- Powerdrill AI, otwierano: czerwca 13, 2025, [https://powerdrill.ai/discover/summary-search-contempt-a-hybrid-mcts-algorithm-for-cm9d9uap4dcsk07ravq62cesc](https://powerdrill.ai/discover/summary-search-contempt-a-hybrid-mcts-algorithm-for-cm9d9uap4dcsk07ravq62cesc)  
19. PGN parsing and writing — python-chess 1.11.2 documentation, otwierano: czerwca 13, 2025, [https://python-chess.readthedocs.io/en/latest/pgn.html](https://python-chess.readthedocs.io/en/latest/pgn.html)  
20. Using Python-Chess with Pandas for High-Volume PGN Parsing \- Ryan Wingate, otwierano: czerwca 13, 2025, [https://ryanwingate.com/other-interests/chess/using-python-chess-with-pandas-for-high-volume-pgn-parsing/](https://ryanwingate.com/other-interests/chess/using-python-chess-with-pandas-for-high-volume-pgn-parsing/)  
21. clinaresl/pgnparser: This tool parses PGN files, shows a summary review of its contents, sort and filter games by any criteria, and produce histograms on any piece of data. In addition, it also generates LaTeX files that can be processed to generate pdf files showing the contents of the games in any PGN file \- GitHub, otwierano: czerwca 13, 2025, [https://github.com/clinaresl/pgnparser](https://github.com/clinaresl/pgnparser)  
22. tf.data: Build TensorFlow input pipelines, otwierano: czerwca 13, 2025, [https://www.tensorflow.org/guide/data](https://www.tensorflow.org/guide/data)  
23. Data pipelines with tf.data and TensorFlow \- PyImageSearch, otwierano: czerwca 13, 2025, [https://pyimagesearch.com/2021/06/21/data-pipelines-with-tf-data-and-tensorflow/](https://pyimagesearch.com/2021/06/21/data-pipelines-with-tf-data-and-tensorflow/)  
24. Creating Custom Loss Functions in TensorFlow: Understanding the Theory and Practicalities | Towards Data Science, otwierano: czerwca 13, 2025, [https://towardsdatascience.com/creating-custom-loss-functions-in-tensorflow-understanding-the-theory-and-practicalities-383a19e387d6/](https://towardsdatascience.com/creating-custom-loss-functions-in-tensorflow-understanding-the-theory-and-practicalities-383a19e387d6/)  
25. How to design a custom loss function to add two loss \- Google AI Developers Forum, otwierano: czerwca 13, 2025, [https://discuss.ai.google.dev/t/how-to-design-a-custom-loss-function-to-add-two-loss/32208](https://discuss.ai.google.dev/t/how-to-design-a-custom-loss-function-to-add-two-loss/32208)  
26. What Comes After HDF5? Seeking a Data Storage Format for Deep Learning \- KDnuggets, otwierano: czerwca 13, 2025, [https://www.kdnuggets.com/2021/11/after-hdf5-data-storage-format-deep-learning.html](https://www.kdnuggets.com/2021/11/after-hdf5-data-storage-format-deep-learning.html)  
27. Best Practice for Data Formats in Deep Learning \- SURF User Knowledge Base, otwierano: czerwca 13, 2025, [https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/56295498/Best+Practice+for+Data+Formats+in+Deep+Learning](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/56295498/Best+Practice+for+Data+Formats+in+Deep+Learning)  
28. Guide to File Formats for Machine Learning \- Hopsworks, otwierano: czerwca 13, 2025, [https://www.hopsworks.ai/post/guide-to-file-formats-for-machine-learning](https://www.hopsworks.ai/post/guide-to-file-formats-for-machine-learning)  
29. TFRecords: Reading and Writing \- Slideflow Documentation, otwierano: czerwca 13, 2025, [https://slideflow.dev/tfrecords/](https://slideflow.dev/tfrecords/)  
30. How to inspect a Tensorflow .tfrecord file? \- GeeksforGeeks, otwierano: czerwca 13, 2025, [https://www.geeksforgeeks.org/how-to-inspect-a-tensorflow-tfrecord-file/](https://www.geeksforgeeks.org/how-to-inspect-a-tensorflow-tfrecord-file/)  
31. TensorBoard \- TensorFlow, otwierano: czerwca 13, 2025, [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)  
32. Deep Dive Into TensorBoard: Tutorial With Examples \- Neptune.ai, otwierano: czerwca 13, 2025, [https://neptune.ai/blog/tensorboard-tutorial](https://neptune.ai/blog/tensorboard-tutorial)  
33. The complete guide to ML model visualization with Tensorboard | Intel® Tiber™ AI Studio, otwierano: czerwca 13, 2025, [https://cnvrg.io/tensorboard-guide/](https://cnvrg.io/tensorboard-guide/)  
34. TensorBoard Scalars: Logging training metrics in Keras \- TensorFlow, otwierano: czerwca 13, 2025, [https://www.tensorflow.org/tensorboard/scalars\_and\_keras](https://www.tensorflow.org/tensorboard/scalars_and_keras)  
35. Get started with TensorBoard \- TensorFlow, otwierano: czerwca 13, 2025, [https://www.tensorflow.org/tensorboard/get\_started](https://www.tensorflow.org/tensorboard/get_started)  
36. LightZero's Logging and Monitoring System \- GitHub Pages, otwierano: czerwca 13, 2025, [https://opendilab.github.io/LightZero/tutorials/logs/logs.html](https://opendilab.github.io/LightZero/tutorials/logs/logs.html)  
37. Developing an Elo Based, Data-Driven Rating System for 2v2 Multiplayer Games, otwierano: czerwca 13, 2025, [https://towardsdatascience.com/developing-an-elo-based-data-driven-ranking-system-for-2v2-multiplayer-games-7689f7d42a53/](https://towardsdatascience.com/developing-an-elo-based-data-driven-ranking-system-for-2v2-multiplayer-games-7689f7d42a53/)