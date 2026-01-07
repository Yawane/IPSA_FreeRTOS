import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Configuration Constants
COMPILER = "g++"
FLAGS = ["-std=c++17", "-O2"]
TIMEOUT_SEC = 5
SAFETY_FACTOR = 1.2
DEADLINE_THRESHOLD = 0.50  # 50%
MAX_UTILIZATION = 0.69     # Liu & Layland bound for infinite tasks (approx)


@dataclass
class TaskStats:
    """Data container for task statistics."""
    mean: float
    median: float
    q1: float
    q3: float
    min_val: float
    max_val: float
    std_dev: float
    wcet: float
    wcet_safe: float


class TaskAnalyzer:
    """Handles execution, timing measurement, and statistical analysis of a binary."""
    
    def __init__(self, name: str, source_file: str):
        self.name = name
        self.source = Path(f"{source_file}.cpp") # .cpp file
        self.executable = Path(source_file)      # .exe executable file
        self.timing_samples: np.ndarray = np.array([]) # execution timings of task


    def compile(self) -> bool: # compile the .cpp file of the current TaskAnalyser
        """Compiles the C++ source code."""
        cmd = [COMPILER] + FLAGS + [str(self.source), "-o", str(self.executable)] # compilation command line
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True) # execute the command line 'cmd'
            print(f"[OK] {self.name} build successful.")
            return True
        
        except subprocess.CalledProcessError as e: # shows errors if 'try' produced errors
            print(f"[ERR] {self.name} build failed: {e.stderr}")
            return False


    def execute_iterations(self, iterations: int = 100_000) -> None:
        """Runs the executable multiple times to collect execution samples."""
        print(f"\n>> Benchmarking {self.name} ({iterations} cycles)...")
        raw_times = [] # temporary list containing row execution times
        
        # Using a relative path context for the subprocess
        exec_path = f"./{self.executable.name}"
        
        for _ in tqdm(range(iterations), mininterval=2.0, desc=self.name): # loop over iterations, and show progress bar every 2 seconds.
            try:
                out = subprocess.run([exec_path], capture_output=True, text=True, timeout=TIMEOUT_SEC) # run the executable file, and store output in 'out'

                # last line (last cout of c++ code) contains the measured execution time.
                output_lines = out.stdout.strip().splitlines()
                if output_lines: # take lase element (measured execution time) only if there is an output
                    raw_times.append(int(output_lines[-1]))

            except (ValueError, subprocess.TimeoutExpired, IndexError) as e: # shows errors if execution of task produced errors in 'try'
                print(f"Runtime warning on {self.name}: {e}")
                continue

        self.timing_samples = np.array(raw_times) # store the execution times in the attribute of the current TaskAnalyser
        print(f">> Completed {self.name}. Samples: {len(self.timing_samples)}")


    @property
    def statistics(self) -> TaskStats:
        """Computes and returns a structured statistics object."""
        if self.timing_samples.size == 0: # in case there is no timing info
            raise ValueError("No data collected yet.")
            
        wcet = float(np.max(self.timing_samples)) # maximum measured execution time is WCET

        # return a dataclass as defined earlier
        return TaskStats(
            mean=float(np.mean(self.timing_samples)), # using numpy basic statistical functions
            median=float(np.median(self.timing_samples)),
            q1=float(np.percentile(self.timing_samples, 25)),
            q3=float(np.percentile(self.timing_samples, 75)),
            min_val=float(np.min(self.timing_samples)),
            max_val=wcet,
            std_dev=float(np.std(self.timing_samples)),
            wcet=wcet,
            wcet_safe=wcet * SAFETY_FACTOR
        )


    def compute_miss_ratio(self, threshold_factor: float) -> Tuple[float, float]:
        """Calculates probability of exceeding a specific execution time threshold."""
        wcet = np.max(self.timing_samples)
        deadline_limit = wcet * threshold_factor # compute new deadline limit
        miss_count = np.sum(self.timing_samples > deadline_limit) # compute number of missed deadline
        return (miss_count / self.timing_samples.size), deadline_limit # return ratio and deadline_limit


    def export_plot(self, output_dir: str = ".") -> None:
        """Generates a histogram of the execution distribution."""
        stats = self.statistics # take statistics
        # plt.style.use('bmh') # Scientific style
        plt.figure(figsize=(5, 4))
        
        
        data_µs = self.timing_samples / 1000.0 # Plotting in Microseconds for readability
  
        start = np.log10(data_µs.min()) if data_µs.min() > 0 else 0
        stop = np.log10(data_µs.max())

        log_bins = np.logspace(start, stop, num=80)
        
        plt.hist(data_µs, bins= log_bins, color="#416283", alpha=0.8, rwidth=0.9)
        plt.yscale('log')
        plt.xscale('log')

        plt.axvline(stats.mean/1000, color='red', linestyle='--', label='Mean')     # show mean with vertical line
        plt.axvline(stats.wcet/1000, color='orange', linestyle='--', label='WCET')  # show WCET with vertical line
        
        plt.xlabel('Execution Time (µs)')
        plt.ylabel('Frequency (log Count)')
        plt.title(f'Distribution Analysis: {self.name}')
        plt.legend(loc='best')
        plt.grid(True, which="major", color="#666666", linestyle="-", alpha=0.3)
        plt.grid(True, which="minor", color="#999999", linestyle="-", alpha=0.1)

        plt.tight_layout()
        
        # Annotation box
        info_text = (f"Mean: {stats.mean/1000:.2f} µs\n"
                     f"Median: {stats.median/1000:.2f} µs\n"
                     f"WCET (Safe): {stats.wcet_safe/1000:.2f} µs")
        
        plt.gca().text(0.95, 0.95, info_text, transform=plt.gca().transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        filename = Path(output_dir) / f"{self.name}_dist.png"
        plt.savefig(filename, dpi=150)
        plt.close()


class SchedulabilityAnalyzer:
    """Performs RMS and Response Time Analysis (RTA)."""
    
    @staticmethod
    def derive_system_periods() -> Dict[str, float]:
        """Defines the temporal constraints of the system (in ms)."""
        # Baseline constraint logic
        base_t4 = 450
        return {
            'Task1':100,               # most important task
            'Task2': base_t4 / 4.0,    # 4 times frequency of Task4
            'Task3': base_t4 / 3.0,    # 3 times frequency of Task4
            'Task4': base_t4,          # Baseline
            'Task5': 200.0             # fixed
        }

    @staticmethod
    def assign_priorities_rms(periods: Dict[str, float]) -> Dict[str, int]:
        """Rate Monotonic Scheduling: Shorter period = Higher Priority."""
        # Sort by period (ascending)
        sorted_tasks = sorted(periods.items(), key=lambda item: item[1])
        
        # Assign priorities: Higher integer ==> Higher priority
        priority_map = {name: len(sorted_tasks) - i for i, (name, _) in enumerate(sorted_tasks)}
        
        # Constraint Override: Task1 is critical
        max_prio = max(priority_map.values())
        priority_map['Task1'] = max_prio + 1
        return priority_map

    @staticmethod
    def check_utilization(periods: Dict[str, float], wcets_ns: Dict[str, float]) -> bool:
        """Checks if Total Utilization <= Liu & Layland bound."""
        U_total = 0.0
        print("\n=== UTILIZATION ANALYSIS ===")
        for name, T in periods.items():
            C = wcets_ns[name] / 1e6  # ns to ms
            U = C / T
            U_total += U
            print(f"  {name}: {U*100:6.2f}% (C={C:.3f}ms, T={T:.1f}ms)")
            
        print(f"  TOTAL: {U_total*100:6.2f}% / Max: {MAX_UTILIZATION*100:.0f}%")
        return U_total <= MAX_UTILIZATION

    def check_preemptive_rta(self, periods: Dict[str, float], wcets_ns: Dict[str, float], priorities: Dict[str, int]) -> bool:
        """Calculates Response Time Analysis (RTA) for Preemptive scheduling."""
        print("\n=== RTA (PREEMPTIVE) ===")
        system_schedulable = True
        
        for name, T in periods.items():
            C = wcets_ns[name] / 1e6 # ms
            R = C # Initial response time guess
            
            while True:
                interference = 0.0
                for hp_task, hp_prio in priorities.items():
                    if hp_prio > priorities[name]:
                        C_hp = wcets_ns[hp_task] / 1e6
                        T_hp = periods[hp_task]
                        interference += np.ceil(R / T_hp) * C_hp
                
                R_new = C + interference
                if R_new > T:
                    print(f"  [FAIL] {name}: R={R_new:.4f}ms > T={T}ms")
                    system_schedulable = False
                    break
                if R_new == R:
                    print(f"  [PASS] {name}: R={R:.4f}ms <= T={T}ms")
                    break
                R = R_new
                
        return system_schedulable

    def check_non_preemptive_rta(self, periods: Dict[str, float], 
                                 wcets_ns: Dict[str, float], 
                                 priorities: Dict[str, int]) -> bool:
        """Calculates RTA for Non-Preemptive scheduling (includes Blocking)."""
        print("\n=== RTA (NON-PREEMPTIVE) ===")
        system_schedulable = True
        
        # Max blocking time is the longest C of any lower priority task.
        # Since it's non-preemptive, effectively max(C_all) is a safe upper bound for B_i
        max_C_global = max(wcets_ns.values()) / 1e6
        
        for name, T in periods.items():
            C = wcets_ns[name] / 1e6
            B = max_C_global # Blocking term
            R = C + B
            
            # Interference logic
            interference = 0.0
            for hp_task, hp_prio in priorities.items():
                if hp_prio > priorities[name]:
                    C_hp = wcets_ns[hp_task] / 1e6
                    T_hp = periods[hp_task]
                    
                    interference += np.ceil(R / T_hp) * C_hp

            R_total = R + interference
            
            is_ok = R_total <= T
            system_schedulable = system_schedulable and is_ok
            print(f"  [{'PASS' if is_ok else 'FAIL'}] {name}: R={R_total:.4f}ms (B={B:.3f}ms)")
            
        return system_schedulable


def main():
    # 1. Setup Tasks
    task_registry = [
        TaskAnalyzer('Task1', 'task1'),
        TaskAnalyzer('Task2', 'task2'),
        TaskAnalyzer('Task3', 'task3'),
        TaskAnalyzer('Task4', 'task4'),
        TaskAnalyzer('Task5', 'task5'),
    ]

    # 2. Compilation
    print("=== BUILD PHASE ===")
    for task in task_registry:
        if not task.compile():
            print("Aborting due to build errors.")
            return

    # 3. Execution
    print("\n=== BENCHMARK PHASE ===")
    wcet_map = {}
    
    for task in task_registry:
        task.execute_iterations(iterations=100000)
        stats = task.statistics
        wcet_map[task.name] = stats.wcet_safe
        
        # Summary
        print(f"  -> Stats for {task.name}:")
        print(f"     Mean: {stats.mean:.2f} ns | WCET(1.2x): {stats.wcet_safe:.2f} ns")
        
        # Deadline Analysis
        prob, deadline_val = task.compute_miss_ratio(DEADLINE_THRESHOLD)
        print(f"     Deadline Risk ({DEADLINE_THRESHOLD*100}% WCET): {prob*100:.4f}%")
        
        task.export_plot()

    # 4. Scheduling Analysis
    analyzer = SchedulabilityAnalyzer()
    periods = analyzer.derive_system_periods()
    priorities = analyzer.assign_priorities_rms(periods)

    print("\n=== SYSTEM CONFIGURATION ===")
    for t in periods:
        print(f"{t}: Period={periods[t]}ms, Priority={priorities[t]}")

    # Run Checks
    utilization_ok = analyzer.check_utilization(periods, wcet_map)
    preemptive_ok = analyzer.check_preemptive_rta(periods, wcet_map, priorities)
    non_preemptive_ok = analyzer.check_non_preemptive_rta(periods, wcet_map, priorities)

    # Final Report
    print("\n" + "="*30)
    print("       FINAL VERDICT")
    print("="*30)
    print(f"Utilization Check:      {'[PASS]' if utilization_ok else '[FAIL]'}")
    print(f"Preemptive Sched:       {'[PASS]' if preemptive_ok else '[FAIL]'}")
    print(f"Non-Preemptive Sched:   {'[PASS]' if non_preemptive_ok else '[FAIL]'}")
    print("="*30)

if __name__ == "__main__":
    main()
