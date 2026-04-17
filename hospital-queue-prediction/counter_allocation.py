"""
Counter Allocation Optimization Engine
Provides intelligent recommendations for counter/doctor allocation across departments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from config import allocation_config, department_config

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Result of counter allocation optimization"""
    department: str
    current_counters: int
    recommended_counters: int
    predicted_wait_time: float
    predicted_queue_length: int
    utilization: float
    priority: float
    reasoning: str
    alert_level: str  # 'normal', 'warning', 'critical'


@dataclass
class DepartmentState:
    """Current state of a department"""
    name: str
    queue_length: int
    current_counters: int
    predicted_wait_time: float
    arrival_rate: float
    service_rate: float
    utilization: float


class CounterAllocator:
    """
    Intelligent counter allocation optimization engine
    Supports multiple optimization strategies
    """
    
    def __init__(self):
        self.allocation_history = []
        self.last_allocation_time = None
        
    def calculate_optimal_counters(self, 
                                   predicted_wait_time: float,
                                   queue_length: int,
                                   arrival_rate: float,
                                   department: str,
                                   current_counters: int = None) -> int:
        """
        Calculate optimal number of counters for a department
        
        Args:
            predicted_wait_time: Predicted waiting time in minutes
            queue_length: Current queue length
            arrival_rate: Patient arrival rate (patients/hour)
            department: Department name
            current_counters: Current number of counters
            
        Returns:
            Recommended number of counters
        """
        # Get department configuration
        min_counters = department_config.MIN_COUNTERS.get(department, 1)
        max_counters = department_config.MAX_COUNTERS.get(department, 5)
        avg_service_time = department_config.AVG_SERVICE_TIMES.get(department, 8.0)
        
        # Calculate service rate (patients per hour per counter)
        service_rate_per_counter = 60 / avg_service_time
        
        # Method 1: Based on queue length and target wait time
        target_wait = allocation_config.IDEAL_WAIT_TIME
        
        if predicted_wait_time <= target_wait:
            # Current allocation is acceptable
            counters_needed = current_counters if current_counters else min_counters
        else:
            # Calculate counters needed to achieve target wait time
            # Using Little's Law: L = λW, where L=queue, λ=arrival rate, W=wait time
            # Required service rate = arrival_rate + (queue_length / target_wait_hours)
            target_wait_hours = target_wait / 60
            required_service_rate = arrival_rate + (queue_length / target_wait_hours)
            counters_needed = int(np.ceil(required_service_rate / service_rate_per_counter))
        
        # Method 2: Based on utilization
        if current_counters and current_counters > 0:
            current_capacity = current_counters * service_rate_per_counter
            utilization = arrival_rate / current_capacity if current_capacity > 0 else 1.0
            
            # Adjust if utilization is too high or too low
            if utilization > allocation_config.MAX_UTILIZATION:
                adjustment_factor = utilization / allocation_config.MAX_UTILIZATION
                counters_needed = max(counters_needed, int(np.ceil(current_counters * adjustment_factor)))
            elif utilization < allocation_config.MIN_UTILIZATION and current_counters > min_counters:
                # Can reduce counters if utilization is low
                counters_needed = min(counters_needed, max(min_counters, current_counters - 1))
        
        # Ensure within bounds
        counters_needed = max(min_counters, min(max_counters, counters_needed))
        
        return counters_needed
    
    def greedy_allocation(self, departments: List[DepartmentState], 
                         total_staff: int) -> Dict[str, AllocationResult]:
        """
        Greedy algorithm for counter allocation
        Prioritizes departments with highest wait times and priority weights
        
        Args:
            departments: List of department states
            total_staff: Total staff available
            
        Returns:
            Dictionary of allocation results by department
        """
        logger.info("Running greedy allocation algorithm")
        
        results = {}
        remaining_staff = total_staff
        
        # First pass: Allocate minimum required counters
        for dept in departments:
            min_counters = department_config.MIN_COUNTERS.get(dept.name, 1)
            results[dept.name] = AllocationResult(
                department=dept.name,
                current_counters=dept.current_counters,
                recommended_counters=min_counters,
                predicted_wait_time=dept.predicted_wait_time,
                predicted_queue_length=dept.queue_length,
                utilization=dept.utilization,
                priority=department_config.PRIORITY_WEIGHTS.get(dept.name, 1.0),
                reasoning="Minimum allocation",
                alert_level=self._get_alert_level(dept.predicted_wait_time, dept.queue_length)
            )
            remaining_staff -= min_counters
        
        # Second pass: Allocate remaining staff based on priority and need
        while remaining_staff > 0:
            # Calculate need score for each department
            max_score = -1
            best_dept = None
            
            for dept in departments:
                current_allocation = results[dept.name].recommended_counters
                max_counters = department_config.MAX_COUNTERS.get(dept.name, 5)
                
                if current_allocation >= max_counters:
                    continue  # Already at maximum
                
                # Score based on wait time, priority, and queue length
                priority = department_config.PRIORITY_WEIGHTS.get(dept.name, 1.0)
                wait_time_score = dept.predicted_wait_time / allocation_config.IDEAL_WAIT_TIME
                queue_score = dept.queue_length / 10  # Normalized
                utilization_score = dept.utilization
                
                score = (wait_time_score * 0.4 + queue_score * 0.3 + utilization_score * 0.3) * priority
                
                if score > max_score:
                    max_score = score
                    best_dept = dept.name
            
            if best_dept is None:
                break  # No more departments need additional counters
            
            # Allocate one more counter to best department
            results[best_dept].recommended_counters += 1
            results[best_dept].reasoning = "Additional allocation based on workload"
            remaining_staff -= 1
        
        # Update reasoning for final allocations
        for dept_name, result in results.items():
            change = result.recommended_counters - result.current_counters
            if change > 0:
                result.reasoning = f"Increase by {change} counter(s) to reduce wait time"
            elif change < 0:
                result.reasoning = f"Decrease by {abs(change)} counter(s) due to low utilization"
            else:
                result.reasoning = "Maintain current allocation"
        
        logger.info(f"Greedy allocation complete. Used {total_staff - remaining_staff}/{total_staff} staff")
        
        return results
    
    def linear_programming_allocation(self, departments: List[DepartmentState],
                                     total_staff: int) -> Dict[str, AllocationResult]:
        """
        Linear programming approach for optimal allocation
        Minimizes total weighted wait time across all departments
        
        Args:
            departments: List of department states
            total_staff: Total staff available
            
        Returns:
            Dictionary of allocation results by department
        """
        try:
            from scipy.optimize import linprog
            
            logger.info("Running linear programming allocation")
            
            # Objective: minimize weighted wait times
            # Variables: x_i = number of counters for department i
            
            n_depts = len(departments)
            
            # Coefficients for objective function (weights * predicted impact)
            c = []
            for dept in departments:
                priority = department_config.PRIORITY_WEIGHTS.get(dept.name, 1.0)
                # Inverse relationship: more counters = less wait time
                c.append(-priority)  # Negative because we want to maximize service
            
            # Constraint 1: Total staff limit
            A_eq = [[1] * n_depts]
            b_eq = [total_staff]
            
            # Bounds: min and max counters per department
            bounds = []
            for dept in departments:
                min_c = department_config.MIN_COUNTERS.get(dept.name, 1)
                max_c = department_config.MAX_COUNTERS.get(dept.name, 5)
                bounds.append((min_c, max_c))
            
            # Solve
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                allocations = np.round(result.x).astype(int)
                
                # Create results
                results = {}
                for i, dept in enumerate(departments):
                    results[dept.name] = AllocationResult(
                        department=dept.name,
                        current_counters=dept.current_counters,
                        recommended_counters=int(allocations[i]),
                        predicted_wait_time=dept.predicted_wait_time,
                        predicted_queue_length=dept.queue_length,
                        utilization=dept.utilization,
                        priority=department_config.PRIORITY_WEIGHTS.get(dept.name, 1.0),
                        reasoning="Optimal LP allocation",
                        alert_level=self._get_alert_level(dept.predicted_wait_time, dept.queue_length)
                    )
                
                logger.info("Linear programming allocation successful")
                return results
            else:
                logger.warning("LP failed, falling back to greedy")
                return self.greedy_allocation(departments, total_staff)
                
        except ImportError:
            logger.warning("scipy not available, using greedy allocation")
            return self.greedy_allocation(departments, total_staff)
        except Exception as e:
            logger.error(f"LP allocation error: {e}, falling back to greedy")
            return self.greedy_allocation(departments, total_staff)
    
    def hybrid_allocation(self, departments: List[DepartmentState],
                         total_staff: int) -> Dict[str, AllocationResult]:
        """
        Hybrid approach combining greedy and optimization
        Uses greedy for speed with optimization for critical cases
        
        Args:
            departments: List of department states
            total_staff: Total staff available
            
        Returns:
            Dictionary of allocation results by department
        """
        # Check if any department is in critical state
        critical_depts = [
            d for d in departments 
            if d.predicted_wait_time > allocation_config.CRITICAL_WAIT_TIME or
               d.queue_length > allocation_config.QUEUE_LENGTH_ALERT
        ]
        
        if critical_depts:
            logger.info(f"Critical state detected in {len(critical_depts)} departments, using LP")
            return self.linear_programming_allocation(departments, total_staff)
        else:
            logger.info("Normal state, using greedy allocation")
            return self.greedy_allocation(departments, total_staff)
    
    def recommend_allocation(self, department_predictions: Dict[str, Dict],
                           current_allocations: Dict[str, int],
                           total_staff: int = None) -> Dict[str, AllocationResult]:
        """
        Main method to get allocation recommendations
        
        Args:
            department_predictions: Dict with predictions for each department
                {dept_name: {'wait_time': X, 'queue_length': Y, 'arrival_rate': Z, ...}}
            current_allocations: Current counter allocation by department
            total_staff: Total staff available (uses config default if None)
            
        Returns:
            Dictionary of allocation recommendations
        """
        if total_staff is None:
            total_staff = allocation_config.TOTAL_STAFF_AVAILABLE
        
        # Check cooldown period
        if self._should_skip_reallocation():
            logger.info("Skipping reallocation due to cooldown period")
            return self._get_current_state_results(department_predictions, current_allocations)
        
        # Build department states
        departments = []
        for dept_name, predictions in department_predictions.items():
            dept_state = DepartmentState(
                name=dept_name,
                queue_length=predictions.get('queue_length', 0),
                current_counters=current_allocations.get(dept_name, department_config.MIN_COUNTERS.get(dept_name, 2)),
                predicted_wait_time=predictions.get('wait_time', 0),
                arrival_rate=predictions.get('arrival_rate', 1.0),
                service_rate=60 / department_config.AVG_SERVICE_TIMES.get(dept_name, 8.0),
                utilization=predictions.get('utilization', 0.5)
            )
            departments.append(dept_state)
        
        # Select allocation method based on configuration
        method = allocation_config.OPTIMIZATION_METHOD
        
        if method == 'greedy':
            results = self.greedy_allocation(departments, total_staff)
        elif method == 'linear_programming':
            results = self.linear_programming_allocation(departments, total_staff)
        elif method == 'hybrid':
            results = self.hybrid_allocation(departments, total_staff)
        else:
            logger.warning(f"Unknown method {method}, using greedy")
            results = self.greedy_allocation(departments, total_staff)
        
        # Update allocation history
        self.last_allocation_time = datetime.now()
        self.allocation_history.append({
            'timestamp': self.last_allocation_time,
            'results': results
        })
        
        return results
    
    def _should_skip_reallocation(self) -> bool:
        """Check if reallocation should be skipped due to cooldown"""
        if self.last_allocation_time is None:
            return False
        
        cooldown = timedelta(minutes=allocation_config.REALLOCATION_COOLDOWN)
        return datetime.now() - self.last_allocation_time < cooldown
    
    def _get_current_state_results(self, department_predictions: Dict, 
                                   current_allocations: Dict) -> Dict[str, AllocationResult]:
        """Return current state without reallocation"""
        results = {}
        for dept_name, predictions in department_predictions.items():
            current = current_allocations.get(dept_name, 2)
            results[dept_name] = AllocationResult(
                department=dept_name,
                current_counters=current,
                recommended_counters=current,
                predicted_wait_time=predictions.get('wait_time', 0),
                predicted_queue_length=predictions.get('queue_length', 0),
                utilization=predictions.get('utilization', 0.5),
                priority=department_config.PRIORITY_WEIGHTS.get(dept_name, 1.0),
                reasoning="Cooldown period active",
                alert_level=self._get_alert_level(predictions.get('wait_time', 0), 
                                                  predictions.get('queue_length', 0))
            )
        return results
    
    def _get_alert_level(self, wait_time: float, queue_length: int) -> str:
        """Determine alert level based on wait time and queue length"""
        if (wait_time > allocation_config.CRITICAL_WAIT_TIME or 
            queue_length > allocation_config.QUEUE_LENGTH_ALERT * 1.5):
            return 'critical'
        elif (wait_time > allocation_config.ACCEPTABLE_WAIT_TIME or 
              queue_length > allocation_config.QUEUE_LENGTH_ALERT):
            return 'warning'
        else:
            return 'normal'
    
    def get_allocation_summary(self, results: Dict[str, AllocationResult]) -> Dict:
        """
        Generate summary statistics for allocation results
        
        Args:
            results: Dictionary of allocation results
            
        Returns:
            Summary dictionary
        """
        total_current = sum(r.current_counters for r in results.values())
        total_recommended = sum(r.recommended_counters for r in results.values())
        avg_wait_time = np.mean([r.predicted_wait_time for r in results.values()])
        
        alerts = {
            'critical': [r.department for r in results.values() if r.alert_level == 'critical'],
            'warning': [r.department for r in results.values() if r.alert_level == 'warning'],
            'normal': [r.department for r in results.values() if r.alert_level == 'normal']
        }
        
        return {
            'total_staff_current': total_current,
            'total_staff_recommended': total_recommended,
            'staff_change': total_recommended - total_current,
            'average_wait_time': round(avg_wait_time, 2),
            'departments_needing_increase': [
                r.department for r in results.values() 
                if r.recommended_counters > r.current_counters
            ],
            'departments_can_decrease': [
                r.department for r in results.values() 
                if r.recommended_counters < r.current_counters
            ],
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }


def simulate_allocation_scenario():
    """Simulate allocation for testing"""
    allocator = CounterAllocator()
    
    # Sample department predictions
    predictions = {
        'OPD': {
            'wait_time': 35.5,
            'queue_length': 12,
            'arrival_rate': 2.5,
            'utilization': 0.85
        },
        'Diagnostics': {
            'wait_time': 22.3,
            'queue_length': 6,
            'arrival_rate': 1.8,
            'utilization': 0.75
        },
        'Pharmacy': {
            'wait_time': 18.7,
            'queue_length': 8,
            'arrival_rate': 3.2,
            'utilization': 0.80
        },
        'Emergency': {
            'wait_time': 42.1,
            'queue_length': 5,
            'arrival_rate': 1.2,
            'utilization': 0.90
        }
    }
    
    current = {'OPD': 4, 'Diagnostics': 2, 'Pharmacy': 2, 'Emergency': 3}
    
    results = allocator.recommend_allocation(predictions, current, total_staff=15)
    
    print("\n" + "="*80)
    print("COUNTER ALLOCATION RECOMMENDATIONS")
    print("="*80)
    
    for dept_name, result in results.items():
        print(f"\n{dept_name}:")
        print(f"  Current Counters: {result.current_counters}")
        print(f"  Recommended: {result.recommended_counters}")
        print(f"  Predicted Wait Time: {result.predicted_wait_time:.1f} minutes")
        print(f"  Queue Length: {result.predicted_queue_length}")
        print(f"  Utilization: {result.utilization:.1%}")
        print(f"  Alert Level: {result.alert_level.upper()}")
        print(f"  Reasoning: {result.reasoning}")
    
    summary = allocator.get_allocation_summary(results)
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Staff Used: {summary['total_staff_recommended']}/{allocation_config.TOTAL_STAFF_AVAILABLE}")
    print(f"Average Wait Time: {summary['average_wait_time']:.1f} minutes")
    print(f"Departments Needing Increase: {', '.join(summary['departments_needing_increase']) if summary['departments_needing_increase'] else 'None'}")
    print(f"Critical Alerts: {', '.join(summary['alerts']['critical']) if summary['alerts']['critical'] else 'None'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    simulate_allocation_scenario()