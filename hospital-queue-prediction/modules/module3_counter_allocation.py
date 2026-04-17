"""
MODULE 3: Counter Allocation Recommendation Engine
====================================================
AI-Based Queue-Time Prediction System

This module implements optimization algorithms for counter allocation:
- Greedy algorithm for fast decisions
- Linear Programming for optimal allocation
- Workload balancing logic
- Priority-based allocation

As specified in project documentation Section 5, Module 3
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

from config import paths, allocation_config

logger = logging.getLogger(__name__)


@dataclass
class AllocationRecommendation:
    """Data class for counter allocation recommendations"""
    department: str
    current_counters: int
    recommended_counters: int
    predicted_wait_time: float
    expected_wait_reduction: float
    priority_score: float
    action: str  # 'increase', 'decrease', 'maintain'
    justification: str


class GreedyAllocator:
    """
    Greedy algorithm for counter allocation
    Fast, near-optimal solutions suitable for real-time decisions
    """

    def __init__(self):
        self.name = "Greedy Allocator"
        logger.info(f"{self.name} initialized")

    def allocate(self, department_predictions: Dict[str, float],
                 current_allocation: Dict[str, int],
                 total_staff: int) -> Dict[str, int]:

        logger.info(f"\n[GREEDY ALLOCATION] Starting allocation...")

        priorities = {}
        for dept, wait_time in department_predictions.items():
            dept_config = allocation_config.DEPARTMENT_CONFIG.get(dept, {})
            weight = dept_config.get('priority_weight', 1.0)
            priorities[dept] = wait_time * weight

        sorted_depts = sorted(priorities.items(), key=lambda x: x[1], reverse=True)

        new_allocation = {
            dept: allocation_config.DEPARTMENT_CONFIG[dept]['min_counters']
            for dept in department_predictions.keys()
        }

        remaining_staff = total_staff - sum(new_allocation.values())

        while remaining_staff > 0:
            allocated = False
            for dept, priority in sorted_depts:
                max_counters = allocation_config.DEPARTMENT_CONFIG[dept]['max_counters']
                if new_allocation[dept] < max_counters:
                    new_allocation[dept] += 1
                    remaining_staff -= 1
                    allocated = True
                    break
            if not allocated:
                break

        logger.info(f"[OK] Allocation complete: {new_allocation}")
        return new_allocation


class LinearProgrammingAllocator:
    """
    Linear Programming for optimal counter allocation
    """

    def __init__(self):
        self.name = "Linear Programming Allocator"
        try:
            from pulp import LpProblem, LpMinimize, LpVariable, lpSum
            self.pulp_available = True
            logger.info(f"{self.name} initialized")
        except ImportError:
            self.pulp_available = False
            logger.warning("PuLP not installed, LP optimization unavailable")

    def allocate(self, department_predictions: Dict[str, float],
                 current_allocation: Dict[str, int],
                 total_staff: int) -> Dict[str, int]:

        if not self.pulp_available:
            logger.warning("Falling back to greedy allocation")
            greedy = GreedyAllocator()
            return greedy.allocate(department_predictions, current_allocation, total_staff)

        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger

        logger.info(f"\n[LP OPTIMIZATION] Starting allocation...")

        departments = list(department_predictions.keys())

        prob = LpProblem("Counter_Allocation", LpMinimize)

        counters = {
            dept: LpVariable(
                f"counters_{dept}",
                lowBound=allocation_config.DEPARTMENT_CONFIG[dept]['min_counters'],
                upBound=allocation_config.DEPARTMENT_CONFIG[dept]['max_counters'],
                cat=LpInteger
            )
            for dept in departments
        }

        # Fully linear objective (fixed)
        objective = lpSum([
            allocation_config.DEPARTMENT_CONFIG[dept]['priority_weight'] *
            department_predictions[dept]
            - 2 * counters[dept] *
            allocation_config.DEPARTMENT_CONFIG[dept]['priority_weight']
            for dept in departments
        ])

        prob += objective

        prob += lpSum([counters[dept] for dept in departments]) <= total_staff

        prob.solve()

        optimal_allocation = {
            dept: int(counters[dept].varValue)
            for dept in departments
        }

        logger.info(f"[OK] Optimal allocation: {optimal_allocation}")
        return optimal_allocation


# ✅ FIXED INDENTATION HERE (THIS WAS THE ERROR)
class AllocationEngine:
    """
    Complete counter allocation recommendation system
    Integrates prediction, optimization, and reporting
    """

    def __init__(self, method: str = 'linear_programming'):
        self.method = method
        self.allocator = (
            LinearProgrammingAllocator()
            if method == 'linear_programming'
            else GreedyAllocator()
        )
        self.recommendation_history = []
        logger.info("=" * 80)
        logger.info("MODULE 3: Counter Allocation Recommendation Engine")
        logger.info("=" * 80)

    def generate_recommendations(self,
                                 department_predictions: Dict[str, float],
                                 current_allocation: Dict[str, int],
                                 available_staff: Optional[int] = None
                                 ) -> List[AllocationRecommendation]:

        logger.info("\n[PHASE 4] Generating Counter Allocation Recommendations")
        logger.info("-" * 80)

        if available_staff is None:
            available_staff = allocation_config.total_available_staff

        optimal_allocation = self.allocator.allocate(
            department_predictions,
            current_allocation,
            available_staff
        )

        recommendations = []

        for dept in department_predictions.keys():
            current = current_allocation[dept]
            recommended = optimal_allocation[dept]
            predicted_wait = department_predictions[dept]

            if recommended > current:
                reduction = predicted_wait * ((current - recommended) / (current + 1))
                action = 'increase'
            elif recommended < current:
                reduction = predicted_wait * ((current - recommended) / (current + 1))
                action = 'decrease'
            else:
                reduction = 0
                action = 'maintain'

            dept_config = allocation_config.DEPARTMENT_CONFIG[dept]
            priority = predicted_wait * dept_config['priority_weight']

            justification = self._generate_justification(
                dept, predicted_wait, current, recommended, action
            )

            rec = AllocationRecommendation(
                department=dept,
                current_counters=current,
                recommended_counters=recommended,
                predicted_wait_time=predicted_wait,
                expected_wait_reduction=abs(reduction),
                priority_score=priority,
                action=action,
                justification=justification
            )

            recommendations.append(rec)

        recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        self._log_recommendations(recommendations)

        self.recommendation_history.append({
            'timestamp': pd.Timestamp.now(),
            'recommendations': recommendations
        })

        self.latest_recommendations = recommendations

        return recommendations

    def _generate_justification(self, dept, wait_time, current, recommended, action):

        if action == 'increase':
            return (f"High predicted wait time ({wait_time:.1f} min). "
                    f"Increase counters from {current} to {recommended} "
                    f"to reduce congestion.")
        elif action == 'decrease':
            return (f"Low predicted wait time ({wait_time:.1f} min). "
                    f"Reduce counters from {current} to {recommended} "
                    f"to optimize staff utilization.")
        else:
            return (f"Current allocation ({current} counters) is optimal "
                    f"for predicted wait time ({wait_time:.1f} min).")

    def _log_recommendations(self, recommendations):
        logger.info("\n[RECOMMENDATIONS]")
        logger.info("-" * 80)
        for rec in recommendations:
            logger.info(
                f"{rec.department} | {rec.action.upper()} | "
                f"{rec.current_counters} -> {rec.recommended_counters} | "
                f"{rec.predicted_wait_time:.1f} min"
            )
    
    def generate_alerts(self, recommendations):
            """
            Generate operational alerts based on recommendations
            """
            alerts = []

            for rec in recommendations:
                if rec.action == 'increase' and rec.predicted_wait_time > 25:
                    alerts.append(
                        f"[HIGH WAIT ALERT] {rec.department}: "
                        f"Predicted wait {rec.predicted_wait_time:.1f} min. "
                        f"Immediate counter increase recommended."
                    )

                elif rec.action == 'decrease' and rec.predicted_wait_time < 10:
                    alerts.append(
                        f"[LOW LOAD ALERT] {rec.department}: "
                        f"Low demand detected. Staff can be reallocated."
                    )

            logger.info("\n[ALERTS]")
            logger.info("-" * 80)

            if not alerts:
                logger.info("No critical alerts generated.")
            else:
                for alert in alerts:
                    logger.info(alert)

            return alerts
    
    def save_recommendations(self, filename="allocation_recommendations.csv"):
        """
        Save latest recommendations to CSV file
        """
        import pandas as pd
        import os

        if not hasattr(self, "latest_recommendations"):
            logger.warning("No recommendations available to save.")
            return

        data = []
        for rec in self.latest_recommendations:
            data.append({
                "Department": rec.department,
                "Current Counters": rec.current_counters,
                "Recommended Counters": rec.recommended_counters,
                "Action": rec.action.upper(),
                "Predicted Wait (min)": rec.predicted_wait_time
            })

        df = pd.DataFrame(data)

        save_path = os.path.join("outputs", filename)
        os.makedirs("outputs", exist_ok=True)

        df.to_csv(save_path, index=False)

        logger.info(f"[OK] Recommendations saved to {save_path}")




if __name__ == "__main__":
    logger.info("Testing Module 3: Counter Allocation Engine")

    engine = AllocationEngine(method='linear_programming')

    department_predictions = {
        'OPD': 25.5,
        'Diagnostics': 18.3,
        'Pharmacy': 12.1
    }

    current_allocation = {
        'OPD': 4,
        'Diagnostics': 3,
        'Pharmacy': 3
    }

    recommendations = engine.generate_recommendations(
        department_predictions,
        current_allocation,
        available_staff=15
    )
