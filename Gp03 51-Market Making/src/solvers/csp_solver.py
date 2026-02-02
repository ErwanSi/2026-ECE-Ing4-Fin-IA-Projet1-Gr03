from ortools.sat.python import cp_model
import numpy as np

class CSPSolver:
    """
    Experimental solver using Constraint Programming (OR-Tools).
    Models the market making problem as a discrete scheduling problem:
    - At each step, choose to BUY, SELL, or HOLD.
    - Constraints: Max Inventory, Cash Limits.
    - Objective: Maximize Final Cash + Inventory Valuation.
    
    Note: This is a deterministic simplification. It assumes we know the order flow
    or optimize for a worst-case/best-case scenario over a discretized horizon.
    """
    
    def __init__(self, horizon_steps=10, max_inventory=5):
        self.horizon = horizon_steps
        self.max_inv = max_inventory
        self.model = cp_model.CpModel()
        
    def solve_trajectory(self, price_path, buy_arrivals, sell_arrivals):
        """
        Solve for optimal actions given KNOWN arrival times (Oracle/Perfect Foresight).
        Useful for upper-bound analysis.
        
        price_path: list of prices
        buy_arrivals: list of booleans (is there a buy order arriving at t?)
        sell_arrivals: list of booleans (is there a sell order arriving at t?)
        """
        # Variables
        inventory = [self.model.NewIntVar(-self.max_inv, self.max_inv, f'inv_{t}') for t in range(self.horizon + 1)]
        cash = [self.model.NewIntVar(-100000, 100000, f'cash_{t}') for t in range(self.horizon + 1)] # Scaled integer cash
        
        # Actions: 0=None, 1=FillBuy, 2=FillSell
        # We assume we can fill if arrival exists
        actions = [self.model.NewIntVar(0, 2, f'act_{t}') for t in range(self.horizon)]
        
        # Initial State
        self.model.Add(inventory[0] == 0)
        self.model.Add(cash[0] == 0)
        
        for t in range(self.horizon):
            price = int(price_path[t] * 100) # Cent precision
            
            # Transition Logic
            # If Action=1 (Buy Fill) -> Inv++, Cash -= Price (approx, assuming spread=0 for simplified CSP)
            # If Action=2 (Sell Fill) -> Inv--, Cash += Price
            
            # Constraints on Actions based on Arrivals
            # Can only Buy (1) if buy_arrivals[t] is True
            if not buy_arrivals[t]:
                self.model.Add(actions[t] != 1)
            
            # Can only Sell (2) if sell_arrivals[t] is True
            if not sell_arrivals[t]:
                self.model.Add(actions[t] != 2)
                
            # Inventory Updates
            # inv[t+1] = inv[t] + (1 if act==1) - (1 if act==2)
            buy_bool = self.model.NewBoolVar(f'buy_{t}')
            sell_bool = self.model.NewBoolVar(f'sell_{t}')
            self.model.Add(actions[t] == 1).OnlyEnforceIf(buy_bool)
            self.model.Add(actions[t] != 1).OnlyEnforceIf(buy_bool.Not())
            self.model.Add(actions[t] == 2).OnlyEnforceIf(sell_bool)
            self.model.Add(actions[t] != 2).OnlyEnforceIf(sell_bool.Not())
            
            self.model.Add(inventory[t+1] == inventory[t] + buy_bool - sell_bool)
            
            # Cash Updates
            # cash[t+1] = cash[t] - price * buy + price * sell
            self.model.Add(cash[t+1] == cash[t] - price * buy_bool + price * sell_bool)
            
        # Objective: Maximize Final Wealth (Cash + Inv * FinalPrice)
        final_price = int(price_path[-1] * 100)
        final_wealth = self.model.NewIntVar(-1000000, 1000000, 'final_wealth')
        self.model.Add(final_wealth == cash[self.horizon] + inventory[self.horizon] * final_price)
        
        self.model.Maximize(final_wealth)
        
        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return solver.Value(final_wealth) / 100.0, [solver.Value(a) for a in actions]
        else:
            return None, []