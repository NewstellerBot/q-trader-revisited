import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CsvTradingEnv(gym.Env):
    """
    A custom trading environment that simulates realistic trading conditions.

    Attributes:
        df (pd.DataFrame): DataFrame containing historical market data.
        initial_balance (float): The starting balance for the agent (in currency units).
        transaction_fee_percent (float): Transaction fee percentage (e.g., 0.1 for 0.1% fee).
        execution_delay (int): Number of steps (ticks) to delay order execution.
        action_space (gym.Space): The action space (Discrete with 3 actions: Hold, Buy, Sell).
        observation_space (gym.Space): The observation space containing market data and account status.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, df, initial_balance=10000, transaction_fee_percent=0.1, execution_delay=1
    ):
        """
        Initialize the trading environment.

        Parameters:
            df (pd.DataFrame): Historical market data.
            initial_balance (float): Starting balance in the trading account.
            transaction_fee_percent (float): Percentage fee per transaction.
            execution_delay (int): Delay in steps (ticks) before an order is executed.
        """
        super(CsvTradingEnv, self).__init__()

        # Reset the index of the DataFrame to ensure sequential access
        self.df = df.reset_index()

        # Set the initial account balance
        self.initial_balance = initial_balance

        # Set the transaction fee percentage (e.g., 0.1 for 0.1%)
        self.transaction_fee_percent = transaction_fee_percent

        # Set the execution delay in terms of steps (ticks)
        self.execution_delay = execution_delay

        # Define the action space: 0 - Hold, 1 - Buy, 2 - Sell
        self.action_space = spaces.Discrete(3)

        # Define the observation space: market data plus account information
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                df.shape[1] + 5,
            ),  # Market data columns + SMA + EMA + balance + net_worth + shares_held
        )

        # Initialize state variables
        self.reset()

    def reset(self):
        """
        Reset the environment state at the start of an episode.

        Returns:
            observation (np.array): The initial observation of the environment.
            info (dict): Additional information (empty in this case).
        """
        # Set the starting balance and net worth
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance

        # Initialize the number of shares held
        self.shares_held = 0

        # Reset the current step in the data
        self.current_step = 0

        # Initialize the list of pending orders
        self.orders = []

        # Return the initial observation
        return self._next_observation(), {}

    def _next_observation(self):
        """
        Get the next observation from the environment.

        Returns:
            obs (np.array): An array containing market data and account status.
        """
        # Get the market data for the current step
        frame = self.df.loc[self.current_step].values  # Market data at current step

        # Concatenate market data with account information
        obs = np.concatenate((frame, [self.balance, self.net_worth, self.shares_held]))

        return obs

    def _take_action(self, action):
        """
        Queue the action to be executed after the specified execution delay.

        Parameters:
            action (int): The action to be taken (0: Hold, 1: Buy, 2: Sell).
        """
        # Add the action to the orders list with the step when it should be executed
        self.orders.append(
            {"step": self.current_step + self.execution_delay, "action": action}
        )

    def _execute_order(self, action):
        """
        Execute a single order (buy or sell).

        Parameters:
            action (int): The action to execute (1: Buy, 2: Sell).
        """
        # Get the execution price at the current step
        current_price = self.df.loc[self.current_step, "Close"]

        if action == 1:  # Buy action
            # Calculate the maximum number of shares that can be bought
            max_shares = self.balance / current_price
            shares_bought = max_shares  # Here, buying as many as possible

            # Calculate the total cost including transaction fee
            cost = shares_bought * current_price
            fee = cost * (self.transaction_fee_percent / 100)

            # Update balance and shares held
            self.balance -= cost + fee
            self.shares_held += shares_bought

        elif action == 2:  # Sell action
            # Sell all shares held
            shares_sold = self.shares_held

            # Calculate the total revenue including transaction fee
            revenue = shares_sold * current_price
            fee = revenue * (self.transaction_fee_percent / 100)

            # Update balance and shares held
            self.balance += revenue - fee
            self.shares_held -= shares_sold

    def _execute_orders(self):
        """
        Execute orders whose execution delay has expired.
        """
        # Create a list to hold orders that have not yet reached their execution time
        remaining_orders = []

        for order in self.orders:
            if order["step"] == self.current_step:
                # Execute the order if the execution delay has expired
                self._execute_order(order["action"])
            elif order["step"] > self.current_step:
                # Keep the order if it's not yet time to execute
                remaining_orders.append(order)

        # Update the orders list with remaining orders
        self.orders = remaining_orders

    def step(self, action):
        """
        Execute one time step within the environment.

        Parameters:
            action (int): The action taken by the agent (0: Hold, 1: Buy, 2: Sell).

        Returns:
            obs (np.array): The next observation.
            reward (float): The reward after taking the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated (always False here).
            info (dict): Additional information, e.g., net worth.
        """
        # First, execute any pending orders whose delay has expired
        self._execute_orders()

        # Then, take the new action (which will be executed after a delay)
        self._take_action(action)

        # Advance to the next time step
        self.current_step += 1

        # Loop back to the start if at the end (optional)
        if self.current_step >= len(self.df) - self.execution_delay:
            self.current_step = 0  # For continuous training

        # Calculate the net worth considering the delayed execution price
        delayed_step = self.current_step + self.execution_delay
        if delayed_step >= len(self.df):
            delayed_step = len(self.df) - 1  # Avoid index error

        # Price at which the shares would be valued after the execution delay
        delayed_price = self.df.loc[delayed_step, "Close"]

        # Update net worth
        self.net_worth = self.balance + self.shares_held * delayed_price

        # Calculate reward (change in net worth)
        reward = self.net_worth - self.initial_balance

        # Determine if the episode is done
        done = self.net_worth <= 0 or self.current_step == len(self.df) - 1

        # Get the next observation
        obs = self._next_observation()

        # Additional info dictionary
        info = {"net_worth": self.net_worth}

        return obs, reward, done, False, info

    def render(self):
        """
        Render the environment's state to the console.
        """
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Shares Held: {self.shares_held}")
        print(f"Net Worth: {self.net_worth:.2f}")
        print(f"Profit: {profit:.2f}")
