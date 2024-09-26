from gymnasium.envs.registration import register


register("trading/csv", "trading_environment.envs:CsvTradingEnv")
