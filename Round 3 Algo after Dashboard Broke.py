from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle
import numpy as np

class OnlineDirectionModel:
    def __init__(self, feature_dim=4, learning_rate=0.01):
        self.weights = np.zeros(feature_dim)
        self.bias = 0.0
        self.lr = learning_rate

    def predict_proba(self, x):
        z = np.dot(self.weights, x) + self.bias
        return 1 / (1 + np.exp(-z))

    def update(self, x, direction):
        pred = self.predict_proba(x)
        error = direction - pred
        grad = self.lr * error
        self.weights += grad * np.array(x)
        self.bias += grad
        self.weights = np.clip(self.weights, -10, 10)
        self.bias = np.clip(self.bias, -100, 100)

class Trader:
    def __init__(self):
        self.models = {}
        self.feature_histories = {}
        self.price_histories = {}

        self.products_ml = [
            "JAMS", "DJEMBES", "VOLCANIC_ROCK",
            "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        self.ml_signal_config = {
            "JAMS": {"high": 0.90, "low": 0.10}, #Changed
            "DJEMBES": {"high": 0.90, "low": 0.10}, #Changed
            "VOLCANIC_ROCK": {"high": 0.80, "low": 0.20},
            "VOLCANIC_ROCK_VOUCHER_9500": {"high": 0.80, "low": 0.20},
            "VOLCANIC_ROCK_VOUCHER_9750": {"high": 0.80, "low": 0.20},
            "VOLCANIC_ROCK_VOUCHER_10000": {"high": 0.80, "low": 0.20},
            "VOLCANIC_ROCK_VOUCHER_10250": {"high": 0.80, "low": 0.20},
            "VOLCANIC_ROCK_VOUCHER_10500": {"high": 0.80, "low": 0.20}
        }

        self.products_stat = ["KELP", "RAINFOREST_RESIN"]
        self.products_zscore = {
            "CROISSANTS": {"short_window": 8, "long_window": 100, "z_entry": 1.8, "volume": 15},
            "SQUID_INK": {"short_window": 10, "long_window": 100, "z_entry": 3, "volume": 15} #Changed
        }
        self.products_basket = ["PICNIC_BASKET1", "PICNIC_BASKET2"]

        self.all_products = self.products_ml + self.products_stat + list(self.products_zscore.keys()) + self.products_basket

        self.position_limits = {
            "CROISSANTS": 250, "JAMS": 0, "DJEMBES": 0, "SQUID_INK": 0,
            "KELP": 50, "RAINFOREST_RESIN": 50,
            "PICNIC_BASKET1": 0, "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }

    def run(self, state: TradingState):
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
            self.feature_histories = data.get("feature_histories", {})
            self.price_histories = data.get("price_histories", {})
            weights = data.get("model_weights", {})
            bias = data.get("model_bias", {})
            for p in self.products_ml:
                model = OnlineDirectionModel()
                model.weights = np.array(weights.get(p, [0.0]*4))
                model.bias = bias.get(p, 0.0)
                self.models[p] = model
        else:
            for p in self.products_ml:
                self.models[p] = OnlineDirectionModel()

        result = {}
        time_fraction = state.timestamp / 1_000_000

        for product in self.all_products:
            order_depth = state.order_depths.get(product)
            if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.price_histories.setdefault(product, []).append(mid_price)
            self.price_histories[product] = self.price_histories[product][-150:]

            pos = state.position.get(product, 0)
            pos_limit = self.position_limits[product]
            decay_factor = max(0.2, 1 - time_fraction)
            safe_limit = int(pos_limit * decay_factor)
            orders: List[Order] = []

            # --- Z-SCORE STRATEGY ---
            if product in self.products_zscore:
                config = self.products_zscore[product]
                if len(self.price_histories[product]) < config["long_window"]:
                    continue
                short_ma = np.mean(self.price_histories[product][-config["short_window"]:])
                long_ma = np.mean(self.price_histories[product][-config["long_window"]:])
                std_dev = np.std(self.price_histories[product][-config["long_window"]:])
                z = (mid_price - long_ma) / (std_dev + 1e-6)

                if z <= -config["z_entry"] and pos < pos_limit:
                    vol = min(config["volume"], pos_limit - pos)
                    orders.append(Order(product, best_ask, vol))
                elif z >= config["z_entry"] and pos > -pos_limit:
                    vol = min(config["volume"], pos_limit + pos)
                    orders.append(Order(product, best_bid, -vol))

            # --- STAT-ARB STRATEGY ---
            elif product in self.products_stat:
                history = self.price_histories[product]
                if len(history) < 20:
                    continue
                fair_value = np.mean(history[-20:])
                momentum = history[-1] - history[-6] if len(history) >= 6 else 0
                bid_vol = sum(order_depth.buy_orders.values())
                ask_vol = -sum(order_depth.sell_orders.values())
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

                fair_value += 0.1 * momentum + 0.3 * imbalance
                fair_value -= 0.03 * pos  # skew

                buy_price = int(fair_value - 1)
                sell_price = int(fair_value + 1)
                buy_vol = min(10, pos_limit - pos)
                sell_vol = min(10, pos_limit + pos)

                if buy_vol > 0:
                    orders.append(Order(product, buy_price, buy_vol))
                if sell_vol > 0:
                    orders.append(Order(product, sell_price, -sell_vol))

            # --- ML STRATEGY (now product-specific thresholds) ---
            elif product in self.products_ml:
                spread = (best_ask - best_bid) / 100
                bid_vol = sum(order_depth.buy_orders.values())
                ask_vol = -sum(order_depth.sell_orders.values())
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
                momentum = 0
                if self.price_histories.get(product):
                    momentum = (mid_price - self.price_histories[product][-1]) / 100
                x = [mid_price / 1000, spread, imbalance, momentum]
                self.feature_histories.setdefault(product, []).append(x)

                if len(self.price_histories[product]) > 2:
                    y = int(self.price_histories[product][-1] > self.price_histories[product][-2])
                    self.models[product].update(self.feature_histories[product][-2], y)

                proba = self.models[product].predict_proba(x)
                confidence = abs(proba - 0.5) * 2

                # Pull per-product thresholds
                thresholds = self.ml_signal_config.get(product, {"high": 0.80, "low": 0.20})
                signal = 1 if proba > thresholds["high"] else -1 if proba < thresholds["low"] else 0
                volume = int(10 * confidence)
                volume = min(volume, safe_limit - pos if signal == 1 else safe_limit + pos)

                if signal == 1 and volume > 0:
                    orders.append(Order(product, best_ask, volume))
                if signal == -1 and volume > 0:
                    orders.append(Order(product, best_bid, -volume))

            # --- BASKET PRICING ---
            elif product in self.products_basket:
                try:
                    croissant = self.price_histories["CROISSANTS"][-1]
                    jam = self.price_histories["JAMS"][-1]
                    djembe = self.price_histories["DJEMBES"][-1]
                except KeyError:
                    continue

                if product == "PICNIC_BASKET1":
                    synthetic = 6 * croissant + 3 * jam + 1 * djembe
                else:
                    synthetic = 4 * croissant + 2 * jam

                buy_price = int(synthetic - 1)
                sell_price = int(synthetic + 1)

                if pos < pos_limit:
                    orders.append(Order(product, buy_price, min(10, pos_limit - pos)))
                if pos > -pos_limit:
                    orders.append(Order(product, sell_price, -min(10, pos_limit + pos)))

            result[product] = orders

        traderData = jsonpickle.encode({
            "feature_histories": {k: v[-100:] for k, v in self.feature_histories.items()},
            "price_histories": {k: v[-100:] for k, v in self.price_histories.items()},
            "model_weights": {k: self.models[k].weights.tolist() for k in self.products_ml},
            "model_bias": {k: self.models[k].bias for k in self.products_ml}
        })

        return result, 0, traderData
