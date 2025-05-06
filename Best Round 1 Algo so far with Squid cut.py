from datamodel import OrderDepth, TradingState, Order, Trade
from typing import List, Dict
import jsonpickle
import numpy as np
import json

class Trader:
    def __init__(self):
        self.history = {}
        self.last_fills = {}

        self.product_config = {
            "RAINFOREST_RESIN": {
                "window": 30,
                "momentum_window": 8,
                "pos_limit": 50, #Changed from 30 in V6
                "skew_multiplier": 0.03,
                "volatility_threshold": 1,
                "imbalance_weight": 0.3,
                "momentum_weight": 0.1,
                "spread_value": lambda bid, ask: max(1, min(5, int((ask - bid) / 2))),
                "base_volume": 9 #Changed from 7 in V6
            },
            "KELP": {
                "window": 15,
                "momentum_window": 6,
                "pos_limit": 50, #Changed from 30 in V6
                "skew_multiplier": 0.03,
                "volatility_threshold": 0,
                "imbalance_weight": 0.5,
                "momentum_weight": 0.1,
                "spread_value": lambda bid, ask: 1.5,
                "base_volume": 14 #Changed from 8 in V6
            },
            "SQUID_INK": {
                "window": 30, #Changed from 15
                "momentum_window": 5, #Changed from 6 in V3
                "pos_limit": 0, #Changed from 20 in V4
                "skew_multiplier": 0.03, #Changed from 0.03
                "volatility_threshold": 0,
                "imbalance_weight": 0.30,
                "momentum_weight": 0.10, #Changed from 0.2
                "spread_value": lambda bid, ask: 2.0, #changed from 1.5
                "base_volume": 5 #Changed from 6
            }
        }

    def run(self, state: TradingState):
        if state.traderData:
            internal_state = jsonpickle.decode(state.traderData)
            self.history = internal_state.get("history", {})
            self.last_fills = internal_state.get("last_fills", {})
        else:
            self.history = {}
            self.last_fills = {}

        result = {}
        conversions = 0

        for product, order_depth in state.order_depths.items():
            if product not in self.product_config:
                continue

            config = self.product_config[product]
            orders: List[Order] = []
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            best_bid_volume = order_depth.buy_orders[best_bid]
            best_ask_volume = abs(order_depth.sell_orders[best_ask])
            mid_price = (best_bid + best_ask) / 2

            # 🧠 Track mid-price history
            if product not in self.history:
                self.history[product] = []
            self.history[product].append(mid_price)
            if len(self.history[product]) > 100:
                self.history[product] = self.history[product][-100:]

            history = self.history[product]
            window = config["window"]
            fair_value = np.mean(history[-window:])

            momentum_window = config["momentum_window"]
            momentum = mid_price - history[-momentum_window] if len(history) >= momentum_window else 0
            imbalance = (best_bid_volume - best_ask_volume) / (best_bid_volume + best_ask_volume + 1e-6)

            pos = state.position.get(product, 0)
            pos_limit = config["pos_limit"]
            skew = pos * config["skew_multiplier"]

            volatility = np.std(history[-10:]) if len(history) >= 10 else 0
            if volatility > config["volatility_threshold"]:
                fair_value += config["imbalance_weight"] * imbalance + config["momentum_weight"] * momentum
            else:
                fair_value += config["imbalance_weight"] * imbalance + config["momentum_weight"] * momentum

            fair_value -= skew

            recent_fills = self.last_fills.get(product, [])
            recent_fills = [f for f in recent_fills if state.timestamp - f['timestamp'] <= 1000]
            recent_fills_up = sum(1 for f in recent_fills if f['side'] == 'buy')
            recent_fills_down = sum(1 for f in recent_fills if f['side'] == 'sell')
            fill_bias = recent_fills_up - recent_fills_down

            spread = config["spread_value"](best_bid, best_ask)
            buy_price = int(fair_value - spread / 2)
            sell_price = int(fair_value + spread / 2)
            base_volume = config["base_volume"]
            turbo_volume = base_volume + min(2, abs(fill_bias))
            buy_volume = min(turbo_volume, pos_limit - pos)
            sell_volume = min(turbo_volume, pos_limit + pos)

            trade_trigger = "STANDARD"
            if abs(momentum) > 2 or abs(fill_bias) > 1:
                trade_trigger = "EXEC_AWARE"

            print(f"[{state.timestamp}] {product} | Mid={mid_price:.2f} | FV={fair_value:.2f} | Mom={momentum:.2f} | "
                  f"Imb={imbalance:.2f} | Pos={pos} | Fills={fill_bias} | Trigger={trade_trigger}")

            if buy_volume > 0:
                orders.append(Order(product, buy_price, buy_volume))
            if sell_volume > 0:
                orders.append(Order(product, sell_price, -sell_volume))

            result[product] = orders

        # 🧠 Update fill memory
        for product, trades in state.own_trades.items():
            if product not in self.last_fills:
                self.last_fills[product] = []
            for t in trades:
                side = "buy" if t.buyer == "SUBMISSION" else "sell"
                self.last_fills[product].append({
                    "timestamp": t.timestamp,
                    "side": side,
                    "price": t.price,
                    "qty": t.quantity
                })
            self.last_fills[product] = self.last_fills[product][-20:]

        traderData = jsonpickle.encode({
            "history": self.history,
            "last_fills": self.last_fills
        })

        return result, conversions, traderData
