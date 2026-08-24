import datetime
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==================== 1. 监控股票池 (19只核心标的) ====================
WATCHLIST = [
    "TSLA",
    "NVDA",
    "AAPL",
    "MSFT",
    "META",
    "GOOGL",
    "INTC",
    "IBM",
    "CRWV",
    "RKLB",
    "AVGO",
    "PLTR",
    "QQQ",
    "SPY",
    "AMD",
    "AMZN",
    "NFLX",
    "SPCX",
    "TSM",
]

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
STATE_FILE = "agent_state.json"


# ==================== 2. 云端状态持久化与持仓管理 ====================
def load_state() -> dict:
    """加载云端保存的历史买点和活跃持仓"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "sent_signals" not in data:
                    data["sent_signals"] = {}
                if "active_positions" not in data:
                    data["active_positions"] = {}
                return data
        except Exception:
            pass
    return {"sent_signals": {}, "active_positions": {}}


def save_state(state: dict):
    """保存最新状态到云端文件"""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存状态文件失败: {e}")


# ==================== 3. 指标与波浪积分算法 ====================
def calculate_macd_and_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["HIST"] = (df["DIF"] - df["DEA"]) * 2
    df["EMA_250"] = df["Close"].ewm(span=250, adjust=False).mean()
    return df


def segment_macd_waves(df: pd.DataFrame) -> List[Dict]:
    waves = []
    if len(df) < 20:
        return waves
    df = df.copy()
    df["sign"] = np.where(df["HIST"] < 0, -1, 1)
    df["group"] = (df["sign"] != df["sign"].shift()).cumsum()

    for _, group_df in df.groupby("group"):
        wave_type = "red" if group_df["sign"].iloc[0] == -1 else "green"
        area = group_df["HIST"].abs().sum()
        if wave_type == "red":
            extreme_dif = group_df["DIF"].min()
            extreme_price = group_df["Low"].min()
        else:
            extreme_dif = group_df["DIF"].max()
            extreme_price = group_df["High"].max()

        waves.append(
            {
                "type": wave_type,
                "start_time": str(group_df.index[0]),
                "end_time": str(group_df.index[-1]),
                "area": float(area),
                "extreme_dif": float(extreme_dif),
                "extreme_price": float(extreme_price),
                "bars": len(group_df),
            }
        )
    return waves


# ==================== 4. Discord 卡片推送模块 ====================
def send_discord_buy_card(data: dict) -> bool:
    """发送绿色买入卡片"""
    if not DISCORD_WEBHOOK:
        return False

    embed = {
        "title": f"🚀【1h MACD 面积背离买点 T1】 - {data['ticker']}",
        "color": 0x2ECC71,  # 绿色
        "description": f"**标的代码**: `{data['ticker']}` | **建议买入价**: `${data['current_price']:.2f}`",
        "fields": [
            {
                "name": "📉 价格形态 (创新低破底)",
                "value": f"P1: `${data['P1']:.2f}` ➔ P2: `${data['P2']:.2f}`",
                "inline": True,
            },
            {
                "name": "⚡ DIF 指标 (底背离抬高)",
                "value": f"A1: `{data['A1']:.4f}` ➔ A2: `{data['A2']:.4f}`",
                "inline": True,
            },
            {
                "name": "📊 面积衰竭 (B2 < 0.5 B1)",
                "value": f"**{data['ratio']:.1%}** (B1: `{data['B1']:.1f}` / B2: `{data['B2']:.1f}`)",
                "inline": False,
            },
            {
                "name": "🎯 第一止盈目标 (250 EMA)",
                "value": f"**${data['target_ema250']:.2f}**",
                "inline": True,
            },
            {
                "name": "🛑 建议止损位 (-5% ~ -10%)",
                "value": f"${data['sl_5']:.2f} ~ ${data['sl_10']:.2f}",
                "inline": True,
            },
        ],
        "footer": {
            "text": "MACD 1h Agent • 已自动加入云端持仓追踪，达标后自动提醒止盈"
        },
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    payload = {
        "username": "MACD Trading Bot",
        "avatar_url": "https://i.imgur.com/4M34hi2.png",
        "embeds": [embed],
    }
    try:
        resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        return resp.status_code in [200, 204]
    except Exception as e:
        print(f"Discord 推送失败: {e}")
        return False


def send_discord_exit_card(
    ticker: str,
    pos: dict,
    current_price: float,
    current_ema250: float,
    reasons: List[str],
    alert_type: str,
) -> bool:
    """发送金色/橙红色止盈与出场卡片"""
    if not DISCORD_WEBHOOK:
        return False

    entry_price = pos["entry_price"]
    profit_pct = (current_price - entry_price) / entry_price
    profit_str = (
        f"+{profit_pct*100:.2f}%"
        if profit_pct >= 0
        else f"{profit_pct*100:.2f}%"
    )

    # 样式颜色：止盈用金色(0xF1C40F)，顶背离用橙色(0xE67E22)，止损用红色(0xE74C3C)
    color = 0xF1C40F if profit_pct >= 0 else 0xE74C3C
    if "顶背离" in "".join(reasons):
        color = 0xE67E22

    embed = {
        "title": f"🔔【持仓出场/止盈提醒】 - {ticker} ({profit_str})",
        "color": color,
        "description": f"**标的代码**: `{ticker}`\n**买入建仓价**: `${entry_price:.2f}` (买入时间: {pos.get('entry_time', 'N/A')})\n**当前现价**: `${current_price:.2f}`\n**累计盈亏**: `{profit_str}`",
        "fields": [
            {
                "name": "📌 触发的具体出场规则",
                "value": "\n".join([f"• {r}" for r in reasons]),
                "inline": False,
            },
            {
                "name": "🎯 当前 1h 250 EMA",
                "value": f"${current_ema250:.2f}",
                "inline": True,
            },
            {
                "name": "💡 操作建议",
                "value": "已达成出场条件，请根据个人策略分批止盈或清仓离场。",
                "inline": False,
            },
        ],
        "footer": {
            "text": "MACD 1h Agent • 出场监控完成 (该标的将从持仓追踪中移除)"
        },
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    payload = {
        "username": "MACD Trading Bot",
        "avatar_url": "https://i.imgur.com/4M34hi2.png",
        "embeds": [embed],
    }
    try:
        resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        return resp.status_code in [200, 204]
    except Exception as e:
        print(f"Discord 推送失败: {e}")
        return False


# ==================== 5. 持仓出场条件检查器 ====================
def check_exit_signals(
    df_1h: pd.DataFrame, ticker: str, pos: dict
) -> List[str]:
    """检查持仓是否满足手稿的 4 项出场规则"""
    reasons = []
    latest_close = float(df_1h["Close"].iloc[-1])
    latest_high = float(df_1h["High"].iloc[-1])
    latest_low = float(df_1h["Low"].iloc[-1])
    current_ema250 = float(df_1h["EMA_250"].iloc[-1])

    entry_price = pos["entry_price"]
    gain_pct = (latest_close - entry_price) / entry_price

    # 规则 1: 触及 250 EMA (1h 级别)
    if (
        latest_low <= current_ema250 <= latest_high
        or latest_close >= current_ema250
    ):
        reasons.append(
            f"🎯 价格触及/突破 1h 250 EMA 目标位 (${current_ema250:.2f})"
        )

    # 规则 2: 止盈 (>10% 随时止盈)
    if gain_pct >= 0.10:
        reasons.append(f"💰 浮盈超过 10% (当前涨幅: +{gain_pct*100:.2f}%)")

    # 规则 3: 1h 顶背离确认 (必须出场)
    waves = segment_macd_waves(df_1h)
    green_waves = [w for w in waves if w["type"] == "green"]
    if len(green_waves) >= 2:
        gw1, gw2 = green_waves[-2], green_waves[-1]
        # 价格新高但 DIF 降低且绿柱处于衰竭阶段
        if (
            gw2["extreme_price"] > gw1["extreme_price"]
            and gw2["extreme_dif"] < gw1["extreme_dif"]
        ):
            if df_1h["HIST"].iloc[-1] < df_1h["HIST"].iloc[-2]:  # 绿柱拐头收缩
                reasons.append(
                    "🚨 1h 级别形成顶背离确认，多头动能衰竭，强制离场！"
                )

    # 规则 4: 止损 (5% - 10%)
    if gain_pct <= -0.05:
        reasons.append(f"⚠️ 跌幅达 {gain_pct*100:.2f}%，触发 5%~10% 止损保护")

    return reasons


# ==================== 6. 主扫描逻辑 ====================
def main():
    state = load_state()
    has_state_change = False

    print(
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始扫描标的池 ({len(WATCHLIST)} 只)..."
    )

    for ticker in WATCHLIST:
        try:
            df_1h = yf.download(
                ticker,
                period="60d",
                interval="1h",
                prepost=True,
                progress=False,
            )
            if df_1h.empty or len(df_1h) < 30:
                continue
            if isinstance(df_1h.columns, pd.MultiIndex):
                df_1h.columns = df_1h.columns.get_level_values(0)

            df_1h = calculate_macd_and_ema(df_1h)
            latest_close = float(df_1h["Close"].iloc[-1])
            target_ema250 = float(df_1h["EMA_250"].iloc[-1])

            # ---------------- 流程 A: 优先检查现有持仓的出场/止盈条件 ----------------
            if ticker in state["active_positions"]:
                pos = state["active_positions"][ticker]
                exit_reasons = check_exit_signals(df_1h, ticker, pos)

                if exit_reasons:
                    print(
                        f"[{ticker}] 触发止盈/出场条件: {', '.join(exit_reasons)}"
                    )
                    if send_discord_exit_card(
                        ticker,
                        pos,
                        latest_close,
                        target_ema250,
                        exit_reasons,
                        "EXIT",
                    ):
                        # 出场提醒成功后，从活跃持仓中移除，避免重复轰炸
                        del state["active_positions"][ticker]
                        has_state_change = True
                continue  # 已有持仓的股票不重复计算买点

            # ---------------- 流程 B: 扫描新买点 (1h 底背离 + 面积减半) ----------------
            waves = segment_macd_waves(df_1h)
            red_waves = [w for w in waves if w["type"] == "red"]
            if len(red_waves) < 2:
                continue

            w1, w2 = red_waves[-2], red_waves[-1]
            P1, A1, B1 = w1["extreme_price"], w1["extreme_dif"], w1["area"]
            P2, A2, B2 = w2["extreme_price"], w2["extreme_dif"], w2["area"]

            cond_divergence = (P2 < P1) and (A2 > A1) and (B2 < (0.5 * B1))
            cond_inflection = (
                df_1h["HIST"].iloc[-1] > df_1h["HIST"].iloc[-2]
            )  # 红柱开始向0轴收敛

            if cond_divergence and cond_inflection:
                signal_id = f"{ticker}_{w1['end_time']}_{w2['start_time']}"

                if signal_id in state["sent_signals"]:
                    continue  # 已推送过，跳过

                signal_data = {
                    "ticker": ticker,
                    "current_price": latest_close,
                    "P1": P1,
                    "P2": P2,
                    "A1": A1,
                    "A2": A2,
                    "B1": B1,
                    "B2": B2,
                    "ratio": B2 / B1,
                    "target_ema250": target_ema250,
                    "sl_5": latest_close * 0.95,
                    "sl_10": latest_close * 0.90,
                }

                if send_discord_buy_card(signal_data):
                    print(f"[{ticker}] 🚀 买入信号 T1 推送成功！")
                    now_str = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    state["sent_signals"][signal_id] = now_str
                    # 自动加入活跃持仓跟踪池
                    state["active_positions"][ticker] = {
                        "entry_price": latest_close,
                        "entry_time": now_str,
                        "p1": P1,
                        "p2": P2,
                        "target_ema250": target_ema250,
                    }
                    has_state_change = True

        except Exception as e:
            print(f"处理 {ticker} 发生异常: {e}")

    if has_state_change:
        save_state(state)

    print("本轮扫描完成。\n")


if __name__ == "__main__":
    main()
