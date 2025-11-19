from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Strategy:
    name: str
    pattern_months: List[int]  # e.g. [6, 6] means 2x6m in a year

@dataclass
class Config:
    total_amount: float        # total money, RUB
    people_count: int          # number of people
    key_rate: float            # e.g. 0.16 for 16%
    tax_rate: float            # e.g. 0.13
    rates_by_term: Dict[int, float]  # months -> annual rate (e.g. {6: 0.145})
    step: float = 500_000.0    # discretization step for allocation

def compute_tax_free_limit_per_person(cfg: Config) -> float:
    """
    Russian rule (с 2021):
    Налоговая база = max(0, процентовый доход - key_rate * 1_000_000).
    key_rate * 1e6 — это «безналоговый» годовой доход по процентам на человека.
    """
    return cfg.key_rate * 1_000_000.0

def effective_annual_rate(strategy: Strategy, cfg: Config) -> float:
    """
    Для годовой стратегии (сумма месяцев = 12) считаем эффективную годовую
    простую ставку, предполагая, что весь год на счету лежит один и тот же
    principal и мы игнорируем сложные проценты между периодами.
    """
    total_year_fraction = 0.0
    total_interest_factor = 0.0
    for m in strategy.pattern_months:
        r = cfg.rates_by_term[m]
        frac = m / 12.0
        total_year_fraction += frac
        total_interest_factor += r * frac
    if abs(total_year_fraction - 1.0) > 1e-6:
        raise ValueError(f"Strategy {strategy.name} does not cover exactly 12 months")
    # total_interest_factor — это сколько процентов набежит за год на 1 рубль
    return total_interest_factor

def net_interest_for_amount_and_rate(amount: float, eff_rate: float, cfg: Config, tax_free_limit: float) -> float:
    """
    Чистые проценты после НДФЛ для одного человека.
    Проценты считаем просто: interest = amount * eff_rate.
    """
    interest = amount * eff_rate
    taxable = max(0.0, interest - tax_free_limit)
    tax = taxable * cfg.tax_rate
    return interest - tax

def best_strategy_for_person_amount(
    amount: float,
    strategies: List[Strategy],
    cfg: Config,
    tax_free_limit: float,
) -> Tuple[float, Optional[Strategy], float]:
    """
    Для заданной суммы на одного человека выбираем стратегию (pattern),
    которая даёт максимальный чистый доход.
    Возвращаем (net_interest, strategy, effective_rate).
    """
    if amount <= 0:
        return 0.0, None, 0.0
    best_net = float("-inf")
    best_strat: Optional[Strategy] = None
    best_rate = 0.0
    for strat in strategies:
        eff = effective_annual_rate(strat, cfg)
        net = net_interest_for_amount_and_rate(amount, eff, cfg, tax_free_limit)
        if net > best_net:
            best_net = net
            best_strat = strat
            best_rate = eff
    return best_net, best_strat, best_rate

def generate_allocations(cfg: Config):
    """
    Генерируем все способы разложить total_amount по people_count
    с шагом cfg.step.
    Возвращает список списков [a1, ..., an].
    """
    units_total = int(round(cfg.total_amount / cfg.step))
    n = cfg.people_count

    def rec(i: int, remaining_units: int, current_units, out):
        if i == n - 1:
            # последний человек получает всё, что осталось
            current_units.append(remaining_units)
            out.append([u * cfg.step for u in current_units])
            current_units.pop()
            return
        for u in range(remaining_units + 1):
            current_units.append(u)
            rec(i + 1, remaining_units - u, current_units, out)
            current_units.pop()

    allocations = []
    rec(0, units_total, [], allocations)
    return allocations

def search_best_config(cfg: Config, strategies: List[Strategy]) -> None:
    tax_free_limit = compute_tax_free_limit_per_person(cfg)
    allocations = generate_allocations(cfg)

    best_total_net = float("-inf")
    best_allocation: Optional[List[float]] = None
    best_per_person_info: Optional[List[Tuple[float, Optional[Strategy], float]]] = None

    for alloc in allocations:
        per_person_info = []
        total_net = 0.0
        for amount in alloc:
            net, strat, eff_rate = best_strategy_for_person_amount(amount, strategies, cfg, tax_free_limit)
            per_person_info.append((net, strat, eff_rate))
            total_net += net

        if total_net > best_total_net:
            best_total_net = total_net
            best_allocation = alloc.copy()
            best_per_person_info = per_person_info

    print("=== Best configuration ===")
    print(f"Total amount: {cfg.total_amount:,.0f} RUB")
    print(f"People: {cfg.people_count}, step: {cfg.step:,.0f} RUB")
    print(f"Key rate: {cfg.key_rate*100:.2f}%, tax-free interest per person: {tax_free_limit:,.0f} RUB")
    print(f"Tax rate: {cfg.tax_rate*100:.1f}%")
    print()
    print(f"Best total net interest: {best_total_net:,.2f} RUB")
    print("Allocation & strategies per person:")
    if best_allocation is None or best_per_person_info is None:
        print("No allocation found (something went wrong).")
        return
    for i, (amount, info) in enumerate(zip(best_allocation, best_per_person_info), start=1):
        net, strat, eff_rate = info
        if strat is None:
            strat_name = "no deposit"
        else:
            strat_name = strat.name
        print(f"Person {i}: amount={amount:,.0f} RUB, strategy={strat_name}, eff_rate={eff_rate*100:.2f}%, net_interest={net:,.2f} RUB")
    print()

def main():
    # Конфиг под твой кейс
    cfg = Config(
        total_amount=9_000_000.0,
        people_count=5,
        key_rate=0.16,      # 16% ключевая
        tax_rate=0.13,      # 13% НДФЛ
        rates_by_term={
            1: 0.15,
            2: 0.15,
            3: 0.147,
            4: 0.145,
            5: 0.145,
            6: 0.145,
            7: 0.14,
            8: 0.14,
            9: 0.14,
            10: 0.13,
            11: 0.13,
            12: 0.13,
        },
        step=500_000.0,   # можно уменьшить до 250_000 или 100_000 — будет точнее, но дольше
    )

    # Стратегии на полный год (сумма месяцев = 12)
    strategies = [
        Strategy("12m_fixed", [12]),
        Strategy("2x6m", [6, 6]),
        Strategy("4x3m", [3, 3, 3, 3]),
        Strategy("monthly_1m", [1] * 12),
        Strategy("3m+9m", [3, 9]),
        Strategy("6m+3m+3m", [6, 3, 3]),
        Strategy("4m+4m+4m", [4, 4, 4]),
    ]

    search_best_config(cfg, strategies)

if __name__ == "__main__":
    main()

