
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ---------- Data structures ----------

@dataclass
class Bank:
    name: str
    # term_id -> (duration_years, annual_rate)
    # например: "6m": (6/12, 0.145)
    terms: Dict[str, Tuple[float, float]]


@dataclass
class Strategy:
    """
    Стратегия = последовательность вкладов в ОДНОМ банке,
    покрывающих примерно 1 календарный год.
    """
    name: str
    bank_name: str
    term_ids: List[str]  # последовательность ключей из Bank.terms


@dataclass
class Config:
    total_amount: float        # всего денег, RUB
    people_count: int          # число людей
    key_rate: float            # ключевая ставка (0.16 = 16%)
    tax_rate: float            # НДФЛ (0.13 = 13%)
    step: float = 500_000.0    # шаг при разбиении суммы по людям (дискретизация)


# ---------- Налог и эффективная ставка ----------

def compute_tax_free_limit_per_person(cfg: Config) -> float:
    """
    По действующему правилу:
    налоговая база по процентам = max(0, проценты_за_год - key_rate * 1_000_000).
    Значит key_rate * 1e6 — это безналогительный годовой доход по процентам.
    """
    return cfg.key_rate * 1_000_000.0


def strategy_growth_factor(strategy: Strategy, banks: Dict[str, Bank]) -> Tuple[float, float]:
    """
    Реальная формула: во всех банках РФ проценты капитализируются ежемесячно.
    Поэтому рост за каждый срок = (1 + r/12)^(months).
    Далее рост перемножается между сроками.

    Возвращает:
      growth_factor – итоговое увеличение суммы
      total_years   – общая длительность стратегии
    """
    if strategy.bank_name not in banks:
        raise ValueError(f"Unknown bank in strategy {strategy.name}: {strategy.bank_name}")

    bank = banks[strategy.bank_name]
    factor = 1.0
    total_years = 0.0

    for term_id in strategy.term_ids:
        if term_id not in bank.terms:
            raise ValueError(f"Unknown term_id {term_id} in bank {bank.name} for strategy {strategy.name}")

        duration_years, annual_rate = bank.terms[term_id]

        # сколько месяцев длится срок (может быть дробным, но это нормально)
        months = duration_years * 12

        # ежемесячная ставка
        monthly_rate = annual_rate / 12.0

        # рост за срок со сложным процентом
        growth = (1 + monthly_rate) ** months

        factor *= growth
        total_years += duration_years

    return factor, total_years



def effective_annual_rate(strategy: Strategy, banks: Dict[str, Bank]) -> float:
    """
    Переводим growth_factor стратегии в эффективную годовую ставку.
    Предполагаем, что стратегия покрывает примерно 1 год (total_years ≈ 1).
    Если чуть отличается (из-за 62/92 дней и т.п.), то нормируем.
    """
    factor, total_years = strategy_growth_factor(strategy, banks)
    if total_years <= 0:
        return 0.0
    # годовая эффективная ставка r_eff такова, что:
    # (1 + r_eff) ** total_years = factor  ->  r_eff = factor**(1/total_years) - 1
    r_eff = factor ** (1.0 / total_years) - 1.0
    return r_eff


def net_interest_for_amount_and_strategy(
    amount: float,
    strategy: Strategy,
    banks: Dict[str, Bank],
    cfg: Config,
    tax_free_limit: float,
) -> Tuple[float, float]:
    """
    Чистые проценты после НДФЛ для одного человека при данной стратегии.

    amount: сколько на него положили за год (предполагаем, что весь год крутится по этой стратегии)
    Возвращаем:
      net_interest (чистая прибыль),
      eff_rate (эффективная годовая ставка, уже с учётом сложного процента внутри года).
    """
    if amount <= 0:
        return 0.0, 0.0

    eff_rate = effective_annual_rate(strategy, banks)
    interest = amount * eff_rate
    taxable = max(0.0, interest - tax_free_limit)
    tax = taxable * cfg.tax_rate
    net = interest - tax
    return net, eff_rate


# ---------- Поиск лучшей стратегии для человека при фиксированной сумме ----------

def best_strategy_for_person_amount(
    amount: float,
    strategies: List[Strategy],
    banks: Dict[str, Bank],
    cfg: Config,
    tax_free_limit: float,
):
    """
    Для фиксированной суммы amount на одного человека перебираем все стратегии (по всем банкам)
    и выбираем максимальный net_interest.
    Возвращаем (net_interest, chosen_strategy, eff_rate).
    """
    if amount <= 0:
        return 0.0, None, 0.0

    best_net = float("-inf")
    best_strat = None
    best_rate = 0.0

    for strat in strategies:
        net, eff_rate = net_interest_for_amount_and_strategy(amount, strat, banks, cfg, tax_free_limit)
        if net > best_net:
            best_net = net
            best_strat = strat
            best_rate = eff_rate

    return best_net, best_strat, best_rate


# ---------- Генерация разбиения суммы между людьми ----------

def generate_allocations(cfg: Config):
    """
    Разбиваем total_amount на people_count частей с шагом cfg.step.
    Возвращает список разбиений: [[a1, a2, ..., aN], ...],
    где сумма ai == total_amount (с точностью до шага).
    """
    units_total = int(round(cfg.total_amount / cfg.step))
    n = cfg.people_count

    def rec(i: int, remaining_units: int, current_units, out):
        if i == n - 1:
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


# ---------- Глобальный поиск лучшей конфигурации ----------

def search_best_config(cfg: Config, banks: Dict[str, Bank], strategies: List[Strategy]) -> None:
    tax_free_limit = compute_tax_free_limit_per_person(cfg)
    allocations = generate_allocations(cfg)

    best_total_net = float("-inf")
    best_allocation = None
    best_per_person_info = None

    for alloc in allocations:
        per_person_info = []
        total_net = 0.0

        for amount in alloc:
            net, strat, eff_rate = best_strategy_for_person_amount(
                amount, strategies, banks, cfg, tax_free_limit
            )
            per_person_info.append((amount, net, strat, eff_rate))
            total_net += net

        if total_net > best_total_net:
            best_total_net = total_net
            best_allocation = alloc.copy()
            best_per_person_info = per_person_info

    # Вывод результата
    print("=== ЛУЧШАЯ КОНФИГУРАЦИЯ ===")
    print(f"Всего денег: {cfg.total_amount:,.0f} ₽")
    print(f"Людей: {cfg.people_count}, шаг по сумме: {cfg.step:,.0f} ₽")
    print(f"Ключевая ставка: {cfg.key_rate*100:.2f}%, безналоговый доход на человека: {tax_free_limit:,.0f} ₽")
    print(f"Ставка НДФЛ: {cfg.tax_rate*100:.1f}%")
    print()
    print(f"Максимальный суммарный ЧИСТЫЙ доход: {best_total_net:,.2f} ₽ | в месяц {best_total_net/12:,.2f}")
    print("Распределение по людям и выбранные стратегии:")
    if best_allocation is None or best_per_person_info is None:
        print("Что-то пошло не так, конфигурация не найдена.")
        return

    for i, info in enumerate(best_per_person_info, start=1):
        amount, net, strat, eff_rate = info
        if strat is None:
            strat_name = "нет вклада"
            bank_name = "-"
        else:
            strat_name = strat.name
            bank_name = strat.bank_name
        print(
            f"Человек {i}: сумма={amount:,.0f} ₽, банк={bank_name}, "
            f"стратегия={strat_name}, эффективная ставка={eff_rate*100:.3f}%, "
            f"чистая прибыль={net:,.2f} ₽"
        )
    print()


# ---------- Пример настройки банков и стратегий под твой кейс ----------

def build_default_banks_and_strategies():
    # Банк 1: Тинькофф, месячные сроки (как раньше)
    tinkoff = Bank(
        name="Tinkoff",
        terms={
            "1m":  (1/12, 0.15),
            "2m":  (2/12, 0.15),
            "3m":  (3/12, 0.147),
            "4m":  (4/12, 0.145),
            "5m":  (5/12, 0.145),
            "6m":  (6/12, 0.145),
            "7m":  (7/12, 0.14),
            "8m":  (8/12, 0.14),
            "9m":  (9/12, 0.14),
            "10m": (10/12, 0.13),
            "11m": (11/12, 0.13),
            "12m": (12/12, 0.13),
        },
    )

    # Банк 2: Alfa Only, сроки в днях (берём только до года, чтобы было честное сравнение за год)
    alfa = Bank(
        name="AlfaOnly",
        terms={
            # duration_years = days / 365
            "62d":   (62/365, 0.1471),
            "92d":   (92/365, 0.1511),
            "123d":  (123/365, 0.1511),
            "153d":  (153/365, 0.1483),
            "184d":  (184/365, 0.1430),
            "276d":  (276/365, 0.1400),
            "365d":  (365/365, 0.1375),
            # длинные сроки можно добавить позже, если будем считать горизонты >1 года
            # "550d":  (550/365, 0.0979),
            # "730d":  (730/365, 0.0975),
            # "1095d": (1095/365, 0.0719),
        },
    )

    banks = {b.name: b for b in (tinkoff, alfa)}

    # Стратегии: каждый человек за год выбирает ОДИН банк и крутит в нём деньги по шаблону
    strategies: List[Strategy] = []

    # --- стратегии Тинькофф ---
    strategies.extend([
        Strategy("TCS_12m", "Tinkoff", ["12m"]),
        Strategy("TCS_2x6m", "Tinkoff", ["6m", "6m"]),
        Strategy("TCS_4x3m", "Tinkoff", ["3m", "3m", "3m", "3m"]),
        Strategy("TCS_monthly_1m", "Tinkoff", ["1m"] * 12),
        Strategy("TCS_3m+9m", "Tinkoff", ["3m", "9m"]),
        Strategy("TCS_6m+3m+3m", "Tinkoff", ["6m", "3m", "3m"]),
        Strategy("TCS_4m+4m+4m", "Tinkoff", ["4m", "4m", "4m"]),
    ])

    # --- стратегии Alfa Only ---
    # Берём несколько разумных комбинаций, близких к году
    strategies.extend([
        Strategy("Alfa_365d", "AlfaOnly", ["365d"]),  # ровно год
        Strategy("Alfa_184d+184d", "AlfaOnly", ["184d", "184d"]),  # ~368 дней
        Strategy("Alfa_92d*4", "AlfaOnly", ["92d", "92d", "92d", "92d"]),  # ~368 дней
        Strategy("Alfa_123d+123d+123d", "AlfaOnly", ["123d", "123d", "123d"]),  # ~369 дней
        Strategy("Alfa_153d+153d", "AlfaOnly", ["153d", "153d"]),  # ~306 дней
        Strategy("Alfa_62d*6", "AlfaOnly", ["62d"] * 6),  # ~372 дней
    ])

    return banks, strategies


def main():
    # Конфиг под твой кейс
    cfg = Config(
        total_amount=9_000_000.0,
        people_count=5,
        key_rate=0.16,      # 16% ключевая
        tax_rate=0.13,      # 13% НДФЛ
        step=500_000.0,     # шаг по разбиению суммы по людям
    )

    banks, strategies = build_default_banks_and_strategies()

    # Можно раскомментировать для проверки стратегий по эффективной ставке
    # for strat in strategies:
    #     r_eff = effective_annual_rate(strat, banks)
    #     print(f"{strat.bank_name:9s} | {strat.name:20s} -> r_eff = {r_eff*100:.3f}%")

    search_best_config(cfg, banks, strategies)


if __name__ == "__main__":
    main()
