from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------- Data structures ----------

@dataclass
class Bank:
    name: str
    # term_id -> (duration_years, base_annual_rate)
    # base_annual_rate задаётся для базовой ключевой ставки cfg.key_rate_base
    terms: Dict[str, Tuple[float, float]]


@dataclass
class Strategy:
    """
    Стратегия = последовательность вкладов в ОДНОМ банке,
    суммарно покрывающих примерно 1 календарный год.
    """
    name: str
    bank_name: str
    term_ids: List[str]  # последовательность ключей из Bank.terms


@dataclass
class KeyRateScenario:
    """Сценарий изменения ключевой ставки."""
    prob: float          # вероятность сценария
    new_key_rate: float  # новая ключевая ставка (например, 0.15 = 15%)


@dataclass
class Config:
    total_amount: float        # всего денег, ₽
    people_count: int          # число людей
    key_rate_base: float       # текущая ключевая ставка (база), напр. 0.16
    tax_rate: float            # НДФЛ, напр. 0.13
    key_rate_scenarios: Tuple[KeyRateScenario, ...]
    # чувствительность банков к изменению ключевой: множитель на отношение
    # new_key_rate / key_rate_base. Если 1.0 — ставки банка меняются пропорционально.
    bank_sensitivity: Dict[str, float]
    step: float = 500_000.0    # шаг при разбиении суммы по людям (дискретизация)
    min_years: float = 0.95    # минимум длительности стратегии (в годах)
    max_years: float = 1.05    # максимум длительности стратегии (в годах)
    max_terms_per_strategy: int = 5  # ограничение на число периодов в стратегии (для поиска)


# ---------- Налог и базовые вычисления ----------

def compute_tax_free_limit_per_person(cfg: Config) -> float:
    return cfg.key_rate_base * 1_000_000.0


def per_scenario_rate(
    base_rate: float,
    bank_name: str,
    term_id: str,
    scenario: KeyRateScenario,
    cfg: Config,
) -> float:
    """
    Сценарная ставка банка.

    Газпром здесь НЕ используем (он обрабатывается отдельно),
    Остальные банки: ретарифируются мультипликативно.
    """
    if bank_name == "Gazprom":
        # для Газпрома реальную ставку считаем помесячно отдельно
        raise RuntimeError("per_scenario_rate should not be used for Gazprom")

    sens = cfg.bank_sensitivity.get(bank_name, 1.0)
    factor = (scenario.new_key_rate / cfg.key_rate_base) ** sens
    return base_rate * factor


# ---------- Помесячная симуляция для Газпрома ----------
def gazprom_term_growth_monthly(
    duration_years: float,
    base_rate: float,
    cfg: Config,
    scen: KeyRateScenario,
) -> float:
    """
    Ростовый множитель для ОДНОГО срока вклада Газпрома
    при помесячной капитализации, если считаем, что в данном
    сценарии ключевая уже стала scen.new_key_rate и такой
    остаётся весь срок вклада.

    Модель:
      - номинальная ставка вклада в этом сценарии:
            r_nom = scen.new_key_rate + margin,
        где margin = base_rate - cfg.key_rate_base (фиксированный спред);
      - капитализация каждый месяц:
            factor = Π (1 + r_nom/12).
    """
    total_months = max(1, int(round(duration_years * 12.0)))
    margin = base_rate - cfg.key_rate_base
    r_nom = scen.new_key_rate + margin        # годовая ставка в этом сценарии
    monthly_rate = r_nom / 12.0

    factor = (1.0 + monthly_rate) ** total_months
    return factor


def strategy_growth_factor(
    strategy: Strategy,
    banks: Dict[str, Bank],
    cfg: Config,
) -> Tuple[float, float]:
    """
    Ожидаемый рост капитала по стратегии с учётом сценариев ключевой
    и ежемесячной капитализации.

    Возвращает:
      growth_factor_exp — E[factor_s] по сценариям;
      total_years       — суммарная длительность стратегии (в годах).
    """
    if strategy.bank_name not in banks:
        raise ValueError(f"Unknown bank in strategy {strategy.name}: {strategy.bank_name}")

    bank = banks[strategy.bank_name]

    for term_id in strategy.term_ids:
        if term_id not in bank.terms:
            raise ValueError(
                f"Unknown term_id {term_id} in bank {bank.name} for strategy {strategy.name}"
            )

    total_years = sum(bank.terms[tid][0] for tid in strategy.term_ids)
    if total_years <= 0:
        return 1.0, 0.0

    growth_factor_exp = 0.0

    for scen in cfg.key_rate_scenarios:
        factor_s = 1.0

        for j, term_id in enumerate(strategy.term_ids):
            duration_years, base_rate = bank.terms[term_id]

            if strategy.bank_name == "Gazprom":
                # Газпром: ставка = (ключевая в сценарии + спред),
                # считаем, что ключевая уже scen.new_key_rate весь срок.
                growth_term = gazprom_term_growth_monthly(duration_years, base_rate, cfg, scen)

            else:
                # Остальные банки:
                #   - первый вклад: фиксированная текущая ставка (base_rate),
                #   - последующие: ретарифируем под сценарий через per_scenario_rate().
                if j == 0:
                    r_s = base_rate
                else:
                    r_s = per_scenario_rate(base_rate, strategy.bank_name, term_id, scen, cfg)

                months = duration_years * 12.0
                monthly_rate = r_s / 12.0
                growth_term = (1.0 + monthly_rate) ** months

            factor_s *= growth_term

        growth_factor_exp += scen.prob * factor_s

    return growth_factor_exp, total_years




def effective_annual_rate(strategy: Strategy, banks: Dict[str, Bank], cfg: Config) -> float:
    factor_exp, total_years = strategy_growth_factor(strategy, banks, cfg)
    if total_years <= 0:
        return 0.0
    r_eff = factor_exp ** (1.0 / total_years) - 1.0
    return r_eff


def net_interest_for_amount_and_strategy(
    amount: float,
    strategy: Strategy,
    banks: Dict[str, Bank],
    cfg: Config,
    tax_free_limit: float,
) -> Tuple[float, float, float, float]:
    """
    Возвращает:
      net        — чистые проценты после НДФЛ,
      eff_rate   — эф. годовая ставка (ожидаемая),
      taxable    — налоговая база (проценты сверх лимита),
      tax        — НДФЛ.
    """
    if amount <= 0:
        return 0.0, 0.0, 0.0, 0.0

    eff_rate = effective_annual_rate(strategy, banks, cfg)
    interest = amount * eff_rate
    taxable = max(0.0, interest - tax_free_limit)
    tax = taxable * cfg.tax_rate
    net = interest - tax
    return net, eff_rate, taxable, tax


def best_strategy_for_person_amount(
    amount: float,
    strategies: List[Strategy],
    banks: Dict[str, Bank],
    cfg: Config,
    tax_free_limit: float,
):
    """
    Для фиксированной суммы подбирает стратегию с максимальным net.
    Возвращает:
      net, chosen_strategy, eff_rate, taxable, tax
    """
    if amount <= 0:
        return 0.0, None, 0.0, 0.0, 0.0

    best_net = float("-inf")
    best_strat = None
    best_rate = 0.0
    best_taxable = 0.0
    best_tax = 0.0

    for strat in strategies:
        net, eff_rate, taxable, tax = net_interest_for_amount_and_strategy(
            amount, strat, banks, cfg, tax_free_limit
        )
        if net > best_net:
            best_net = net
            best_strat = strat
            best_rate = eff_rate
            best_taxable = taxable
            best_tax = tax

    return best_net, best_strat, best_rate, best_taxable, best_tax


# ---------- Генерация разбиения суммы между людьми ----------

def generate_allocations(cfg: Config):
    """
    Возвращает все разбиения total_amount на people_count частей
    с шагом cfg.step (в рублях).
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


# ---------- Автогенерация стратегий ----------

def auto_generate_strategies_for_bank(
    bank: Bank,
    cfg: Config,
) -> List[Strategy]:
    """
    Автогенерация стратегий для одного банка.

    Комбинации сроков с повторениями, суммарная длительность которых
    лежит в [min_years, max_years], с ограничением max_terms_per_strategy.
    Порядок сроков неубывающий, чтобы не плодить перестановки.
    """
    term_items = sorted(bank.terms.items(), key=lambda kv: kv[1][0])
    strategies: List[Strategy] = []

    def rec(start_idx: int, current: List[str], years_sum: float):
        if cfg.min_years <= years_sum <= cfg.max_years and current:
            name = f"{bank.name}_" + "+".join(current)
            strategies.append(Strategy(name=name, bank_name=bank.name, term_ids=current.copy()))
            if len(current) >= cfg.max_terms_per_strategy:
                return

        if years_sum > cfg.max_years or len(current) >= cfg.max_terms_per_strategy:
            return

        for idx in range(start_idx, len(term_items)):
            term_id, (dur, _rate) = term_items[idx]
            new_years_sum = years_sum + dur
            if new_years_sum > cfg.max_years + 1e-9:
                break
            current.append(term_id)
            rec(idx, current, new_years_sum)
            current.pop()

    rec(0, [], 0.0)
    return strategies


def auto_generate_all_strategies(banks: Dict[str, Bank], cfg: Config) -> List[Strategy]:
    all_strats: List[Strategy] = []
    for bank in banks.values():
        bank_strats = auto_generate_strategies_for_bank(bank, cfg)
        all_strats.extend(bank_strats)
    return all_strats


# ---------- ТОП стратегий ----------

def print_top_strategies(strategies: List[Strategy], banks: Dict[str, Bank], cfg: Config, top_n: int = 5) -> None:
    print("=== ТОП стратегий по эффективной годовой ставке (ожидаемой) ===")

    # считаем эффективную ставку для всех стратегий
    stats = []
    for strat in strategies:
        r_eff = effective_annual_rate(strat, banks, cfg)
        stats.append((strat, r_eff))

    # общий топ
    stats_sorted = sorted(stats, key=lambda x: x[1], reverse=True)
    print(f"\n--- Общий ТОП-{top_n} ---")
    for i, (s, r) in enumerate(stats_sorted[:top_n], start=1):
        print(f"{i}. {s.bank_name:8s} | {s.name:30s} -> эф.ставка = {r*100:.3f}%")

    # по каждому банку
    print()
    banks_set = sorted({s.bank_name for s, _ in stats_sorted})
    for bank_name in banks_set:
        bank_stats = [(s, r) for (s, r) in stats_sorted if s.bank_name == bank_name]
        if not bank_stats:
            continue
        print(f"--- Банк {bank_name} — ТОП-{top_n} ---")
        for i, (s, r) in enumerate(bank_stats[:top_n], start=1):
            print(f"{i}. {s.name:30s} -> эф.ставка = {r*100:.3f}%")
        print()


# ---------- Глобальный поиск лучшей конфигурации ----------

def search_best_config(cfg: Config, banks: Dict[str, Bank], strategies: List[Strategy]) -> None:
    tax_free_limit = compute_tax_free_limit_per_person(cfg)
    allocations = generate_allocations(cfg)

    best_total_net = float("-inf")
    best_total_tax = 0.0
    best_allocation = None
    best_per_person_info = None

    for alloc in allocations:
        per_person_info = []
        total_net = 0.0
        total_tax = 0.0

        for amount in alloc:
            net, strat, eff_rate, taxable, tax = best_strategy_for_person_amount(
                amount, strategies, banks, cfg, tax_free_limit
            )
            per_person_info.append((amount, net, strat, eff_rate, taxable, tax))
            total_net += net
            total_tax += tax

        if total_net > best_total_net:
            best_total_net = total_net
            best_total_tax = total_tax
            best_allocation = alloc.copy()
            best_per_person_info = per_person_info

    print("=== ЛУЧШАЯ КОНФИГУРАЦИЯ ===")
    print(f"Всего денег: {cfg.total_amount:,.0f} ₽")
    print(f"Людей: {cfg.people_count}, шаг по сумме: {cfg.step:,.0f} ₽")
    tax_free_limit = compute_tax_free_limit_per_person(cfg)
    print(f"Базовая ключевая ставка: {cfg.key_rate_base*100:.2f}%, "
          f"безналоговый доход на человека: {tax_free_limit:,.0f} ₽")
    print(f"Ставка НДФЛ: {cfg.tax_rate*100:.1f}%")
    print("Сценарии ключевой ставки:")
    for scen in cfg.key_rate_scenarios:
        print(f"  prob={scen.prob:.2f}, key_rate={scen.new_key_rate*100:.2f}%")
    print()

    print(f"Максимальный суммарный ОЖИДАЕМЫЙ ЧИСТЫЙ доход: {best_total_net:,.2f} ₽")
    if cfg.people_count > 0:
        print(f"В среднем в месяц (если горизонт ~1 год): {best_total_net/12:,.2f} ₽")
    print(f"Суммарный НДФЛ (ожидаемо): {best_total_tax:,.2f} ₽")
    print("\nРаспределение по людям и выбранные стратегии:")

    if best_allocation is None or best_per_person_info is None:
        print("Что-то пошло не так, конфигурация не найдена.")
        return

    for i, info in enumerate(best_per_person_info, start=1):
        amount, net, strat, eff_rate, taxable, tax = info
        if strat is None:
            strat_name = "нет вклада"
            bank_name = "-"
        else:
            strat_name = strat.name
            bank_name = strat.bank_name
        print(
            f"Человек {i}: сумма={amount:,.0f} ₽, банк={bank_name}, "
            f"стратегия={strat_name}, эф.ставка={eff_rate*100:.3f}%, "
            f"чистые прибыль={net:,.2f} ₽, НДФЛ={tax:,.2f} ₽"
        )
    print()


# ---------- Настройка банков и сценариев ----------

def build_default_banks() -> Dict[str, Bank]:
    # Тинькофф (примерная сетка)
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

    # Alfa Only
    alfa = Bank(
        name="AlfaOnly",
        terms={
            "62d":   (62/365, 0.1471),
            "92d":   (92/365, 0.1511),
            "123d":  (123/365, 0.1511),
            "153d":  (153/365, 0.1483),
            "184d":  (184/365, 0.1430),
            "276d":  (276/365, 0.1400),
            "365d":  (365/365, 0.1375),
        },
    )

    # Газпромбанк, вклад с ключевой + спред (таблица из скрина, при key_rate_base=16%)
    gazprom = Bank(
        name="Gazprom",
        terms={
            "181d":  (181/365, 0.16 - 0.007),  # -0.7 п.п. к ключевой
            "213d":  (213/365, 0.16 - 0.006),  # -0.6
            "367d":  (367/365, 0.16 + 0.000),  # 0
            "548d":  (548/365, 0.16 + 0.003),  # +0.3
            "731d":  (731/365, 0.16 + 0.010),  # +1.0
            "1095d": (1095/365, 0.16 + 0.020), # +2.0
        },
    )

    return {b.name: b for b in (tinkoff, alfa, gazprom)}


def build_default_config() -> Config:
    scenarios = (
        KeyRateScenario(prob=0.14, new_key_rate=0.15),
        KeyRateScenario(prob=0.80, new_key_rate=0.13),
        #KeyRateScenario(prob=0.01, new_key_rate=0.17),
        #KeyRateScenario(prob=0.05, new_key_rate=0.165),
    )
    #нормализируем вероятности
    scenarios = [KeyRateScenario(prob=x.prob/sum(y.prob for y in scenarios),new_key_rate=x.new_key_rate) for x in scenarios]

    bank_sens = {
        "Tinkoff": 1.0,
        "AlfaOnly": 1.0,
        "Gazprom": 1.0,
    }

    cfg = Config(
        total_amount=9_000_000.0,
        people_count=2,
        key_rate_base=0.16,
        tax_rate=0.13,
        key_rate_scenarios=scenarios,
        bank_sensitivity=bank_sens,
        step=500_000.0,
        min_years=0.90,
        max_years=1.2,
        max_terms_per_strategy=12,
    )
    return cfg


def main():
    cfg = build_default_config()
    banks = build_default_banks()

    print("=== Автогенерация стратегий ===")
    strategies = auto_generate_all_strategies(banks, cfg)
    print(f"Сгенерировано стратегий: {len(strategies)}\n")

    # ТОП-5 стратегий (общий и по банкам)
    print_top_strategies(strategies, banks, cfg, top_n=5)
    print()

    # Поиск лучшей конфигурации по людям
    search_best_config(cfg, banks, strategies)


if __name__ == "__main__":
    main()
