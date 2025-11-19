
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable


# ---------- Data structures ----------

@dataclass
class Bank:
    name: str
    # term_id -> (duration_years, base_annual_rate)
    # base_annual_rate задаётся для текущей ключевой ставки (cfg.key_rate_base)
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
    max_terms_per_strategy: int = 12  # ограничение на число периодов в стратегии (для поиска)


# ---------- Налог и базовые вычисления ----------

def compute_tax_free_limit_per_person(cfg: Config) -> float:
    """
    По действующему правилу:
    налоговая база по процентам = max(0, проценты_за_год - key_rate * 1_000_000).
    Здесь используем базовую ключевую ставку на год (cfg.key_rate_base),
    иначе пришлось бы делать сценарный налог, что уже избыточно.

    => безналоговый годовой доход по процентам на человека:
       key_rate_base * 1_000_000
    """
    return cfg.key_rate_base * 1_000_000.0


def per_scenario_rate(
    base_rate: float,
    bank_name: str,
    scenario: KeyRateScenario,
    cfg: Config,
) -> float:
    """
    Сценарная ставка банка при изменении ключевой.

    Модель: ставка банка меняется пропорционально отношению
            new_key_rate / key_rate_base с коэффициентом чувствительности.
    """
    sens = cfg.bank_sensitivity.get(bank_name, 1.0)
    factor = (scenario.new_key_rate / cfg.key_rate_base) ** sens
    return base_rate * factor


def strategy_growth_factor(
    strategy: Strategy,
    banks: Dict[str, Bank],
    cfg: Config,
) -> Tuple[float, float]:
    """
    Считаем ОЖИДАЕМЫЙ рост капитала по стратегии с учётом сценариев ключевой ставки
    и ежемесячной капитализации.

    growth_factor_exp = E[ growth_factor_scenario ]
    total_years       = суммарная длительность стратегии (в годах)

    Для каждого сценария s:
      - пересчитываем ставки банка на каждую ступень стратегии,
      - считаем сложный процент: последовательное применение (1 + r_s/12)^months,
      - получаем factor_s,
    Затем:
      growth_factor_exp = Σ_s prob_s * factor_s.
    """
    if strategy.bank_name not in banks:
        raise ValueError(f"Unknown bank in strategy {strategy.name}: {strategy.bank_name}")

    bank = banks[strategy.bank_name]

    # Проверка term_ids
    for term_id in strategy.term_ids:
        if term_id not in bank.terms:
            raise ValueError(f"Unknown term_id {term_id} in bank {bank.name} for strategy {strategy.name}")

    # Общая длительность (в годах) не зависит от сценария
    total_years = sum(bank.terms[tid][0] for tid in strategy.term_ids)

    # Если стратегия пустая, роста нет
    if total_years <= 0:
        return 1.0, 0.0

    # Для каждого сценария считаем свой growth_factor_s
    growth_factor_exp = 0.0
    for scen in cfg.key_rate_scenarios:
        factor_s = 1.0
        for term_id in strategy.term_ids:
            duration_years, base_rate = bank.terms[term_id]
            months = duration_years * 12.0
            # Сценарная ставка банка
            r_s = per_scenario_rate(base_rate, strategy.bank_name, scen, cfg)
            monthly_rate = r_s / 12.0
            growth_term = (1.0 + monthly_rate) ** months
            factor_s *= growth_term
        growth_factor_exp += scen.prob * factor_s

    return growth_factor_exp, total_years


def effective_annual_rate(strategy: Strategy, banks: Dict[str, Bank], cfg: Config) -> float:
    """
    Переводим ожидаемый growth_factor в эффективную годовую ставку.
    growth_factor_exp — ожидаемый множитель за total_years лет.

    r_eff такой, что: (1 + r_eff) ** total_years = growth_factor_exp
    """
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
) -> Tuple[float, float]:
    """
    Чистые проценты после НДФЛ для одного человека при данной стратегии.

    amount: сколько на этого человека повесили (во всех банках в сумме).
    Стратегия считается повторяющейся в течение года (горизонт ~1 год).

    Возвращаем:
      net_interest — ожидаемая чистая прибыль за год,
      eff_rate     — ожидаемая эффективная годовая ставка.
    """
    if amount <= 0:
        return 0.0, 0.0

    eff_rate = effective_annual_rate(strategy, banks, cfg)
    interest = amount * eff_rate
    taxable = max(0.0, interest - tax_free_limit)
    tax = taxable * cfg.tax_rate
    net = interest - tax
    return net, eff_rate


# ---------- Поиск лучшей стратегии для фиксированной суммы на человека ----------

def best_strategy_for_person_amount(
    amount: float,
    strategies: List[Strategy],
    banks: Dict[str, Bank],
    cfg: Config,
    tax_free_limit: float,
):
    """
    Для фиксированной суммы amount на одного человека перебираем все стратегии (по всем банкам)
    и выбираем максимальный ОЖИДАЕМЫЙ net_interest.
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


# ---------- Автогенерация стратегий ----------

def auto_generate_strategies_for_bank(
    bank: Bank,
    cfg: Config,
) -> List[Strategy]:
    """
    Автоматически генерируем стратегии для одного банка.
    Идея:
      - есть набор сроков с длительностями (в годах),
      - хотим все комбинации с повторениями, у которых суммарная длительность
        в [cfg.min_years, cfg.max_years],
      - ограничиваемся максимум cfg.max_terms_per_strategy периодами.

    Чтобы не дублировать перестановки (6m+3m+3m и 3m+6m+3m — одно и то же по эффекту),
    фиксируем порядок: добавляем сроки только с неубывающим индексом.

    Имена стратегий строим автоматически как
      "{bank.name}_" + "+".join(term_ids)
    """
    term_items = sorted(bank.terms.items(), key=lambda kv: kv[1][0])  # сортируем по duration_years
    strategies: List[Strategy] = []

    def rec(start_idx: int, current: List[str], years_sum: float):
        # Если уже внутри допустимого окна — записываем стратегию
        if cfg.min_years <= years_sum <= cfg.max_years and current:
            name = f"{bank.name}_" + "+".join(current)
            strategies.append(Strategy(name=name, bank_name=bank.name, term_ids=current.copy()))
            # продолжаем поиск, чтобы найти более длинные варианты (если позволено)
            if len(current) >= cfg.max_terms_per_strategy:
                return

        # Если уже превысили максимум по длительности или количеству периодов — стоп
        if years_sum > cfg.max_years or len(current) >= cfg.max_terms_per_strategy:
            return

        # Перебираем возможные следующие сроки, обеспечивая неубывающий порядок
        for idx in range(start_idx, len(term_items)):
            term_id, (dur, _rate) = term_items[idx]
            new_years_sum = years_sum + dur
            if new_years_sum > cfg.max_years + 1e-9:
                # дальше сроки только длиннее — можно прервать цикл
                break
            current.append(term_id)
            rec(idx, current, new_years_sum)
            current.pop()

    rec(0, [], 0.0)
    return strategies


def auto_generate_all_strategies(banks: Dict[str, Bank], cfg: Config) -> List[Strategy]:
    """
    Запускаем автогенерацию стратегий для всех банков.
    """
    all_strats: List[Strategy] = []
    for bank in banks.values():
        bank_strats = auto_generate_strategies_for_bank(bank, cfg)
        all_strats.extend(bank_strats)
    return all_strats


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
    print(f"Базовая ключевая ставка: {cfg.key_rate_base*100:.2f}%, безналоговый доход на человека: {tax_free_limit:,.0f} ₽")
    print(f"Ставка НДФЛ: {cfg.tax_rate*100:.1f}%")
    print("Сценарии ключевой ставки:")
    for scen in cfg.key_rate_scenarios:
        print(f"  prob={scen.prob:.2f}, key_rate={scen.new_key_rate*100:.2f}%")
    print()

    print(f"Максимальный суммарный ОЖИДАЕМЫЙ ЧИСТЫЙ доход: {best_total_net:,.2f} ₽")
    if cfg.people_count > 0:
        print(f"В среднем в месяц (если горизонт ~1 год): {best_total_net/12:,.2f} ₽")
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
            f"стратегия={strat_name}, эф.ставка={eff_rate*100:.3f}%, "
            f"чистые проценты={net:,.2f} ₽"
        )
    print()


# ---------- Настройка банков и сценариев под твой кейс ----------

def build_default_banks() -> Dict[str, Bank]:
    # Банк 1: Тинькофф (примерная сетка по месяцам)
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

    # Банк 2: Alfa Only (по дням, до 1 года)
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
        },
    )

    return {b.name: b for b in (tinkoff, alfa)}


def build_default_config() -> Config:
    # Пример прогнозов ключевой ставки:
    # допустим, сейчас 16%, и мы считаем, что:
    #  - 60% вероятность, что упадёт до 15%
    #  - 30% вероятность, что упадёт до 14%
    #  - 10% вероятность, что останется 16%
    scenarios = (
        KeyRateScenario(prob=0.60, new_key_rate=0.15),
        KeyRateScenario(prob=0.05, new_key_rate=0.17),
        KeyRateScenario(prob=0.35, new_key_rate=0.16),
    )

    # Чувствительность банков к изменению ключевой
    bank_sens = {
        "Tinkoff": 1.0,
        "AlfaOnly": 1.0,
    }

    cfg = Config(
        total_amount=9_000_000.0,
        people_count=7,
        key_rate_base=0.16,       # текущая ключевая ставка 16%
        tax_rate=0.13,            # НДФЛ 13%
        key_rate_scenarios=scenarios,
        bank_sensitivity=bank_sens,
        step=500_000.0,           # шаг по 500k
        min_years=0.95,
        max_years=1.05,
        max_terms_per_strategy=12, # до 5 периодов в год на одного человека
    )
    return cfg


def main():
    cfg = build_default_config()
    banks = build_default_banks()

    print("=== Автогенерация стратегий ===")
    strategies = auto_generate_all_strategies(banks, cfg)
    print(f"Сгенерировано стратегий: {len(strategies)}")
    # Можно раскомментировать для просмотра топ-нескольких стратегий и их eff_rate
    # for strat in strategies[:15]:
    #     r_eff = effective_annual_rate(strat, banks, cfg)
    #     print(f"{strat.bank_name:9s} | {strat.name:30s} -> r_eff = {r_eff*100:.3f}%")

    print()
    search_best_config(cfg, banks, strategies)


if __name__ == "__main__":
    main()
