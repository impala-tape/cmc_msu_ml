from typing import List


def hello(name: str | None = None) -> str:
    if name is None or name == "":
        return "Hello!"
    return f"Hello, {name}!"


def int_to_roman(num: int) -> str:
    vals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    res: List[str] = []
    i = 0
    while num > 0:
        cnt, num = divmod(num, vals[i])
        if cnt:
            res.append(syms[i] * cnt)
        i += 1
    return "".join(res)


def longest_common_prefix(strs_input: List[str]) -> str:
    if not strs_input:
        return ""
    arr = [s.strip() for s in strs_input]
    first, last = min(arr), max(arr)
    i = 0
    for c1, c2 in zip(first, last):
        if c1 != c2:
            break
        i += 1
    return first[:i]


def primes() -> int:
    primes_list: List[int] = []
    n = 2
    while True:
        is_prime = True
        for p in primes_list:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes_list.append(n)
            yield n
        n += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int | None = None):
        self.total_sum = int(total_sum)
        self.balance_limit = None if balance_limit is None else int(balance_limit)

    def __call__(self, sum_spent: int):
        if sum_spent > self.total_sum:
            print(f"Not enough money to spend {sum_spent} dollars.")
            raise ValueError
        self.total_sum -= sum_spent
        print(f"You spent {sum_spent} dollars.")

    def __str__(self) -> str:
        return "To learn the balance call balance."

    @property
    def balance(self) -> int:
        if self.balance_limit is not None:
            if self.balance_limit <= 0:
                print("Balance check limits exceeded.")
                raise ValueError
            self.balance_limit -= 1
        return self.total_sum

    def put(self, sum_put: int):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")

    def __add__(self, other: "BankCard") -> "BankCard":
        if not isinstance(other, BankCard):
            return NotImplemented
        if self.balance_limit is None or other.balance_limit is None:
            new_limit = None
        else:
            new_limit = max(self.balance_limit, other.balance_limit)
        return BankCard(self.total_sum + other.total_sum, new_limit)
