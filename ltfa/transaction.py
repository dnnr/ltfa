from dataclasses import dataclass
from decimal import Decimal
import dataclasses

@dataclass
class Transaction:
    value: Decimal
    balance: Decimal
    date: str
    subject: str
    peername: str
    peeraccount: str
    peerbic: str
    account: str
    # Indicates if txn is just between own accounts (i.e., doesn't
    # reflect a change in wealth)
    isneutral: bool

    def __init__(self, **kwargs):
        # Some defaults:
        self.isneutral = None
        self.peeraccount = None
        self.balance = None

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __str__(self):
        slots = [f.name for f in dataclasses.fields(self.__class__)]
        return "Transaction(" + ', '.join([str(k) + "=" + str(getattr(self, k)) for k in slots if hasattr(self, k)]) + ")"

    def shortstr(self):
        return "{}, € {}".format(self.date, self.value)

    def match(self, key, val):
        if key == 'notBefore':
            return self.date >= val
        if key == 'notAfter':
            return self.date <= val

        return getattr(self, key, None) == val
