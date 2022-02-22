# Import the environment
import numpy
import math
import datetime
import matplotlib
import scipy.optimize
import pandas
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, title
from numpy import log

data = pandas.read_excel('data.xlsx', sheet_name='data')
data['MATURITY_DATE'] = pandas.to_datetime(data['MATURITY_DATE'])
data['ISSUE_DATE'] = pandas.to_datetime(data['ISSUE_DATE'])
data_selected = data.loc[[8, 11, 13, 4, 5, 25, 26, 28, 29,31],]


class bond:
    def __init__(self, ISIN, ten_day_price, maturity_date, coupon_rate, time_to_maturity):
        self.ISIN = ISIN
        self.ten_day_price = ten_day_price
        self.maturity_date = maturity_date
        self.coupon_rate = coupon_rate
        self.time_to_maturity = time_to_maturity
        # As the coupon are paid semi-annually
        self.periods = self.time_to_maturity * 2

    def bond_ytm(self, price, par=100, freq=2, x0=0.05):
        coupon_payment = self.coupon_rate * par / freq
        time = [(j + 1) / freq for j in range(int(self.periods))]
        ytm = lambda y: sum([coupon_payment / (1 + y / freq) ** (freq * t) for t in time]) + \
                        par / (1 + y / freq) ** (freq * self.time_to_maturity) - price
        return scipy.optimize.newton(ytm, x0)


price_list = data_selected.iloc[:, 4:].values.tolist()
bonds = []
for i in range(10):
    bonds.append(bond(ISIN=data_selected.iloc[i, 0],
                      ten_day_price=price_list[i],
                      maturity_date=data_selected.iloc[i, 2],
                      coupon_rate=data_selected.iloc[i, 3],
                      time_to_maturity=(i + 1) * 0.5))

ytm_dict = {}
dates = ['1-10', '1-11', '1-12', '1-13', '1-14', '1-17', '1-18', '1-19', '1-20', '1-21']
for i in range(len(dates)):
    ytm_dict[dates[i]] = []
    for bond in bonds:
        bond_ytm = [bond.bond_ytm(price=bond.ten_day_price[i])]
        ytm_dict[dates[i]].append(bond_ytm)
x_label = ['22/2', '22/8', '23/2', '23/8', '24/3', '24/9', '25/3', '25/9', '26/3', '26/9']
plt.figure(figsize=(10, 6))
plt.xlabel('time to maturity')
plt.ylabel('ytm')
plt.title('Five year yield curve')

for bond in ytm_dict:
    plt.plot(x_label, ytm_dict[bond])
plt.legend(dates, loc="lower right")

# save first then show
plt.savefig('E:yield_curve.png')
plt.show()
plt.close()


class BootstrapYieldCurve(object):

    def __init__(self):
        self.zero_rates = dict()
        self.instruments = dict()

    def add_instrument(self, par, T, coup, price, compounding_freq=2):
        self.instruments[T] = (par, coup, price, compounding_freq)

    def get_maturities(self):
        """
        :return: a list of maturities of added instruments
        """
        return sorted(self.instruments.keys())

    def get_zero_rates(self):
        """
        Returns a list of spot rates on the yield curve.
        """
        self.bootstrap_zero_coupons()
        self.get_bond_spot_rates()
        return [self.zero_rates[T] for T in self.get_maturities()]

    def bootstrap_zero_coupons(self):
        """
        Bootstrap the yield curve with zero coupon instruments first.
        """
        for (T, instrument) in self.instruments.items():
            (par, coup, price, freq) = instrument
            if coup == 0:
                spot_rate = self.zero_coupon_spot_rate(par, price, T)
                self.zero_rates[T] = spot_rate

    def zero_coupon_spot_rate(self, par, price, T):
        """
        :return: the zero coupon spot rate with continuous compounding.
        """
        spot_rate = math.log(par / price) / T
        return spot_rate

    def get_bond_spot_rates(self):
        """
        Get spot rates implied by bonds, using short-term instruments.
        """
        for T in self.get_maturities():
            instrument = self.instruments[T]
            (par, coup, price, freq) = instrument
            if coup != 0:
                spot_rate = self.calculate_bond_spot_rate(T, instrument)
                self.zero_rates[T] = spot_rate

    def calculate_bond_spot_rate(self, T, instrument):
        try:
            (par, coup, price, freq) = instrument
            periods = T * freq
            value = price
            per_coupon = coup / freq
            for i in range(int(periods) - 1):
                t = (i + 1) / float(freq)
                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon * math.exp(-spot_rate * t)
                value -= discounted_coupon

            last_period = int(periods) / float(freq)
            spot_rate = -math.log(value / (par + per_coupon)) / last_period
            return spot_rate
        except:
            print("Error: spot rate not found for T=", t)


maturity_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
zero_rate_dict = {}
for i in range(10):
    zero_rate_dict[maturity_list[i]] = []

fig = plt.figure(figsize=(12, 8))
title("Zero Curve")
ylabel("Zero Rate")
xlabel("Maturity in Years")

for i in range(10):
    yield_curve = BootstrapYieldCurve()
    for bond in bonds:
        yield_curve.add_instrument(par=100,
                                   T=bond.time_to_maturity,
                                   coup=bond.coupon_rate * 100,
                                   price=bond.ten_day_price[i],
                                   compounding_freq=2)
    y = yield_curve.get_zero_rates()

    for j in range(10):
        zero_rate_dict[maturity_list[j]].append(y[j])

    plt.legend(dates, loc="lower right")
    plot(maturity_list, y)
    plt.savefig('E:/zero_curve.png')


class ForwardRates(object):

    def __init__(self):
        self.forward_rates = []
        self.spot_rates = dict()

    def add_spot_rate(self, T, spot_rate):
        self.spot_rates[T] = spot_rate

    def get_forward_rates(self):
        """
        Returns a list of forward rates
        starting from the second time period.
        """
        periods = sorted(self.spot_rates.keys())
        for T2, T1 in zip(periods, periods[1:]):
            forward_rate = self.calculate_forward_rate(T1, T2)
            self.forward_rates.append(forward_rate)

        return self.forward_rates

    def calculate_forward_rate(self, T1, T2):
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]
        forward_rate = (R2 * T2 - R1 * T1) / (T2 - T1)
        return forward_rate


fig = plt.figure(figsize=(10, 8))
title("Forward Rate Curve")
ylabel("Forward Rate")
# x_label("Years")
xlabel = ['1-1', '1-2', '1-3', '1-4']

# subset zero rate dict with odd indices only
keys_to_extract = [1.0,2.0,3.0,4.0,5.0]
subset_zero_rate_dict = {key: zero_rate_dict[key] for key in keys_to_extract}
forward_rate_dict = {}

for j in range(10):
    fr = ForwardRates()
    for i in subset_zero_rate_dict:
        fr.add_spot_rate(i, zero_rate_dict[i][j])
    forward = fr.get_forward_rates()
    forward_rate_dict[j] = forward

    plt.legend(dates, loc="lower right")
    plot(xlabel, forward)
    plt.savefig('E:/forward_curve.png')

def flatten(t):
    return [item for sublist in t for item in sublist]

yield_dict = ytm_dict.copy()
for key in ytm_dict:
    yield_dict[key] = yield_dict[key][1::2]
    yield_dict[key] = flatten(yield_dict[key])

yield_df = pandas.DataFrame.from_dict(yield_dict)
log_yield = numpy.empty(shape=45)
for i in range(5):
    yield_i = yield_df.iloc[i,:]
    for j in range(9):
        log_yield_ij = log(yield_i[j+1]/yield_i[j])
        numpy.append(log_yield, log_yield_ij)
log_yield = log_yield.reshape((5,9))
# covariance matrix of daily log-returns of yield
log_yield_cov = numpy.cov(log_yield)
# eigenvalues and eigenvectors of covariance matrix of daily log-returns of yield
print(log_yield)
yield_eigen = numpy.linalg.eig(log_yield_cov)
yield_eigenval = yield_eigen[0]
yield_eigenvec = yield_eigen[1]
print(yield_eigenval)
print(yield_eigenvec)
forward_df = pandas.DataFrame.from_dict(forward_rate_dict)
log_forward = []
for i in range(4):
    forward_i = forward_df.iloc[i,:]
    for j in range(9):
        log_forward_ij = log(forward_i[j+1]/forward_i[j])
        log_forward.append(log_forward_ij)
log_forward = numpy.asarray(log_forward).reshape((4,9))
# covariance matrix of log forward rate
log_forward_cov = numpy.cov(log_forward)
# eigenvalues and eigenvectors of covariance matrix of daily log forward rate
forward_eigen = numpy.linalg.eig(log_forward_cov)
forward_eigenval = forward_eigen[0]
forward_eigenvec = forward_eigen[1]
print(forward_eigenval)
print(forward_eigenvec)